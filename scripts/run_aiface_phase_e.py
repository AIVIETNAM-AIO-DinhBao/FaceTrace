"""Run the AI-Face generator-disjoint Global/Local Phase E experiment.

This entrypoint is intentionally independent from the historical Week 3 residual
runner. It extracts Frozen DINOv3 features once, trains fixed MLP probes for the
Global and Local representations, and evaluates equal-weight late fusion.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_e import (  # noqa: E402
    confusion_rows,
    metrics_by_generator,
    prediction_frame,
    read_manifest,
    select_threshold,
    validate_fixed_manifests,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.input_data.feature_dataset import CachedFeatureDataset  # noqa: E402
from src.input_data.manifest_dataset import ManifestImageDataset  # noqa: E402
from src.models.classifier import MLPClassifier  # noqa: E402
from src.models.dinov3 import DINOv3FeatureExtractor  # noqa: E402
from src.evaluation.shortcut import run_shortcut_diagnostic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aiface_phase_e.yaml")
    parser.add_argument("--data-root", required=True, help="Unpacked AI-Face root containing train/val/test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run only two feature batches and one epoch")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolve_manifests(data_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Support both the Kaggle package layout and the local pilot layout."""

    packaged = {split: data_root / split / "manifest.csv" for split in ("train", "val", "test")}
    if all(path.is_file() for path in packaged.values()):
        return packaged, {split: data_root / split for split in packaged}
    local = {split: data_root / "manifests" / f"{split}.csv" for split in ("train", "val", "test")}
    if all(path.is_file() for path in local.values()):
        return local, {split: data_root for split in local}
    raise FileNotFoundError(
        f"Expected either train/manifest.csv layout or manifests/train.csv layout under {data_root}"
    )


def write_runtime_manifests(
    manifests: dict[str, Path], image_bases: dict[str, Path], output_dir: Path
) -> dict[str, Path]:
    runtime_dir = output_dir / "manifests"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for split, path in manifests.items():
        frame = read_manifest(path).copy()
        base = image_bases[split].resolve()
        resolved = []
        for value in frame["image_path"].astype(str):
            candidate = Path(value)
            resolved.append(str((candidate if candidate.is_absolute() else base / candidate).resolve()))
        frame["image_path"] = resolved
        out = runtime_dir / f"{split}.csv"
        frame.to_csv(out, index=False)
        output[split] = out
    return output


def load_processor(model_name: str, revision: str | None, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    processor.size = {"height": image_size, "width": image_size}
    return processor


@torch.inference_mode()
def extract_features(
    extractor: DINOv3FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    global_parts, local_parts, labels, ids = [], [], [], []
    for batch_index, batch in enumerate(tqdm(loader, desc="DINOv3 features")):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = extractor(batch["pixel_values"].to(device, non_blocking=True))
        global_parts.append(features["global"].float().cpu())
        local_parts.append(features["patches"].mean(dim=1).float().cpu())
        labels.append(batch["label"].cpu())
        ids.extend(str(value) for value in batch["image_id"])
    if not labels:
        raise ValueError("Feature loader produced no batches")
    return {
        "global": torch.cat(global_parts),
        "local": torch.cat(local_parts),
        "labels": torch.cat(labels),
        "image_ids": ids,
    }


def smoke_indices(frame: pd.DataFrame, per_class: int = 16) -> list[int]:
    """Choose a tiny but label-balanced smoke subset from a manifest."""

    indices = []
    for label in (0, 1):
        indices.extend(frame.index[frame["label"].astype(int) == label].tolist()[:per_class])
    if len(indices) < 2:
        raise ValueError("Smoke subset must contain both classes")
    return indices


def train_probe(
    features: dict[str, object],
    branch: str,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    smoke: bool,
) -> dict[str, object]:
    branch_features = features[branch]
    train_indices = features["split_indices"]["train"]
    val_indices = features["split_indices"]["val"]
    train_dataset = CachedFeatureDataset(branch_features, features["labels"], train_indices, features["image_ids"])
    val_dataset = CachedFeatureDataset(branch_features, features["labels"], val_indices, features["image_ids"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = MLPClassifier(branch_features.shape[1], hidden_dim=hidden_dim, num_classes=2, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_score = (float("-inf"), float("-inf"))
    history = []
    total_epochs = 1 if smoke else epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"{branch} epoch {epoch}", leave=False):
            values = batch["features"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        val_labels, val_probabilities = predict_model(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probabilities, 0.5)
        row = {"epoch": epoch, "train_loss": total_loss / max(count, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = (val_metrics["auroc"], val_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {branch}")
    model.load_state_dict(best_state)
    val_labels, val_probabilities = predict_model(model, val_loader, device)
    threshold, threshold_metrics = select_threshold(val_labels, val_probabilities)
    return {
        "model": model,
        "best_state": best_state,
        "history": history,
        "val_labels": val_labels,
        "val_probabilities": val_probabilities,
        "threshold": threshold,
        "val_metrics": threshold_metrics,
    }


@torch.inference_mode()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits = model(batch["features"].to(device))
        labels.append(batch["label"].cpu())
        probabilities.append(logits.softmax(dim=1)[:, 1].cpu())
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


def save_branch_artifacts(
    branch_dir: Path,
    branch_result: dict[str, object],
    test_features: dict[str, object],
    test_frame: pd.DataFrame,
    device: torch.device,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    (branch_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pd.DataFrame(branch_result["history"]).to_csv(branch_dir / "train_log.csv", index=False)
    threshold = float(branch_result["threshold"])
    val_metrics = dict(branch_result["val_metrics"])
    (branch_dir / "val_metrics.json").write_text(
        json.dumps({**val_metrics, "threshold": threshold}, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save({"classifier": branch_result["best_state"], "threshold": threshold, "config": config}, branch_dir / "checkpoint.pt")
    test_dataset = CachedFeatureDataset(
        test_features["branch"], test_features["labels"], list(range(len(test_features["labels"]))), test_features["image_ids"]
    )
    test_loader = DataLoader(test_dataset, batch_size=int(config["data"]["batch_size"]), shuffle=False)
    labels, probabilities = predict_model(branch_result["model"], test_loader, device)
    predictions = prediction_frame(test_frame, probabilities, threshold)
    overall, by_generator = metrics_by_generator(predictions, threshold)
    (branch_dir / "test_metrics_overall.json").write_text(json.dumps(overall, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    by_generator.to_csv(branch_dir / "test_metrics_by_generator.csv", index=False)
    predictions.to_csv(branch_dir / "predictions_test.csv", index=False)
    confusion_rows(predictions).to_csv(branch_dir / "confusion_matrix.csv", index=False)
    (branch_dir / "run_metadata.json").write_text(
        json.dumps({"branch": branch_dir.name, "threshold": threshold, "feature": branch_dir.name.replace("_only", "")}, indent=2) + "\n",
        encoding="utf-8",
    )
    return predictions, threshold


def main() -> int:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    manifests, image_bases = resolve_manifests(data_root)
    protocol = validate_fixed_manifests(manifests, image_bases)
    runtime_manifests = write_runtime_manifests(manifests, image_bases, output_dir)
    print(f"Dataset root: {data_root}")
    print(f"Split mode: generator_disjoint_fixed")
    print(f"train generators: {protocol['generator_splits']['train']}")
    print(f"val generators: {protocol['generator_splits']['val']}")
    print(f"test generators: {protocol['generator_splits']['test']}")
    print(f"Held-out test manifest: {runtime_manifests['test']}")
    print("Random fallback: disabled")

    device = select_device(config["training"].get("device", "auto"))
    model_name = config["model"]["backbone"]
    processor = load_processor(model_name, config["model"].get("revision"), int(config["data"]["image_size"]))
    extractor = DINOv3FeatureExtractor(model_name, freeze=True, revision=config["model"].get("revision")).to(device).eval()
    if any(parameter.requires_grad for parameter in extractor.parameters()):
        raise RuntimeError("DINOv3 backbone is not frozen")

    datasets = {}
    loaders = {}
    frames = {}
    for split in ("train", "val", "test"):
        frames[split] = read_manifest(runtime_manifests[split])
        datasets[split] = ManifestImageDataset(runtime_manifests[split], processor=processor, image_path_base=Path("/"))
        dataset = datasets[split]
        loader_dataset = dataset
        if args.smoke:
            from torch.utils.data import Subset
            loader_dataset = Subset(dataset, smoke_indices(frames[split]))
        loaders[split] = DataLoader(
            loader_dataset, batch_size=int(config["data"]["batch_size"]), shuffle=False,
            num_workers=int(config["data"].get("num_workers", 0)),
            pin_memory=bool(config["training"].get("pin_memory", True) and device.type == "cuda"),
        )
    max_batches = None
    extracted = {split: extract_features(extractor, loaders[split], device, max_batches) for split in ("train", "val", "test")}
    if args.smoke:
        frames = {
            split: frames[split].set_index("image_id").loc[extracted[split]["image_ids"]].reset_index()
            for split in ("train", "val", "test")
        }
    if args.smoke:
        print("Phase E smoke feature pass: PASS")
    all_features = {
        "global": torch.cat([extracted[split]["global"] for split in ("train", "val", "test")]),
        "local": torch.cat([extracted[split]["local"] for split in ("train", "val", "test")]),
        "labels": torch.cat([extracted[split]["labels"] for split in ("train", "val", "test")]),
        "image_ids": sum((extracted[split]["image_ids"] for split in ("train", "val", "test")), []),
        "split_indices": {},
    }
    cursor = 0
    for split in ("train", "val", "test"):
        size = len(extracted[split]["labels"])
        all_features["split_indices"][split] = list(range(cursor, cursor + size))
        cursor += size
    torch.save(all_features, output_dir / "feature_cache.pt")

    results = {}
    for branch in ("global", "local"):
        result = train_probe(
            all_features, branch, seed, device,
            epochs=int(config["training"]["epochs"]),
            batch_size=int(config["data"]["batch_size"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            dropout=float(config["model"]["dropout"]),
            smoke=args.smoke,
        )
        results[branch] = result

    test_frame = frames["test"]
    branch_predictions = {}
    for branch in ("global", "local"):
        test_features = {
            "branch": all_features[branch][all_features["split_indices"]["test"][0]:all_features["split_indices"]["test"][-1] + 1],
            "labels": all_features["labels"][all_features["split_indices"]["test"][0]:all_features["split_indices"]["test"][-1] + 1],
            "image_ids": extracted["test"]["image_ids"],
        }
        branch_predictions[branch], _ = save_branch_artifacts(
            output_dir / f"{branch}_only", results[branch], test_features, test_frame, device, config
        )

    val_predictions = {}
    for branch in ("global", "local"):
        indices = all_features["split_indices"]["val"]
        val_predictions[branch] = prediction_frame(
            frames["val"], results[branch]["val_probabilities"], float(results[branch]["threshold"])
        )
    fusion_val_probability = (val_predictions["global"]["probability"] + val_predictions["local"]["probability"]) / 2
    fusion_threshold, fusion_val_metrics = select_threshold(frames["val"]["label"], fusion_val_probability)
    fusion_probability = (branch_predictions["global"]["probability"] + branch_predictions["local"]["probability"]) / 2
    fusion_predictions = prediction_frame(test_frame, fusion_probability, fusion_threshold)
    fusion_dir = output_dir / "global_local"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    (fusion_dir / "val_metrics.json").write_text(json.dumps({**fusion_val_metrics, "threshold": fusion_threshold}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fusion_overall, fusion_by_generator = metrics_by_generator(fusion_predictions, fusion_threshold)
    (fusion_dir / "test_metrics_overall.json").write_text(json.dumps(fusion_overall, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fusion_by_generator.to_csv(fusion_dir / "test_metrics_by_generator.csv", index=False)
    fusion_predictions.to_csv(fusion_dir / "predictions_test.csv", index=False)
    confusion_rows(fusion_predictions).to_csv(fusion_dir / "confusion_matrix.csv", index=False)
    torch.save({"branches": ["global_only", "local_only"], "fusion": "equal_probability_mean", "threshold": fusion_threshold}, fusion_dir / "checkpoint.pt")
    (fusion_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "model_id": model_name, "model_revision": getattr(extractor.backbone.config, "_commit_hash", None),
        "packages": {name: package_version(name) for name in ("torch", "transformers", "numpy", "pandas", "Pillow", "scikit-learn", "PyYAML")},
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "dataset_slug": config["experiment"].get("dataset_slug"), "split_mode": "generator_disjoint_fixed",
        "generator_splits": protocol["generator_splits"], "manifest_sha256": protocol["manifest_sha256"],
        "seed": seed, "device": str(device), "preprocessing": {"image_size": config["data"]["image_size"], "processor": model_name},
        "checkpoint_selection": "validation AUROC then validation Balanced Accuracy",
        "threshold_selection": config["evaluation"]["threshold_selection"], "fusion": "mean(global_probability, local_probability)",
        "test_rows": int(len(test_frame)), "smoke": bool(args.smoke),
        "kaggle_url": os.environ.get("KAGGLE_URL"),
        "kaggle_run_id": os.environ.get("KAGGLE_KERNEL_RUN_ID"),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    shortcut_dir = output_dir / "shortcut"
    run_shortcut_diagnostic(runtime_manifests["train"], runtime_manifests["val"], Path("/"), shortcut_dir)
    print(f"Phase E artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
