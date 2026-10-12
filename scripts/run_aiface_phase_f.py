"""Run AI-Face Phase F robustness inference on fixed test IDs.

This entrypoint consumes full AI-Face baseline/spectral artifacts.  It never
trains, changes a checkpoint, or selects a threshold from corrupted test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_e import read_manifest, sha256_file, select_threshold  # noqa: E402
from src.evaluation.aiface_phase_f import (  # noqa: E402
    CONDITIONS, MODELS, corrupt_image, evaluate_frame, fuse_probability_frames,
    resolve_image_path, validate_prediction_alignment,
)
from src.evaluation.spectral import resolve_manifests, validate_pilot  # noqa: E402
from src.models.classifier import MLPClassifier  # noqa: E402
from src.models.dinov3 import DINOv3FeatureExtractor  # noqa: E402
from src.training.spectral_trainer import SpectralPredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--forensic-root", required=True)
    parser.add_argument("--fusion-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_branch(root: Path, branch: str, device: torch.device):
    checkpoint = torch.load(root / branch / "checkpoint.pt", map_location="cpu", weights_only=True)
    model_config = checkpoint["config"]["model"]
    input_dim = int(checkpoint.get("input_dim", 768))
    model = MLPClassifier(input_dim, model_config["hidden_dim"], 2, model_config["dropout"])
    model.load_state_dict(checkpoint["classifier"])
    return model.to(device).eval(), float(checkpoint["threshold"])


def load_test_frame(data_root: Path):
    manifests, bases = resolve_manifests(data_root)
    protocol = validate_pilot(manifests, bases, require_32k=True)
    return protocol["frames"]["test"].reset_index(drop=True), bases["test"], manifests


def reconstruct_baseline_validation(root: Path, branch: str, reference: pd.DataFrame) -> pd.DataFrame:
    """Rebuild validation probabilities from legacy full baseline artifacts.

    Older seed-42 runs may predate ``predictions_val.csv``.  Their feature cache
    and branch checkpoints are sufficient to recover validation probabilities
    without touching test labels or changing the saved run.
    """

    cache_path = root / "feature_cache.pt"
    checkpoint_path = root / branch / "checkpoint.pt"
    if not cache_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing validation predictions and fallback cache/checkpoint for {branch} under {root}"
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    indices = list(cache["split_indices"]["val"])
    ids = [str(cache["image_ids"][index]) for index in indices]
    if set(ids) != set(reference["image_id"].astype(str)):
        raise ValueError(f"Legacy validation cache IDs do not match the validation manifest for {branch}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["config"]["model"]
    feature_name = branch.removesuffix("_only")
    features = cache[feature_name][indices].float()
    model = MLPClassifier(features.shape[1], model_config["hidden_dim"], 2, model_config["dropout"])
    model.load_state_dict(checkpoint["classifier"])
    model.eval()
    with torch.inference_mode():
        probabilities = model(features).softmax(dim=1)[:, 1].numpy()
    reference = reference.copy()
    reference["image_id"] = reference["image_id"].astype(str)
    result = reference.set_index("image_id").loc[ids].reset_index()
    result["probability"] = probabilities
    result["prob_fake"] = probabilities
    return result


def load_images(frame: pd.DataFrame, base: Path, condition: str):
    images = []
    for value in frame["image_path"].astype(str):
        path = resolve_image_path(base, value)
        with Image.open(path) as image:
            images.append(corrupt_image(image, condition))
    return images


@torch.inference_mode()
def predict_dino(
    frame: pd.DataFrame,
    base: Path,
    condition: str,
    processor,
    extractor,
    heads,
    device: torch.device,
    batch_size: int,
):
    probabilities = {"global_only": [], "local_only": []}
    for start in tqdm(range(0, len(frame), batch_size), desc=f"DINO {condition}"):
        batch_frame = frame.iloc[start:start + batch_size]
        images = load_images(batch_frame, base, condition)
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        features = extractor(pixels)
        for branch, values in (("global_only", features["global"]), ("local_only", features["patches"].mean(dim=1))):
            logits = heads[branch](values.float())
            probabilities[branch].extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist())
    return {name: np.asarray(values, dtype=float) for name, values in probabilities.items()}


@torch.inference_mode()
def predict_spectral(frame, base, condition, predictor: SpectralPredictor, batch_size: int):
    values = []
    for start in tqdm(range(0, len(frame), batch_size), desc=f"Forensic {condition}"):
        images = load_images(frame.iloc[start:start + batch_size], base, condition)
        pixels = torch.stack([predictor.image_transform(image) for image in images]).to(predictor.device)
        features = predictor.fft(pixels)
        values.extend(predictor.model.predict(features).detach().cpu().tolist())
    return np.asarray(values, dtype=float)


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    result = frame[["image_id", "label", "generator_id", "split"]].copy()
    result["probability"] = probabilities
    return result


def load_thresholds(baseline_root: Path, forensic_root: Path, fusion_root: Path | None, val_reference: pd.DataFrame):
    frames = {}
    thresholds = {}
    for name, root, filename in (
        ("global_only", baseline_root, "predictions_val.csv"),
        ("local_only", baseline_root, "predictions_val.csv"),
        ("global_local", baseline_root, "predictions_val.csv"),
        ("forensic_only", forensic_root, "predictions_val.csv"),
    ):
        if name == "global_local":
            frame = fuse_probability_frames(frames, ("global_only", "local_only"))
            frames[name] = frame
            branch_dir = baseline_root / name
            metrics_path = branch_dir / "val_metrics.json"
            if metrics_path.is_file():
                thresholds[name] = float(load_json(metrics_path)["threshold"])
            else:
                thresholds[name] = select_threshold(frame["label"], frame["probability"])[0]
            continue
        branch_dir = root / name
        if name == "forensic_only":
            branch_dir = forensic_root
        path = branch_dir / filename
        if path.is_file():
            frame = pd.read_csv(path, keep_default_na=False, dtype={"image_id": str})
        elif name in {"global_only", "local_only"}:
            frame = reconstruct_baseline_validation(baseline_root, name, val_reference)
        else:
            raise FileNotFoundError(f"Missing validation predictions: {path}")
        frames[name] = validate_prediction_alignment(val_reference, frame)
        metrics_path = branch_dir / "val_metrics.json"
        if metrics_path.is_file():
            thresholds[name] = float(load_json(metrics_path)["threshold"])
        else:
            thresholds[name] = select_threshold(frames[name]["label"], frames[name]["probability"])[0]

    fused = {
        "local_forensic": (frames["local_only"].probability + frames["forensic_only"].probability) / 2,
        "global_local_forensic": (frames["global_only"].probability + frames["local_only"].probability + frames["forensic_only"].probability) / 3,
    }
    for name, probabilities in fused.items():
        if fusion_root and (fusion_root / name / "val_metrics.json").is_file():
            thresholds[name] = float(load_json(fusion_root / name / "val_metrics.json")["threshold"])
        else:
            thresholds[name] = select_threshold(val_reference["label"], probabilities)[0]
    return frames, thresholds


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    forensic_root = Path(args.forensic_root).resolve()
    fusion_root = Path(args.fusion_root).resolve() if args.fusion_root else None
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    test_frame, test_base, manifests = load_test_frame(data_root)
    val_frame = read_manifest(manifests["val"])
    baseline_metadata = load_json(baseline_root / "run_metadata.json")
    forensic_metadata = load_json(forensic_root / "run_metadata.json")
    if baseline_metadata.get("smoke") or forensic_metadata.get("smoke"):
        raise ValueError("Phase F requires full AI-Face artifacts, not smoke runs")
    for metadata in (baseline_metadata, forensic_metadata):
        if metadata.get("split_mode") != "generator_disjoint_fixed":
            raise ValueError("Phase F requires generator_disjoint_fixed artifacts")
    if baseline_metadata.get("seed") != forensic_metadata.get("seed"):
        raise ValueError("Baseline and forensic artifacts must use the same training seed")
    if baseline_metadata.get("manifest_sha256") != forensic_metadata.get("manifest_sha256"):
        raise ValueError("Baseline and forensic artifacts use different manifests")

    if args.smoke:
        selected = []
        for label in (0, 1):
            selected.extend(test_frame.index[test_frame.label.astype(int) == label].tolist()[:8])
        test_frame = test_frame.loc[sorted(set(selected))].reset_index(drop=True)

    device = select_device(args.device)
    config = yaml.safe_load((baseline_root / "config_resolved.yaml").read_text(encoding="utf-8"))
    model_name = config["model"]["backbone"]
    environment_path = baseline_root / "environment.json"
    environment = load_json(environment_path) if environment_path.is_file() else {}
    revision = (
        baseline_metadata.get("model_revision")
        or environment.get("model_revision")
        or config["model"].get("revision")
    )
    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    processor.size = {"height": int(config["data"]["image_size"]), "width": int(config["data"]["image_size"])}
    extractor = DINOv3FeatureExtractor(model_name, freeze=True, revision=revision).to(device).eval()
    heads = {name: load_branch(baseline_root, name, device) for name in ("global_only", "local_only")}
    head_models = {name: value[0] for name, value in heads.items()}
    spectral = SpectralPredictor(forensic_root / "checkpoint.pt", device=str(device))
    val_frames, thresholds = load_thresholds(baseline_root, forensic_root, fusion_root, val_frame)

    metric_rows = []
    prediction_rows = []
    for condition in args.conditions:
        dino_probabilities = predict_dino(test_frame, test_base, condition, processor, extractor, head_models, device, args.batch_size)
        forensic_probabilities = predict_spectral(test_frame, test_base, condition, spectral, args.batch_size)
        branch_frames = {
            "global_only": prediction_frame(test_frame, dino_probabilities["global_only"]),
            "local_only": prediction_frame(test_frame, dino_probabilities["local_only"]),
            "forensic_only": prediction_frame(test_frame, forensic_probabilities),
        }
        branch_frames["global_local"] = fuse_probability_frames(branch_frames, ("global_only", "local_only"))
        branch_frames["local_forensic"] = fuse_probability_frames(branch_frames, ("local_only", "forensic_only"))
        branch_frames["global_local_forensic"] = fuse_probability_frames(branch_frames, ("global_only", "local_only", "forensic_only"))
        for model in MODELS:
            frame = branch_frames[model]
            threshold = thresholds[model]
            overall, by_generator = evaluate_frame(frame, threshold)
            row = {"seed": int(baseline_metadata["seed"]), "condition": condition, "model": model,
                   "threshold": threshold, "n_images": len(frame),
                   "auroc": overall["auroc"], "balanced_accuracy": overall["balanced_accuracy"],
                   "accuracy": overall["accuracy"], "f1": overall["f1"],
                   "macro_auroc": float(by_generator.auroc.mean()),
                   "macro_balanced_accuracy": float(by_generator.balanced_accuracy.mean())}
            metric_rows.append(row)
            for generator_row in by_generator.to_dict(orient="records"):
                prediction_rows.append({"seed": int(baseline_metadata["seed"]), "condition": condition,
                                        "model": model, **generator_row})
            frame.assign(threshold=threshold, seed=int(baseline_metadata["seed"]), condition=condition, model=model).to_csv(
                output / f"predictions_{condition}_{model}.csv", index=False
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics_by_condition_model.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "metrics_by_generator.csv", index=False)
    clean = metrics[metrics.condition == "clean"].set_index("model")
    deltas = metrics.copy()
    deltas["clean_auroc"] = deltas["model"].map(clean.auroc)
    deltas["clean_balanced_accuracy"] = deltas["model"].map(clean.balanced_accuracy)
    deltas["delta_auroc"] = deltas.auroc - deltas.clean_auroc
    deltas["delta_balanced_accuracy"] = deltas.balanced_accuracy - deltas.clean_balanced_accuracy
    deltas.to_csv(output / "deltas_vs_clean.csv", index=False)
    (output / "corruption_config.yaml").write_text(yaml.safe_dump({
        "conditions": list(args.conditions), "crop": "center_crop_80", "jpeg_quality": 70,
        "resize": "112_to_224_bilinear", "blur_radius": 1.0,
        "checkpoint_policy": "clean_validation_locked", "threshold_policy": "clean_validation_locked",
    }, sort_keys=False), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({
        "dataset": baseline_metadata.get("dataset_slug"), "seed": baseline_metadata.get("seed"),
        "split_mode": "generator_disjoint_fixed", "manifest_sha256": baseline_metadata.get("manifest_sha256"),
        "test_rows": int(len(test_frame)), "smoke": bool(args.smoke), "device": str(device),
        "conditions": list(args.conditions), "models": list(MODELS), "bootstrap_not_run": True,
        "baseline_root": str(baseline_root), "forensic_root": str(forensic_root),
        "model_revision": revision,
        "fusion_root": str(fusion_root) if fusion_root else None,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"AI-Face Phase F robustness complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
