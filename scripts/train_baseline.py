"""Train Frozen DINOv3 global-feature linear and MLP probes."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from src.input_data.manifest_dataset import ManifestImageDataset
from src.models.classifier import LinearClassifier, MLPClassifier
from src.models.dinov3 import DINOv3FeatureExtractor
from src.models.probe import GlobalProbe
from src.training.trainer import fit_probe


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def path_for_artifact(path: Path, base: Path) -> str:
    """Use portable relative paths when possible, absolute paths for mounted data."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_processor(model_name: str, revision: str | None, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    # Keep the checkpoint's resize/rescale/normalization settings; enforce the
    # input resolution fixed in the experiment config.
    processor.size = {"height": image_size, "width": image_size}
    return processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    repo_root = Path(__file__).resolve().parents[1]
    experiment, data_cfg = config["experiment"], config["data"]
    model_cfg, train_cfg = config["model"], config["training"]
    eval_cfg = config["evaluation"]
    if not model_cfg.get("freeze_backbone", False):
        parser.error("Phase 3 requires the DINOv3 backbone to remain frozen.")

    train_manifest = (repo_root / data_cfg["train_manifest"]).resolve()
    val_manifest = (repo_root / data_cfg["validation_manifest"]).resolve()
    for path in (train_manifest, val_manifest):
        if not path.is_file():
            parser.error(f"Manifest not found: {path}. Run scripts/create_splits.py first.")

    device = select_device(train_cfg.get("device", "auto"))
    if device.type == "cpu":
        print("WARNING: no accelerator detected; Frozen DINOv3 training on CPU will be slow.")
    print(f"Python={platform.python_version()} torch={torch.__version__} "
          f"transformers={transformers.__version__} device={device}")

    model_name = model_cfg["backbone"]
    revision = model_cfg.get("revision")
    try:
        processor = load_processor(model_name, revision, int(data_cfg["image_size"]))
        extractor = DINOv3FeatureExtractor(model_name, freeze=True, revision=revision)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load {model_name!r}. This official Hugging Face checkpoint is gated: "
            "accept Meta's DINOv3 license and request/access the model on Hugging Face, "
            "then authenticate with Hugging Face CLI (hf auth login) or provide a permitted local checkpoint."
        ) from exc

    extractor.to(device)
    extractor.eval()
    image_path_base = repo_root / data_cfg.get("image_path_base", ".")
    common_dataset_args = {
        "processor": processor,
        "image_column": data_cfg.get("image_column", "image_path"),
        "label_column": data_cfg.get("label_column", "label"),
        "image_path_base": image_path_base,
    }
    train_dataset = ManifestImageDataset(train_manifest, **common_dataset_args)
    val_dataset = ManifestImageDataset(val_manifest, **common_dataset_args)
    pin_memory = bool(train_cfg.get("pin_memory", True) and device.type == "cuda")

    run_dir = repo_root / experiment["output_dir"] / f"{experiment['name']}_seed{experiment['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "processor.json").write_text(
        json.dumps(processor.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    package_names = ["torch", "torchvision", "transformers", "numpy", "pandas", "Pillow", "scikit-learn", "PyYAML"]
    package_versions = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None
    environment = {
        "python": platform.python_version(),
        "packages": package_versions,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "model_id": model_name,
        "model_revision": revision,
        "resolved_model_commit": getattr(extractor.backbone.config, "_commit_hash", None),
        "train_manifest": path_for_artifact(train_manifest, repo_root),
        "validation_manifest": path_for_artifact(val_manifest, repo_root),
        "seed": int(experiment["seed"]),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    head_factories = {
        "linear": lambda: LinearClassifier(
            extractor.hidden_size, num_classes=int(model_cfg["num_classes"])
        ),
        "mlp": lambda: MLPClassifier(
            extractor.hidden_size,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_classes=int(model_cfg["num_classes"]),
            dropout=float(model_cfg["dropout"]),
        ),
    }
    results = {}
    for head_name, make_head in head_factories.items():
        set_reproducibility(int(experiment["seed"]))
        generator = torch.Generator().manual_seed(int(experiment["seed"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
        )
        model = GlobalProbe(extractor, make_head()).to(device)
        head_dir = run_dir / head_name
        head_dir.mkdir(parents=True, exist_ok=True)
        best_state, best_epoch, best_metrics, history = fit_probe(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            device=device,
            epochs=int(train_cfg["epochs"]),
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            threshold=float(eval_cfg["threshold"]),
            mixed_precision=bool(train_cfg.get("mixed_precision", False)),
            history_path=head_dir / "history.csv",
        )
        model.classifier.load_state_dict(best_state)
        torch.save(
            {
                "classifier": model.classifier.state_dict(),
                "classifier_name": head_name,
                "backbone": model_name,
                "resolved_model_commit": environment["resolved_model_commit"],
                "processor": processor.to_dict(),
                "image_size": int(data_cfg["image_size"]),
                "threshold": float(eval_cfg["threshold"]),
                "seed": int(experiment["seed"]),
                "best_epoch": best_epoch,
                "validation_metrics": best_metrics,
                "train_manifest": environment["train_manifest"],
                "validation_manifest": environment["validation_manifest"],
                "classifier_config": {
                    "hidden_dim": int(model_cfg["hidden_dim"]),
                    "dropout": float(model_cfg["dropout"]),
                    "num_classes": int(model_cfg["num_classes"]),
                },
            },
            head_dir / "best_classifier.pt",
        )
        results[head_name] = {
            "best_epoch": best_epoch,
            "metrics": best_metrics,
            "history": history,
            "checkpoint": path_for_artifact(head_dir / "best_classifier.pt", repo_root),
        }

    selected = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["val_auroc"],
            results[name]["metrics"]["val_balanced_accuracy"],
        ),
    )
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "selected_head.json").write_text(
        json.dumps(
            {
                "selected_head": selected,
                "selection_rule": "max validation AUROC, then validation Balanced Accuracy",
                "metrics": results[selected]["metrics"],
                "warning": "In-domain validation only; no unseen-source test is available.",
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Selected head: {selected} — validation AUROC "
          f"{results[selected]['metrics']['val_auroc']:.5f}; run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
