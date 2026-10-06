"""Train the Week 3 residual and matched RGB-control models.

Example:
    python3 scripts/train_residual.py --config configs/week03.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make the entry point work when invoked as ``python scripts/train_residual.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.input_data.residual_dataset import ResidualImageDataset
from src.models.residual import ResidualImageModel, trainable_parameter_count
from src.training.residual_trainer import evaluate_image_model, fit_image_model, write_predictions


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


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def package_versions() -> dict[str, str | None]:
    result = {}
    for package_name in ("torch", "numpy", "pandas", "Pillow", "scikit-learn", "PyYAML"):
        try:
            result[package_name] = version(package_name)
        except PackageNotFoundError:
            result[package_name] = None
    return result


def train_branch(
    branch: str,
    seed: int,
    config: dict,
    repo_root: Path,
    device: torch.device,
    train_dataset: ResidualImageDataset,
    val_dataset: ResidualImageDataset,
    run_dir: Path,
) -> dict:
    data_cfg, model_cfg, train_cfg, eval_cfg = (
        config["data"], config["model"], config["training"], config["evaluation"]
    )
    set_reproducibility(seed)
    model = ResidualImageModel(
        use_residual=branch == "residual_only",
        kernel_size=int(model_cfg["residual_kernel_size"]),
        residual_gain=float(model_cfg["residual_gain"]),
        input_channels=int(model_cfg["image_channels"]),
        num_classes=int(model_cfg["num_classes"]),
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    pin_memory = bool(train_cfg.get("pin_memory", True) and device.type == "cuda")
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
    branch_dir = run_dir / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    best_state, best_epoch, best_metrics, history = fit_image_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        threshold=float(eval_cfg["threshold"]),
        mixed_precision=bool(train_cfg.get("mixed_precision", False)),
        history_path=branch_dir / "history.csv",
    )
    metrics, prediction = evaluate_image_model(model, val_loader, device, float(eval_cfg["threshold"]))
    checkpoint = {
        "model_state": best_state,
        "model_type": "ResidualImageModel",
        "branch": branch,
        "use_residual": model.use_residual,
        "residual_kernel_size": model.kernel_size,
        "residual_gain": model.residual_gain,
        "image_size": int(data_cfg["image_size"]),
        "threshold": float(eval_cfg["threshold"]),
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
        "trainable_parameters": trainable_parameter_count(model),
        "train_manifest": str(Path(data_cfg["train_manifest"])),
        "validation_manifest": str(Path(data_cfg["validation_manifest"])),
    }
    torch.save(checkpoint, branch_dir / "best_checkpoint.pt")
    write_predictions(prediction, branch_dir / "validation_predictions.csv", float(eval_cfg["threshold"]))
    (branch_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (branch_dir / "completion.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "branch": branch,
                "seed": seed,
                "best_epoch": best_epoch,
                "metrics": metrics,
                "history_rows": len(history),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"branch": branch, "seed": seed, "best_epoch": best_epoch, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week03.yaml")
    args = parser.parse_args()
    repo_root = REPO_ROOT
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    train_manifest = (repo_root / data_cfg["train_manifest"]).resolve()
    val_manifest = (repo_root / data_cfg["validation_manifest"]).resolve()
    for path in (train_manifest, val_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/create_splits.py first.")

    device = select_device(config["training"].get("device", "auto"))
    if device.type == "cpu":
        print("WARNING: no accelerator detected; image training will be slow.")
    print(f"Python={platform.python_version()} torch={torch.__version__} device={device}")
    image_path_base = (repo_root / data_cfg.get("image_path_base", ".")).resolve()
    dataset_args = {
        "image_size": int(data_cfg["image_size"]),
        "image_column": data_cfg.get("image_column", "image_path"),
        "label_column": data_cfg.get("label_column", "label"),
        "image_path_base": image_path_base,
    }
    train_dataset = ResidualImageDataset(train_manifest, **dataset_args)
    val_dataset = ResidualImageDataset(val_manifest, **dataset_args)
    output_dir = repo_root / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in config["experiment"].get("seeds", [42]):
        run_dir = output_dir / f"seed{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "packages": package_versions(),
                    "torch_version": torch.__version__,
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                    "seed": int(seed),
                    "train_manifest": str(train_manifest),
                    "validation_manifest": str(val_manifest),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for branch in ("residual_only", "rgb_control"):
            results.append(train_branch(branch, int(seed), config, repo_root, device, train_dataset, val_dataset, run_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote Week 3 residual results to {output_dir}")


if __name__ == "__main__":
    main()
