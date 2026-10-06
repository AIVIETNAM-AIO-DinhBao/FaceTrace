"""Run a minimal local DINOv3 smoke test on the AI-Face pilot subset.

This checks image loading, the official processor, frozen DINOv3 inference and
feature shapes. It deliberately does not train a classifier or compute metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.input_data.manifest_dataset import ManifestImageDataset
from src.models.dinov3 import DINOv3FeatureExtractor


DEFAULT_SUBSET = REPO_ROOT / "data/AI_Face/aiface_pilot_32k"
DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/aiface_smoke_test/dinov3_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return torch.device("mps")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_processor(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"height": image_size, "width": image_size}
    return processor


def check_batch(
    *,
    name: str,
    loader: DataLoader,
    extractor: DINOv3FeatureExtractor,
    device: torch.device,
    image_size: int,
) -> dict[str, object]:
    batch = next(iter(loader))
    pixel_values = batch["pixel_values"]
    if list(pixel_values.shape[1:]) != [3, image_size, image_size]:
        raise AssertionError(
            f"{name} processor output has shape {list(pixel_values.shape)}; "
            f"expected [batch, 3, {image_size}, {image_size}]"
        )
    if not torch.isfinite(pixel_values).all().item():
        raise AssertionError(f"{name} processor output contains NaN or Inf")

    with torch.no_grad():
        features = extractor(pixel_values.to(device))
    global_feature = features["global"]
    patch_feature = features["patches"]
    if global_feature.shape[0] != pixel_values.shape[0]:
        raise AssertionError(f"{name} global feature batch dimension is incorrect")
    if patch_feature.shape[0] != pixel_values.shape[0]:
        raise AssertionError(f"{name} patch feature batch dimension is incorrect")
    if not torch.isfinite(global_feature).all().item():
        raise AssertionError(f"{name} global feature contains NaN or Inf")
    if not torch.isfinite(patch_feature).all().item():
        raise AssertionError(f"{name} patch feature contains NaN or Inf")

    return {
        "manifest_rows": len(loader.dataset),
        "batch_size": int(pixel_values.shape[0]),
        "image_tensor_shape": list(pixel_values.shape),
        "global_feature_shape": list(global_feature.shape),
        "patch_feature_shape": list(patch_feature.shape),
        "labels": [int(value) for value in batch["label"].tolist()],
        "image_ids": [str(value) for value in batch["image_id"]],
        "image_paths": [str(value) for value in batch["image_path"]],
    }


def base_environment(model_name: str, device: str) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model_id": model_name,
        "device": device,
        "device_name": platform.processor(),
        "packages": {
            "Pillow": package_version("Pillow"),
            "PyYAML": package_version("PyYAML"),
        },
    }


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    subset_dir = resolve_path(args.subset_dir)
    output_path = resolve_path(args.output)
    train_manifest = subset_dir / "manifests/train.csv"
    val_manifest = subset_dir / "manifests/val.csv"
    result: dict[str, object] = {
        "protocol": {
            "subset_dir": str(subset_dir),
            "train_manifest": str(train_manifest),
            "validation_manifest": str(val_manifest),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "model_id": args.model,
            "requested_device": args.device,
        },
        "passed": False,
    }

    try:
        for path in (train_manifest, val_manifest):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest not found: {path}")
        if args.batch_size < 1:
            raise ValueError("batch-size must be at least 1")

        device = select_device(args.device)
        result["environment"] = base_environment(args.model, str(device))
        processor = load_processor(args.model, args.image_size)
        extractor = DINOv3FeatureExtractor(args.model, freeze=True).to(device)
        extractor.eval()
        if any(parameter.requires_grad for parameter in extractor.parameters()):
            raise AssertionError("DINOv3 extractor is not fully frozen")

        dataset_args = {
            "processor": processor,
            "image_column": "image_path",
            "label_column": "label",
            "image_path_base": subset_dir,
        }
        train_loader = DataLoader(
            ManifestImageDataset(train_manifest, **dataset_args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            ManifestImageDataset(val_manifest, **dataset_args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        result["feature_hidden_size"] = extractor.hidden_size
        result["train"] = check_batch(
            name="train",
            loader=train_loader,
            extractor=extractor,
            device=device,
            image_size=args.image_size,
        )
        result["validation"] = check_batch(
            name="validation",
            loader=val_loader,
            extractor=extractor,
            device=device,
            image_size=args.image_size,
        )
        result["passed"] = True
        print("DINOv3 smoke test: PASS")
    except Exception as exc:  # Keep a machine-readable failure artifact.
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        print(f"DINOv3 smoke test: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        write_result(output_path, result)
        print(f"Artifact: {output_path}")
        return 2

    write_result(output_path, result)
    print(f"Artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
