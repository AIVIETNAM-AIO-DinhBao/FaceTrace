"""Train the AI-Face radial FFT + MLP baseline on the existing fixed split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from torch.utils.data import Subset
import yaml

from src.evaluation.aiface_phase_e import sha256_file
from src.evaluation.spectral import (
    resolve_manifests, save_predictions, smoke_frame, validate_pilot, write_json,
)
from src.input_data.spectral_dataset import SpectralDataset
from src.models.spectral import RadialFFT, parameter_count
from src.training.spectral_trainer import (
    extract_spectral_features, fit_spectral_probe, predict_features, select_device,
)


def validate_config(config):
    expected = {
        "data": {"resize": "square_bilinear_antialias", "crop": "none", "pixel_range": [0.0, 1.0]},
        "model": {"window": "hann_symmetric", "fft_norm": "backward", "spectrum": "log1p_magnitude",
                  "radial_range": "zero_to_corner", "channel_mode": "rgb_separate",
                  "feature_normalization": "train_mean_std"},
        "evaluation": {"checkpoint_selection": "validation_auroc_then_balanced_accuracy_at_0.5",
                       "threshold_selection": "validation_balanced_accuracy_then_f1_then_closest_to_0.5",
                       "fusion": "equal_probability_mean"},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if config[section][key] != value:
                raise ValueError(f"Unsupported {section}.{key}; expected {value!r}")
    for section, keys in {"data": ["image_size", "batch_size", "smoke_per_group"],
                          "training": ["epochs", "batch_size"]}.items():
        if any(config[section][key] < 1 for key in keys):
            raise ValueError(f"Positive dimensions/epochs required in {section}")
    if config["data"]["require_32k"] and config["data"]["image_size"] != 224:
        raise ValueError("Full AI-Face protocol uses 224x224, matching baseline config")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/spectral.yaml"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.device:
        config["training"]["device"] = args.device
    output = Path(args.output_dir or config["experiment"]["output_dir"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if args.smoke:
        output = output.with_name(output.name + "_smoke")
        config["training"]["epochs"] = 1
    if (output / "checkpoint.pt").exists():
        raise FileExistsError(f"Existing run checkpoint: {output}. Use another --output-dir to preserve it.")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root).resolve()
    manifests, bases = resolve_manifests(root)
    protocol = validate_pilot(manifests, bases, config["data"]["require_32k"])
    metadata_path = root / "subset_metadata.json"
    if config["data"]["require_32k"] and not metadata_path.is_file():
        raise FileNotFoundError(f"Required subset_metadata.json is missing under {root}")
    subset_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    print("Split mode: generator_disjoint_fixed; Random fallback: disabled", flush=True)
    for split in ("train", "val", "test"):
        print(f"{split} generators: {protocol['generator_splits'][split]}", flush=True)
    print(f"Held-out test manifest: {manifests['test']}", flush=True)
    print(f"Smoke: {args.smoke}; HF_TOKEN/DINOv3 weights: not required", flush=True)
    frames, datasets = {}, {}
    for split in ("train", "val", "test"):
        dataset = SpectralDataset(manifests[split], bases[split], config["data"]["image_size"])
        frame = dataset.frame
        if args.smoke:
            selected = smoke_frame(frame, config["data"]["smoke_per_group"])
            indices = frame.index[frame.image_id.isin(selected.image_id)].tolist()
            dataset = Subset(dataset, indices)
            frame = selected
        frames[split], datasets[split] = frame, dataset
    frame_dir = output / "manifests"
    frame_dir.mkdir(exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(frame_dir / f"{split}.csv", index=False)
    device = select_device(config["training"]["device"])
    fft = RadialFFT(config["data"]["image_size"], config["model"]["radial_bins"])
    cache = {}

    def extract(split):
        result = extract_spectral_features(datasets[split], fft, device,
                    config["data"]["batch_size"], config["data"]["num_workers"])
        if result["image_ids"] != frames[split].image_id.astype(str).tolist():
            raise ValueError(f"{split} feature IDs differ from manifest order")
        torch.save(result, output / f"features_{split}.pt")
        return result

    # Never fit normalization/checkpoint/threshold on test features or labels.
    for split in ("train", "val"):
        cache[split] = extract(split)
    result = fit_spectral_probe(cache["train"], cache["val"], config["model"], config["training"],
                                config["experiment"]["seed"], device)
    commit = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "status", "--porcelain"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    config["resolved"] = {
        "feature_dim": fft.output_dim, "trainable_parameters": parameter_count(result["model"]),
        "radial_edges_pixels": fft.radial_edges.cpu().tolist(),
        "radial_binning": RadialFFT.BINNING_VERSION,
        "best_epoch": result["best_epoch"], "threshold": result["threshold"], "smoke": args.smoke,
    }
    metadata = {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in (
            REPO_ROOT / "src/models/spectral.py", REPO_ROOT / "src/input_data/spectral_dataset.py",
            REPO_ROOT / "src/training/spectral_trainer.py", REPO_ROOT / "src/evaluation/spectral.py",
            Path(__file__), REPO_ROOT / "src/evaluation/aiface_phase_e.py", REPO_ROOT / "src/evaluation/metrics.py",
        )},
        "dataset_slug": config["experiment"]["dataset_slug"],
        "dataset_version": config["experiment"].get("dataset_version"),
        "manifest_sha256": protocol["manifest_sha256"],
        "subset_metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        "subset_metadata": subset_metadata, "generator_splits": protocol["generator_splits"],
        "split_mode": "generator_disjoint_fixed", "seed": config["experiment"]["seed"],
        "device": str(device), "smoke": args.smoke,
        "rows_used": {split: len(frame) for split, frame in frames.items()},
        "used_manifest_sha256": {split: sha256_file(frame_dir / f"{split}.csv") for split in frames},
        "preprocessing": config["data"], "checkpoint_selection": config["evaluation"]["checkpoint_selection"],
        "radial_binning": RadialFFT.BINNING_VERSION,
        "threshold_selection": config["evaluation"]["threshold_selection"],
        "fusion": config["evaluation"]["fusion"], "threshold": result["threshold"],
        "test_evaluation_policy": "after checkpoint and validation threshold are locked",
        "limitation": "imdb_wiki real source; identity overlap remains; not identity-disjoint",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_url": os.environ.get("KAGGLE_URL"), "kaggle_run_id": os.environ.get("KAGGLE_KERNEL_RUN_ID"),
    }
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pd.DataFrame(result["history"]).to_csv(output / "train_log.csv", index=False)
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pandas", "Pillow", "scikit-learn", "PyYAML")},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    torch.save({"state_dict": result["state_dict"], "config": config, "threshold": result["threshold"],
                "best_epoch": result["best_epoch"], "run_metadata": metadata}, output / "checkpoint.pt")
    save_predictions(output, frames["val"], result["val_probabilities"], result["threshold"], "val")
    print(f"Checkpoint/threshold locked: epoch={result['best_epoch']}, threshold={result['threshold']}", flush=True)
    cache["test"] = extract("test")
    probabilities = predict_features(result["model"], cache["test"]["features"], device)
    save_predictions(output, frames["test"], probabilities, result["threshold"], "test")
    if args.smoke:
        write_json(output / "smoke_test.json", {"passed": True, "rows": metadata["rows_used"],
                    "prediction_id_order_verified": True, "scientific_result": False})
    print(f"Forensic-only {'SMOKE PASS' if args.smoke else 'complete'}: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
