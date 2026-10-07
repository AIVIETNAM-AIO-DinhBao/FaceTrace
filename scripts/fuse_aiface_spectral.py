"""Fuse spectral and AI-Face Global/Local artifacts by ID, with rescue/harm.

    python scripts/fuse_aiface_spectral.py --forensic-dir ... --baseline-dir ...

Validation predictions are required for selecting each fusion threshold. The
current baseline runner does not export them; this script reconstructs them from
its frozen feature_cache.pt and saved classifier checkpoints, without retraining.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.aiface_phase_e import prediction_frame, sha256_file
from src.evaluation.spectral import (
    align_predictions, save_predictions, select_validation_threshold, write_json,
)
from src.models.classifier import MLPClassifier
from src.training.spectral_trainer import predict_features


def load_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"image_id": str})


def load_baseline_validation(root: Path, branch: str, reference: pd.DataFrame):
    path = root / branch / "predictions_val.csv"
    if path.is_file():
        return align_predictions(reference, load_csv(path))
    cache_path, checkpoint_path = root / "feature_cache.pt", root / branch / "checkpoint.pt"
    if not cache_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Need {path}, OR baseline feature_cache.pt and {branch}/checkpoint.pt to reconstruct validation predictions"
        )
    # Plain tensor-only artifacts from run_aiface_phase_e.py; no backbone/token needed.
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    feature_name = branch.removesuffix("_only")
    indices = cache["split_indices"]["val"]
    cache_ids = [str(cache["image_ids"][i]) for i in indices]
    if len(set(cache_ids)) != len(cache_ids) or set(cache_ids) != set(reference.image_id):
        raise ValueError("Baseline validation cache has different/duplicate image IDs")
    positions = {value: i for value, i in zip(cache_ids, indices)}
    ordered = [positions[value] for value in reference.image_id]
    if not np.array_equal(cache["labels"][ordered].numpy(), reference.label.to_numpy()):
        raise ValueError("Baseline validation cache labels disagree")
    features = cache[feature_name][ordered]
    cfg = checkpoint["config"]["model"]
    model = MLPClassifier(features.shape[1], cfg["hidden_dim"], 2, cfg["dropout"])
    model.load_state_dict(checkpoint["classifier"])
    probabilities = predict_features(model, features, torch.device("cpu"))
    return prediction_frame(reference, probabilities, float(checkpoint["threshold"]))


def rescue_harm(baseline, candidate, baseline_threshold, candidate_threshold):
    candidate = align_predictions(baseline, candidate)
    y = baseline.label.to_numpy(dtype=int)
    baseline_ok = (baseline.probability.to_numpy() >= baseline_threshold) == y
    candidate_ok = (candidate.probability.to_numpy() >= candidate_threshold) == y
    cases = baseline[["image_id", "label", "generator_id", "split"]].copy()
    cases["baseline_probability"] = baseline.probability.to_numpy()
    cases["candidate_probability"] = candidate.probability.to_numpy()
    cases["baseline_threshold"] = baseline_threshold
    cases["candidate_threshold"] = candidate_threshold
    cases["group"] = np.select(
        [baseline_ok & candidate_ok, ~baseline_ok & candidate_ok, baseline_ok & ~candidate_ok],
        ["both_correct", "rescue", "harm"], default="both_wrong",
    )
    rows = []
    # Each generator group uses the same real test pool as metrics_by_generator.
    groups = [("overall", cases)] + [
        (generator, cases[(cases.label == 0) | (cases.generator_id == generator)])
        for generator in sorted(cases.loc[cases.label == 1, "generator_id"].unique())
    ]
    for generator, group in groups:
        counts = group.group.value_counts()
        rescue, harm = int(counts.get("rescue", 0)), int(counts.get("harm", 0))
        rows.append({"generator_id": generator, "n_images": len(group), "rescue": rescue, "harm": harm,
                     "net_gain": rescue - harm, "both_correct": int(counts.get("both_correct", 0)),
                     "both_wrong": int(counts.get("both_wrong", 0)),
                     "baseline_threshold": baseline_threshold, "candidate_threshold": candidate_threshold})
    return cases, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forensic-dir", required=True)
    parser.add_argument("--baseline-dir", required=True, help="Root containing global_only/local_only/global_local")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    forensic_root, baseline_root, output = map(Path, (args.forensic_dir, args.baseline_dir, args.output_dir))
    forensic_metadata = json.loads((forensic_root / "run_metadata.json").read_text())
    baseline_metadata = json.loads((baseline_root / "run_metadata.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    fusion_source = {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in (
        Path(__file__), REPO_ROOT / "src/evaluation/spectral.py", REPO_ROOT / "src/evaluation/aiface_phase_e.py",
        REPO_ROOT / "src/evaluation/metrics.py", REPO_ROOT / "src/training/spectral_trainer.py",
        REPO_ROOT / "src/models/classifier.py",
    )}
    if forensic_metadata.get("smoke") or baseline_metadata.get("smoke"):
        raise ValueError("Scientific fusion requires full runs; smoke predictions are not accepted")
    for metadata in (forensic_metadata, baseline_metadata):
        if metadata.get("split_mode") != "generator_disjoint_fixed":
            raise ValueError("Fusion requires generator_disjoint_fixed artifacts")
    for key in ("manifest_sha256", "generator_splits", "seed"):
        if forensic_metadata[key] != baseline_metadata[key]:
            raise ValueError(f"Global/Local and spectral runs disagree on {key}")
    spectral_config = yaml.safe_load((forensic_root / "config_resolved.yaml").read_text())
    if spectral_config["data"]["image_size"] != baseline_metadata["preprocessing"]["image_size"]:
        raise ValueError("Baseline/spectral image geometry differs")
    forensic = {split: load_csv(forensic_root / f"predictions_{split}.csv") for split in ("val", "test")}
    for split, frame in forensic.items():
        if set(frame.split) != {split}:
            raise ValueError(f"Forensic {split} prediction split mismatch")
        forensic[split] = align_predictions(frame, frame)
    baseline_val, baseline_test = {}, {}
    for branch in ("global_only", "local_only"):
        baseline_val[branch] = load_baseline_validation(baseline_root, branch, forensic["val"])
    # Lock all fusion thresholds with validation before reading test predictions.
    val_probabilities = {
        "local_forensic": (baseline_val["local_only"].probability + forensic["val"].probability) / 2,
        "global_local_forensic": (baseline_val["global_only"].probability + baseline_val["local_only"].probability + forensic["val"].probability) / 3,
    }
    thresholds = {name: select_validation_threshold(forensic["val"].label, probability)[0]
                  for name, probability in val_probabilities.items()}
    for branch in ("global_only", "local_only", "global_local"):
        baseline_test[branch] = align_predictions(forensic["test"], load_csv(baseline_root / branch / "predictions_test.csv"))
    test_probabilities = {
        "local_forensic": (baseline_test["local_only"].probability + forensic["test"].probability) / 2,
        "global_local_forensic": (baseline_test["global_only"].probability + baseline_test["local_only"].probability + forensic["test"].probability) / 3,
    }
    if not np.allclose(baseline_test["global_local"].probability,
                       (baseline_test["global_only"].probability + baseline_test["local_only"].probability) / 2):
        raise ValueError("Baseline Global+Local predictions are not equal probability mean")
    for name in thresholds:
        directory = output / name
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Fusion output exists; choose another --output-dir: {directory}")
        candidate = save_predictions(directory, forensic["test"], test_probabilities[name], thresholds[name], "test")
        save_predictions(directory, forensic["val"], val_probabilities[name], thresholds[name], "val")
        baseline_name = "local_only" if name == "local_forensic" else "global_local"
        baseline_threshold = float(json.loads((baseline_root / baseline_name / "val_metrics.json").read_text())["threshold"])
        cases, counts = rescue_harm(baseline_test[baseline_name], candidate, baseline_threshold, thresholds[name])
        cases.to_csv(directory / "error_cases.csv", index=False)
        counts.to_csv(directory / "rescue_harm.csv", index=False)
        # Also provide a common fixed 0.5 comparison, separate from validation-selected metrics.
        _, fixed_counts = rescue_harm(baseline_test[baseline_name], candidate, 0.5, 0.5)
        fixed_counts.to_csv(directory / "rescue_harm_fixed_0.5.csv", index=False)
        cfg = {"fusion": "equal_probability_mean", "threshold": thresholds[name],
               "branches": ["local_only", "forensic_only"] if name == "local_forensic" else ["global_only", "local_only", "forensic_only"],
               "threshold_selection": spectral_config["evaluation"]["threshold_selection"]}
        (directory / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        pd.DataFrame(columns=["epoch", "train_loss"]).to_csv(directory / "train_log.csv", index=False)
        torch.save({**cfg, "baseline_dir": str(baseline_root), "forensic_dir": str(forensic_root)}, directory / "checkpoint.pt")
        write_json(directory / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "gpu": None,
            "packages": {package: version(package) for package in ("torch", "numpy", "pandas", "scikit-learn", "PyYAML")},
        })
        write_json(directory / "run_metadata.json", {
            **forensic_metadata, "model_name": name, "fusion": cfg, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cpu", "threshold": thresholds[name],
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_sha256": fusion_source,
            "forensic_git_commit": forensic_metadata.get("git_commit"),
            "forensic_run_metadata_sha256": sha256_file(forensic_root / "run_metadata.json"),
            "baseline_dir": str(baseline_root), "forensic_dir": str(forensic_root),
            "baseline_run_metadata_sha256": sha256_file(baseline_root / "run_metadata.json"),
            "baseline_comparison": baseline_name, "baseline_threshold": baseline_threshold,
            "rescue_harm_threshold_policy": "each model's locked validation threshold; common 0.5 also reported",
        })
        print(f"Fusion complete: {directory}")


if __name__ == "__main__":
    main()
