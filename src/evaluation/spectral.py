"""AI-Face spectral protocol validation, ID alignment and result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.aiface_phase_e import (
    confusion_rows, metrics_by_generator, prediction_frame, validate_fixed_manifests,
)
from src.evaluation.metrics import compute_metrics


GENERATOR_SPLITS = {
    "train": {"taming_transformer_VQGAN", "stylegan3", "StableDiffusion1.5", "latent_diffusion"},
    "val": {"AttGAN", "Palette"},
    "test": {"STARGAN", "StableDiffusion_Inpainting"},
}


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_manifests(root: str | Path):
    root = Path(root).resolve()
    packaged = {split: root / split / "manifest.csv" for split in GENERATOR_SPLITS}
    if all(path.is_file() for path in packaged.values()):
        return packaged, {split: root / split for split in GENERATOR_SPLITS}
    local = {split: root / "manifests" / f"{split}.csv" for split in GENERATOR_SPLITS}
    if all(path.is_file() for path in local.values()):
        return local, {split: root for split in GENERATOR_SPLITS}
    raise FileNotFoundError(f"No complete train/val/test manifests under {root}")


def validate_pilot(manifests, bases, require_32k: bool = True):
    protocol = validate_fixed_manifests(manifests, bases)
    all_paths = set()
    for split, frame in protocol["frames"].items():
        if set(protocol["generator_splits"][split]) != GENERATOR_SPLITS[split]:
            raise ValueError(f"Unexpected AI-Face generators in {split}")
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{split} must contain binary labels 0=real, 1=fake")
        real = frame["label"] == 0
        if not (frame.loc[real, "family"] == "Real").all():
            raise ValueError("Real labels/family disagree")
        if not frame.loc[~real, "family"].isin(["GANs", "DMs"]).all():
            raise ValueError("Fake family must be GANs/DMs; deepfakes are excluded")
        if "real_source" in frame and not (frame.loc[real, "real_source"] == "imdb_wiki").all():
            raise ValueError("Unexpected real source in pilot")
        paths = [str((Path(bases[split]) / str(p)).resolve()) for p in frame["image_path"]]
        if len(set(paths)) != len(paths) or all_paths.intersection(paths):
            raise ValueError("Image path overlap within/across splits")
        all_paths.update(paths)
        if require_32k:
            expected_fake = len(GENERATOR_SPLITS[split]) * 2000
            counts = frame.loc[~real].groupby("generator_id").size()
            if len(frame) != 2 * expected_fake or int(real.sum()) != expected_fake or not (counts == 2000).all():
                raise ValueError(f"{split} counts do not match the fixed 32k pilot")
    return protocol


def smoke_frame(frame: pd.DataFrame, per_group: int = 8) -> pd.DataFrame:
    """Include real and every fake generator, in original manifest order."""
    indices = frame.index[frame.label == 0].tolist()[:per_group]
    for _, group in frame[frame.label == 1].groupby("generator_id", sort=False):
        indices.extend(group.index.tolist()[:per_group])
    return frame.loc[sorted(indices)].reset_index(drop=True)


def select_validation_threshold(labels, probabilities):
    """Same tie policy as Phase E, scored in O(N log N) rather than O(N²).

    Candidates are unique validation probabilities plus 0.5. Maximize BAcc,
    then F1, then proximity to 0.5, then the larger threshold (baseline policy).
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or set(y) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary validation arrays")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid validation probabilities")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    candidates = np.unique(np.r_[p, 0.5])
    positions = np.searchsorted(sorted_p, candidates, side="left")
    cumulative_positive = np.r_[0, np.cumsum(sorted_y)]
    positives = int(y.sum())
    negatives = len(y) - positives
    fn = cumulative_positive[positions]
    tn = positions - fn
    tp, fp = positives - fn, negatives - tn
    balanced = (tp / positives + tn / negatives) / 2
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = max(range(len(candidates)), key=lambda i: (balanced[i], f1[i], -abs(candidates[i] - 0.5), candidates[i]))
    threshold = float(candidates[best])
    return threshold, compute_metrics(y, p, threshold)


def align_predictions(reference: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "label", "generator_id", "split"}
    if required - set(incoming.columns):
        raise ValueError(f"Prediction metadata missing: {sorted(required - set(incoming.columns))}")
    incoming = incoming.copy()
    reference = reference.copy()
    incoming["image_id"] = incoming.image_id.astype(str)
    reference["image_id"] = reference.image_id.astype(str)
    if incoming.image_id.duplicated().any() or reference.image_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if set(incoming.image_id) != set(reference.image_id):
        raise ValueError("Prediction branches do not cover the same image IDs")
    aligned = incoming.set_index("image_id").loc[reference.image_id].reset_index()
    for column in ("label", "generator_id", "split"):
        if not np.array_equal(reference[column].astype(str), aligned[column].astype(str)):
            raise ValueError(f"Prediction branches disagree on {column}")
    if "probability" not in aligned and "prob_fake" in aligned:
        aligned["probability"] = aligned["prob_fake"]
    if "probability" not in aligned:
        raise ValueError("Missing probability column")
    probabilities = aligned.probability.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid prediction probabilities")
    return aligned


def save_predictions(output: Path, frame, probabilities, threshold: float, split: str):
    output.mkdir(parents=True, exist_ok=True)
    predictions = prediction_frame(frame, probabilities, threshold)
    predictions.to_csv(output / f"predictions_{split}.csv", index=False)
    if split == "test":
        overall, per_generator = metrics_by_generator(predictions, threshold)
        write_json(output / "test_metrics_overall.json", {**overall, "threshold": threshold})
        per_generator.to_csv(output / "test_metrics_by_generator.csv", index=False)
        confusion_rows(predictions).to_csv(output / "confusion_matrix.csv", index=False)
    else:
        write_json(output / "val_metrics.json", {
            **compute_metrics(frame.label, probabilities, threshold), "threshold": threshold,
        })
    return predictions
