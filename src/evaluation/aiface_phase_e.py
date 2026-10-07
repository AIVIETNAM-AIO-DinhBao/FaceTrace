"""Protocol helpers for the AI-Face generator-disjoint Phase E experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "image_path",
    "label",
    "split",
    "family",
    "generator_id",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Manifest contains duplicate image_id values: {path}")
    return frame


def validate_fixed_manifests(
    manifests: dict[str, str | Path], image_bases: dict[str, str | Path]
) -> dict[str, object]:
    """Validate split ownership and image paths before any model fitting."""

    frames = {split: read_manifest(path) for split, path in manifests.items()}
    expected_splits = set(manifests)
    if expected_splits != {"train", "val", "test"}:
        raise ValueError("AI-Face Phase E requires train, val and test manifests")

    generator_sets: dict[str, set[str]] = {}
    for split, frame in frames.items():
        declared = set(frame["split"].astype(str))
        if declared != {split}:
            raise ValueError(f"Manifest {split} declares unexpected split values: {declared}")
        generator_sets[split] = {
            str(value)
            for value in frame.loc[frame["family"].astype(str) != "Real", "generator_id"]
            if str(value)
        }
        image_base = Path(image_bases[split]).resolve()
        for relative in frame["image_path"].astype(str):
            path = Path(relative)
            resolved = path if path.is_absolute() else image_base / path
            if not resolved.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {resolved}")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = generator_sets[left] & generator_sets[right]
        if overlap:
            raise ValueError(f"Generator overlap {left}/{right}: {sorted(overlap)}")

    image_ids = {split: set(frame["image_id"].astype(str)) for split, frame in frames.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = image_ids[left] & image_ids[right]
        if overlap:
            raise ValueError(f"Image ID overlap {left}/{right}: {len(overlap)} rows")

    return {
        "rows": {split: int(len(frame)) for split, frame in frames.items()},
        "generator_splits": {split: sorted(values) for split, values in generator_sets.items()},
        "manifest_sha256": {split: sha256_file(path) for split, path in manifests.items()},
        "frames": frames,
    }


def select_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[float, dict[str, float]]:
    """Select a validation threshold without looking at the held-out test set."""

    labels_array = np.asarray(list(labels), dtype=int)
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    if labels_array.size == 0 or labels_array.size != probabilities_array.size:
        raise ValueError("Threshold selection requires equally sized non-empty arrays")
    candidates = np.unique(np.concatenate(([0.5], probabilities_array)))
    scored = []
    for threshold in candidates:
        metrics = compute_metrics(labels_array, probabilities_array, float(threshold))
        scored.append((metrics["balanced_accuracy"], metrics["f1"], -abs(float(threshold) - 0.5), float(threshold), metrics))
    _, _, _, threshold, metrics = max(scored, key=lambda row: row[:4])
    return threshold, metrics


def prediction_frame(
    frame: pd.DataFrame, probabilities: Iterable[float], threshold: float
) -> pd.DataFrame:
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    if len(frame) != len(probabilities_array):
        raise ValueError("Prediction length does not match manifest")
    result = frame[["image_id", "label", "generator_id", "split"]].copy()
    result["probability"] = probabilities_array
    result["prob_fake"] = probabilities_array
    result["predicted_label"] = (probabilities_array >= threshold).astype(int)
    result["correct"] = result["predicted_label"].to_numpy() == result["label"].astype(int).to_numpy()
    return result


def metrics_by_generator(predictions: pd.DataFrame, threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate overall test and each fake generator against the shared real pool."""

    overall = compute_metrics(
        predictions["label"].to_numpy(), predictions["probability"].to_numpy(), threshold
    )
    rows = []
    fake_generators = sorted(
        value for value in predictions.loc[predictions["label"].astype(int) == 1, "generator_id"].unique()
    )
    for generator in fake_generators:
        subset = predictions[
            (predictions["label"].astype(int) == 0) | (predictions["generator_id"] == generator)
        ]
        rows.append({
            "generator_id": generator,
            "n_images": int(len(subset)),
            "n_real": int((subset["label"] == 0).sum()),
            "n_fake": int((subset["label"] == 1).sum()),
            **compute_metrics(subset["label"].to_numpy(), subset["probability"].to_numpy(), threshold),
        })
    by_generator = pd.DataFrame(rows)
    if not by_generator.empty:
        macro = {
            "auroc": float(by_generator["auroc"].mean()),
            "balanced_accuracy": float(by_generator["balanced_accuracy"].mean()),
            "accuracy": float(by_generator["accuracy"].mean()),
            "f1": float(by_generator["f1"].mean()),
        }
    else:
        macro = {key: float("nan") for key in ("auroc", "balanced_accuracy", "accuracy", "f1")}
    return {**overall, "macro": macro}, by_generator


def confusion_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"actual": actual, "predicted": predicted, "count": int(((predictions["label"] == actual) & (predictions["predicted_label"] == predicted)).sum())}
        for actual in (0, 1)
        for predicted in (0, 1)
    ])
