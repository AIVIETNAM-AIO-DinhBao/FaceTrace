"""Evaluation helpers for the AI-Face Phase F protocol.

This module is intentionally independent from the historical Week 2/3
Who-Is-AI and Residual runners.  It contains only deterministic image
corruptions, prediction alignment, bootstrap uncertainty and validation-only
model selection for the AI-Face generator-disjoint pilot.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import roc_auc_score

from src.evaluation.aiface_phase_e import metrics_by_generator, select_threshold
from src.evaluation.metrics import compute_metrics


CONDITIONS = ("clean", "jpeg_q70", "resize_112_224", "blur_r1", "center_crop_80")
MODELS = (
    "global_only",
    "local_only",
    "forensic_only",
    "global_local",
    "local_forensic",
    "global_local_forensic",
)


def _rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def corrupt_image(image: Image.Image, condition: str, output_size: int = 224) -> Image.Image:
    """Apply one deterministic corruption to a clean RGB image.

    Every condition starts from the original decoded image.  Corruptions are
    never chained, and the returned image is a fresh object.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown AI-Face Phase F condition: {condition}")
    rgb = _rgb(image)
    if condition == "clean":
        return rgb.copy()
    if condition == "jpeg_q70":
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=70, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB").copy()
    if condition == "resize_112_224":
        down = rgb.resize((112, 112), Image.Resampling.BILINEAR)
        return down.resize((output_size, output_size), Image.Resampling.BILINEAR)
    if condition == "blur_r1":
        return rgb.filter(ImageFilter.GaussianBlur(radius=1.0))

    side = int(0.8 * min(rgb.size))
    if side < 2:
        raise ValueError(f"Image is too small for center_crop_80: {rgb.size}")
    left = (rgb.width - side) // 2
    top = (rgb.height - side) // 2
    cropped = rgb.crop((left, top, left + side, top + side))
    return cropped.resize((output_size, output_size), Image.Resampling.BILINEAR)


def resolve_image_path(base: str | Path, value: str | Path) -> Path:
    path = Path(str(value))
    resolved = path if path.is_absolute() else Path(base) / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"AI-Face image does not exist: {resolved}")
    return resolved


def validate_prediction_alignment(reference: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Align a prediction frame and reject metadata or ID mismatches."""

    required = {"image_id", "label", "generator_id", "split"}
    missing = required - set(incoming.columns)
    if missing:
        raise ValueError(f"Prediction metadata missing: {sorted(missing)}")
    left = reference.copy()
    right = incoming.copy()
    left["image_id"] = left["image_id"].astype(str)
    right["image_id"] = right["image_id"].astype(str)
    if left["image_id"].duplicated().any() or right["image_id"].duplicated().any():
        raise ValueError("Prediction IDs must be unique")
    if set(left.image_id) != set(right.image_id):
        raise ValueError("Prediction IDs do not match the reference manifest")
    aligned = right.set_index("image_id").loc[left.image_id].reset_index()
    for column in ("label", "generator_id", "split"):
        if not np.array_equal(left[column].astype(str), aligned[column].astype(str)):
            raise ValueError(f"Prediction metadata disagrees on {column}")
    probability_column = "probability" if "probability" in aligned else "prob_fake"
    if probability_column not in aligned:
        raise ValueError("Prediction frame needs probability or prob_fake")
    aligned["probability"] = aligned[probability_column].astype(float)
    values = aligned["probability"].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Prediction probabilities must be finite values in [0, 1]")
    return aligned


def fuse_probability_frames(frames: Mapping[str, pd.DataFrame], names: Iterable[str]) -> pd.DataFrame:
    """Mean branch probabilities after strict ID/label alignment."""

    names = list(names)
    if not names:
        raise ValueError("At least one branch is required for fusion")
    reference = frames[names[0]].copy()
    aligned = [validate_prediction_alignment(reference, frames[name]) for name in names]
    result = reference[["image_id", "label", "generator_id", "split"]].copy()
    result["probability"] = np.mean([frame["probability"].to_numpy() for frame in aligned], axis=0)
    return result


def evaluate_frame(frame: pd.DataFrame, threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    """Return overall and shared-real-pool per-generator metrics."""

    probabilities = frame["probability"].to_numpy(dtype=float)
    result = frame.copy()
    result["predicted_label"] = (probabilities >= threshold).astype(int)
    result["correct"] = result["predicted_label"].to_numpy() == result["label"].astype(int).to_numpy()
    overall, by_generator = metrics_by_generator(result, threshold)
    return overall, by_generator


def _bootstrap_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Sample real pool once and each fake generator independently."""

    sampled: list[int] = []
    labels = frame["label"].astype(int)
    generators = frame["generator_id"].astype(str)
    real_indices = frame.index[labels == 0].to_numpy()
    if not len(real_indices):
        raise ValueError("Bootstrap requires real test images")
    sampled.extend(rng.choice(real_indices, size=len(real_indices), replace=True).tolist())
    for generator in sorted(generators[labels == 1].unique()):
        fake_indices = frame.index[(labels == 1) & (generators == generator)].to_numpy()
        if not len(fake_indices):
            continue
        sampled.extend(rng.choice(fake_indices, size=len(fake_indices), replace=True).tolist())
    return np.asarray(sampled, dtype=int)


def paired_bootstrap(
    frames: Mapping[str, pd.DataFrame],
    metric: str = "macro_auroc",
    comparisons: Iterable[tuple[str, str]] = (),
    replicates: int = 2000,
    seed: int = 2026,
    threshold_by_model: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute paired percentile CIs using shared generator/class samples.

    ``frames`` must be aligned test prediction frames.  Thresholds are fixed
    inputs from clean validation; bootstrap never selects a threshold.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    threshold_by_model = dict(threshold_by_model or {name: 0.5 for name in frames})
    names = list(frames)
    reference = frames[names[0]]
    aligned = {name: validate_prediction_alignment(reference, frame) for name, frame in frames.items()}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in names}
    differences: dict[str, list[float]] = {
        f"{candidate}-{baseline}": [] for baseline, candidate in comparisons
    }
    point: dict[str, float] = {}

    labels = reference["label"].to_numpy(dtype=int)
    generators = reference["generator_id"].astype(str).to_numpy()
    real_indices = np.flatnonzero(labels == 0)
    fake_indices = {
        generator: np.flatnonzero((labels == 1) & (generators == generator))
        for generator in sorted(set(generators[labels == 1]))
    }

    def score_arrays(probabilities, sampled_real, sampled_fake, threshold):
        real_scores = probabilities[sampled_real]
        fake_scores = [probabilities[sampled_fake[generator]] for generator in fake_indices]
        if metric == "overall_auroc":
            scores = np.concatenate([real_scores, *fake_scores])
            target = np.concatenate([
                np.zeros(len(real_scores), dtype=int),
                *[np.ones(len(values), dtype=int) for values in fake_scores],
            ])
            return float(roc_auc_score(target, scores))
        if metric == "overall_balanced_accuracy":
            fake_scores_all = np.concatenate(fake_scores)
            true_negative_rate = float((real_scores < threshold).mean())
            true_positive_rate = float((fake_scores_all >= threshold).mean())
            return (true_negative_rate + true_positive_rate) / 2.0
        values = []
        for scores in fake_scores:
            if metric == "macro_auroc":
                target = np.concatenate([
                    np.zeros(len(real_scores), dtype=int),
                    np.ones(len(scores), dtype=int),
                ])
                values.append(float(roc_auc_score(target, np.concatenate([real_scores, scores]))))
            else:
                true_negative_rate = float((real_scores < threshold).mean())
                true_positive_rate = float((scores >= threshold).mean())
                values.append((true_negative_rate + true_positive_rate) / 2.0)
        return float(np.mean(values)) if values else float("nan")

    def score_frame(frame: pd.DataFrame, threshold: float) -> float:
        # Preserve the public helper's behavior for unsupported metric names.
        if metric not in {"overall_auroc", "overall_balanced_accuracy", "macro_auroc", "macro_balanced_accuracy"}:
            overall, by_generator = evaluate_frame(frame, threshold)
            if metric == "overall_auroc":
                return float(overall["auroc"])
            key = "auroc" if metric == "macro_auroc" else "balanced_accuracy"
            return float(by_generator[key].mean()) if not by_generator.empty else float("nan")
        probabilities = frame["probability"].to_numpy(dtype=float)
        sampled_fake = {generator: fake_indices[generator] for generator in fake_indices}
        return score_arrays(probabilities, real_indices, sampled_fake, threshold)

    for name, frame in aligned.items():
        point[name] = score_frame(frame, threshold_by_model[name])
    for baseline, candidate in comparisons:
        point[f"{candidate}-{baseline}"] = point[candidate] - point[baseline]

    for _ in range(replicates):
        sampled_real = rng.choice(real_indices, size=len(real_indices), replace=True)
        sampled_fake = {
            generator: rng.choice(indices, size=len(indices), replace=True)
            for generator, indices in fake_indices.items()
        }
        scored = {
            name: score_arrays(
                frame["probability"].to_numpy(dtype=float),
                sampled_real,
                sampled_fake,
                threshold_by_model[name],
            )
            for name, frame in aligned.items()
        }
        for name, value in scored.items():
            values[name].append(value)
        for baseline, candidate in comparisons:
            differences[f"{candidate}-{baseline}"].append(scored[candidate] - scored[baseline])

    rows = []
    for name, samples in {**values, **differences}.items():
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(finite, [0.025, 0.975])
        rows.append({
            "metric": metric,
            "comparison": name,
            "point_estimate": float(point[name]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "replicates": replicates,
            "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def validation_selection_table(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Rank model configurations using validation-only multi-seed metrics."""

    frame = pd.DataFrame(records)
    required = {"model", "seed", "macro_auroc", "macro_balanced_accuracy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation selection records missing: {sorted(missing)}")
    grouped = frame.groupby("model", as_index=False).agg(
        val_macro_auroc_mean=("macro_auroc", "mean"),
        val_macro_auroc_sd=("macro_auroc", "std"),
        val_macro_bacc_mean=("macro_balanced_accuracy", "mean"),
        val_macro_bacc_sd=("macro_balanced_accuracy", "std"),
        seeds=("seed", lambda values: ",".join(str(int(value)) for value in sorted(values))),
    )
    if "worst_generator_auroc" in frame:
        worst = frame.groupby("model", as_index=False)["worst_generator_auroc"].mean()
        grouped = grouped.merge(worst, on="model", how="left")
    else:
        grouped["worst_generator_auroc"] = np.nan
    grouped["val_macro_auroc_sd"] = grouped["val_macro_auroc_sd"].fillna(0.0)
    return grouped.sort_values(
        ["val_macro_auroc_mean", "val_macro_bacc_mean", "val_macro_auroc_sd", "worst_generator_auroc"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
