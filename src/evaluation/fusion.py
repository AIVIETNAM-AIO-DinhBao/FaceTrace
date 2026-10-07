"""ID-safe probability fusion for Week 3 branch comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


def load_prediction_csv(path) -> pd.DataFrame:
    """Load and validate one validation prediction CSV."""

    frame = pd.read_csv(path)
    probability_column = "prob_fake" if "prob_fake" in frame.columns else "probability"
    required = {"image_id", "label", probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Prediction file contains duplicate image_id values: {path}")
    if not np.isfinite(frame[probability_column]).all() or not frame[probability_column].between(0.0, 1.0).all():
        raise ValueError(f"Prediction file contains invalid probabilities: {path}")
    result = frame[["image_id", "label", probability_column]].copy()
    if probability_column != "prob_fake":
        result = result.rename(columns={probability_column: "prob_fake"})
    return result


def merge_predictions(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-merge branch predictions and assert one-to-one labels."""

    if len(predictions) < 2:
        raise ValueError("At least two branches are required for fusion")
    names = list(predictions)
    merged = None
    for name in names:
        frame = predictions[name].copy()
        required = {"image_id", "label", "prob_fake"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Branch {name!r} is missing columns: {sorted(missing)}")
        if frame["image_id"].duplicated().any():
            raise ValueError(f"Branch {name!r} contains duplicate image_id values")
        frame = frame.rename(columns={"label": f"label__{name}", "prob_fake": f"prob_fake__{name}"})
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="outer", validate="one_to_one")
    label_columns = [f"label__{name}" for name in names]
    if merged[label_columns].isna().any().any():
        raise ValueError("Fusion branches do not cover the same image IDs")
    labels = merged[label_columns[0]].astype(int)
    for column in label_columns[1:]:
        if not labels.equals(merged[column].astype(int)):
            raise ValueError("Fusion branches disagree on labels")
    merged.insert(1, "label", labels)
    return merged


def mean_probability(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return fixed equal-weight probability fusion, without fitting weights."""

    merged = merge_predictions(predictions)
    names = list(predictions)
    probability_columns = [f"prob_fake__{name}" for name in names]
    result = merged[["image_id", "label"]].copy()
    result["prob_fake"] = merged[probability_columns].mean(axis=1)
    return result


def evaluate_probability_frame(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Compute the standard binary metrics for an ID-keyed probability frame."""

    required = {"label", "prob_fake"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Probability frame is missing columns: {sorted(missing)}")
    return compute_metrics(frame["label"].to_numpy(), frame["prob_fake"].to_numpy(), threshold)


def predefined_fusions(predictions: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build the fixed fusions planned for Week 3 when branches are available."""

    available = set(predictions)
    requested: dict[str, tuple[str, ...]] = {
        "GL_mean": ("global", "local"),
        "GR_mean": ("global", "residual"),
        "LR_mean": ("local", "residual"),
        "GLR_mean": ("global", "local", "residual"),
        "LC_mean": ("local", "rgb_control"),
    }
    result = {}
    for fusion_name, branch_names in requested.items():
        if set(branch_names).issubset(available):
            result[fusion_name] = mean_probability({name: predictions[name] for name in branch_names})
    return result
