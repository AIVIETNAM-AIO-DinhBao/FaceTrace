"""Small-feature shortcut diagnostics for the Who Is AI split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "log_file_size_bytes",
    "mean_luminance",
    "std_luminance",
    "mean_saturation",
    "gray_pixel_fraction",
    "edge_mean",
]
MODEL_FEATURES = {
    "file_size_only": ["log_file_size_bytes"],
    "simple_features": FEATURE_COLUMNS,
}


def _resolve_image_path(row: pd.Series, image_path_base: Path) -> Path:
    candidates: list[Path] = []
    image_path = str(row.get("image_path", ""))
    original_path = str(row.get("original_path", ""))
    for value in (image_path, original_path):
        if value:
            path = Path(value).expanduser()
            candidates.extend([path, image_path_base / path])
            if original_path and value == original_path:
                candidates.append(image_path_base / "train" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image for {row.get('image_id')}: {candidates}")


def extract_simple_features(image_path: Path) -> dict[str, float]:
    """Match the definitions used by notebook 00 without using labels."""

    file_size = float(image_path.stat().st_size)
    with Image.open(image_path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    array16 = array.astype(np.int16)
    gray_fraction = float(np.mean(array16.max(axis=2) - array16.min(axis=2) <= 2))
    hsv = np.asarray(Image.fromarray(array, mode="RGB").convert("HSV"), dtype=np.uint8)
    luminance = (0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]).astype(np.uint8)
    edges = np.asarray(Image.fromarray(array, mode="RGB").convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    return {
        "log_file_size_bytes": float(np.log1p(file_size)),
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "mean_saturation": float(hsv[:, :, 1].mean()),
        "gray_pixel_fraction": gray_fraction,
        "edge_mean": float(edges.mean()),
    }


def build_feature_table(manifests: Iterable[tuple[str, Path]], image_path_base: Path) -> pd.DataFrame:
    frames = []
    for split_name, manifest_path in manifests:
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        required = {"image_id", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        rows = []
        for _, row in manifest.iterrows():
            path = _resolve_image_path(row, image_path_base)
            features = extract_simple_features(path)
            rows.append({
                "image_id": str(row["image_id"]),
                "label": int(row["label"]),
                "split": split_name,
                "image_path": str(path),
                **features,
            })
        frames.append(pd.DataFrame(rows))
    result = pd.concat(frames, ignore_index=True)
    if result["image_id"].duplicated().any():
        raise ValueError("Train and validation manifests overlap by image_id")
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
    }


def _save_feature_plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS):
        for label, group in table[table["split"] == "train"].groupby("label"):
            axis.hist(group[feature], bins=25, alpha=0.55, density=True, label="Fake" if label else "Real")
        axis.set_title(feature)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_case_montage(predictions: pd.DataFrame, output_path: Path, max_each: int = 6) -> None:
    errors = predictions[~predictions["correct"]].copy()
    false_positive = errors[errors["label"] == 0].sort_values("prob_fake", ascending=False).head(max_each)
    false_negative = errors[errors["label"] == 1].sort_values("prob_fake", ascending=True).head(max_each)
    cases = pd.concat([false_positive, false_negative], ignore_index=True)
    if cases.empty:
        return
    columns = min(6, len(cases))
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (_, row) in zip(axes.flat, cases.iterrows()):
        with Image.open(row["image_path"]) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(
            f"{'Fake' if row['label'] else 'Real'} / pred={'Fake' if row['predicted_label'] else 'Real'}\n"
            f"p_fake={row['prob_fake']:.3f}\n{Path(row['image_path']).name}",
            fontsize=8,
        )
        axis.axis("off")
    fig.suptitle("Shortcut logistic model: highest-confidence validation errors", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_shortcut_diagnostic(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    image_path_base: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Fit train-only-scaled logistic diagnostics and save auditable outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_feature_table(
        [("train", Path(train_manifest)), ("val", Path(validation_manifest))],
        Path(image_path_base).resolve(),
    )
    table.to_csv(output / "feature_table.csv", index=False)
    _save_feature_plot(table, output / "feature_distributions.png")

    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    metrics_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_frames = []
    scaler_records: dict[str, object] = {}

    for model_name, features in MODEL_FEATURES.items():
        scaler = StandardScaler().fit(train[features])
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
        classifier.fit(scaler.transform(train[features]), train["label"].to_numpy())
        scaler_records[model_name] = {
            "features": features,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "converged": bool(classifier.n_iter_.max() < 1000),
            "n_iter": classifier.n_iter_.tolist(),
        }
        for feature, coefficient in zip(features, classifier.coef_[0]):
            coefficient_rows.append({
                "model": model_name,
                "feature": feature,
                "standardized_coefficient": float(coefficient),
                "odds_ratio_per_standard_deviation": float(np.exp(coefficient)),
            })
        for split_name, frame in (("train", train), ("val", val)):
            probabilities = classifier.predict_proba(scaler.transform(frame[features]))[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            result = frame[["image_id", "label", "image_path", *FEATURE_COLUMNS]].copy()
            result.insert(1, "model", model_name)
            result.insert(3, "prob_fake", probabilities)
            result.insert(4, "predicted_label", predictions)
            result.insert(5, "correct", predictions == result["label"].to_numpy())
            result.insert(6, "split", split_name)
            prediction_frames.append(result)
            metrics_rows.append({"model": model_name, "split": split_name, **_metrics(frame["label"].to_numpy(), probabilities, threshold)})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    val_predictions = predictions[predictions["split"] == "val"].copy()
    predictions.to_csv(output / "predictions.csv", index=False)
    val_predictions.to_csv(output / "validation_predictions.csv", index=False)
    suspect_cases = val_predictions[
        (~val_predictions["correct"]) & (val_predictions["model"] == "simple_features")
    ].sort_values("prob_fake")
    suspect_cases.to_csv(output / "shortcut_suspect_cases.csv", index=False)
    _save_case_montage(
        val_predictions[val_predictions["model"] == "simple_features"],
        output / "shortcut_error_montage.png",
    )
    pd.DataFrame(metrics_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "coefficients.csv", index=False)
    (output / "scaler_and_protocol.json").write_text(
        json.dumps({
            "models": scaler_records,
            "threshold": threshold,
            "train_only_fit": True,
            "feature_definitions": FEATURE_COLUMNS,
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": metrics_rows,
        "output_dir": str(output),
        "n_train": len(train),
        "n_validation": len(val),
    }
