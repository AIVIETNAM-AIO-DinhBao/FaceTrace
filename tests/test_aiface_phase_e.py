from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.aiface_phase_e import (
    metrics_by_generator,
    prediction_frame,
    select_threshold,
    validate_fixed_manifests,
)
from src.evaluation.fusion import load_prediction_csv


def _make_split(root: Path, split: str, generator: str, start: int) -> Path:
    split_root = root / split
    image_root = split_root / "images"
    image_root.mkdir(parents=True)
    rows = []
    for offset, label in enumerate((0, 1)):
        image_id = f"{split}-{start + offset}"
        image_path = image_root / f"{image_id}.png"
        Image.fromarray(np.full((8, 8, 3), 255 * label, dtype=np.uint8)).save(image_path)
        rows.append({
            "image_path": f"images/{image_path.name}",
            "image_id": image_id,
            "label": label,
            "split": split,
            "family": "Real" if label == 0 else "GANs",
            "generator_id": "imdb_wiki" if label == 0 else generator,
        })
    manifest = split_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_fixed_manifest_validation_and_generator_metrics(tmp_path):
    manifests = {
        "train": _make_split(tmp_path, "train", "train_gen", 0),
        "val": _make_split(tmp_path, "val", "val_gen", 10),
        "test": _make_split(tmp_path, "test", "test_gen", 20),
    }
    bases = {split: tmp_path / split for split in manifests}
    result = validate_fixed_manifests(manifests, bases)
    assert result["generator_splits"] == {"train": ["train_gen"], "val": ["val_gen"], "test": ["test_gen"]}

    frame = pd.read_csv(manifests["test"])
    predictions = prediction_frame(frame, [0.1, 0.9], 0.5)
    overall, by_generator = metrics_by_generator(predictions, 0.5)
    assert overall["auroc"] == 1.0
    assert list(by_generator["generator_id"]) == ["test_gen"]


def test_threshold_selection_is_deterministic():
    threshold, metrics = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert threshold == 0.5
    assert metrics["balanced_accuracy"] == 1.0


def test_fusion_loader_accepts_phase_e_probability_column(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"image_id": ["a"], "label": [1], "probability": [0.8]}).to_csv(path, index=False)
    loaded = load_prediction_csv(path)
    assert list(loaded.columns) == ["image_id", "label", "prob_fake"]
    assert loaded.loc[0, "prob_fake"] == 0.8
