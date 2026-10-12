"""Unit tests for the AI-Face Phase F analysis protocol."""

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.aiface_phase_f import (
    CONDITIONS,
    corrupt_image,
    fuse_probability_frames,
    paired_bootstrap,
    validation_selection_table,
)


def _predictions(probabilities):
    return pd.DataFrame({
        "image_id": ["r0", "r1", "g0", "g1", "h0", "h1"],
        "label": [0, 0, 1, 1, 1, 1],
        "generator_id": ["imdb_wiki", "imdb_wiki", "STARGAN", "STARGAN", "StableDiffusion_Inpainting", "StableDiffusion_Inpainting"],
        "split": ["test"] * 6,
        "probability": probabilities,
    })


def test_corruptions_are_deterministic_and_output_size_is_fixed():
    image = Image.new("RGB", (320, 240), (90, 120, 150))
    for condition in CONDITIONS:
        first = corrupt_image(image, condition)
        second = corrupt_image(image, condition)
        expected_size = (224, 224) if condition in {"resize_112_224", "center_crop_80"} else (320, 240)
        assert first.size == expected_size
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_late_fusion_aligns_by_image_id():
    global_frame = _predictions([0.1, 0.2, 0.8, 0.9, 0.7, 0.6])
    local_frame = global_frame.iloc[::-1].copy()
    local_frame["probability"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fused = fuse_probability_frames({"global": global_frame, "local": local_frame}, ("global", "local"))
    assert fused.image_id.tolist() == global_frame.image_id.tolist()
    assert fused.loc[0, "probability"] == 0.3


def test_paired_bootstrap_returns_model_and_delta_rows():
    baseline = _predictions([0.1, 0.2, 0.8, 0.9, 0.7, 0.6])
    candidate = _predictions([0.2, 0.1, 0.9, 0.8, 0.8, 0.7])
    result = paired_bootstrap(
        {"local_only": baseline, "local_forensic": candidate},
        metric="overall_auroc",
        comparisons=(("local_only", "local_forensic"),),
        replicates=20,
        seed=2026,
    )
    assert set(result.comparison) == {"local_only", "local_forensic", "local_forensic-local_only"}
    assert (result.ci_lower <= result.point_estimate).all()
    assert (result.point_estimate <= result.ci_upper).all()


def test_validation_selection_prefers_mean_then_stability():
    rows = [
        {"model": "a", "seed": 42, "macro_auroc": 0.90, "macro_balanced_accuracy": 0.8, "worst_generator_auroc": 0.7},
        {"model": "a", "seed": 43, "macro_auroc": 0.90, "macro_balanced_accuracy": 0.8, "worst_generator_auroc": 0.7},
        {"model": "b", "seed": 42, "macro_auroc": 0.90, "macro_balanced_accuracy": 0.8, "worst_generator_auroc": 0.8},
        {"model": "b", "seed": 43, "macro_auroc": 0.80, "macro_balanced_accuracy": 0.8, "worst_generator_auroc": 0.8},
    ]
    selected = validation_selection_table(rows)
    assert selected.iloc[0].model == "a"
