"""Numerical, protocol and end-to-end tests without external data or models."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from src.evaluation.aiface_phase_e import select_threshold
from src.evaluation.spectral import (
    GENERATOR_SPLITS, align_predictions, find_subset_metadata, save_predictions, select_validation_threshold, validate_pilot, write_json,
)
from src.input_data.spectral_dataset import SpectralDataset
from src.models.classifier import MLPClassifier
from src.models.spectral import RadialFFT, SpectralMLP, parameter_count
from src.training.spectral_trainer import SpectralPredictor, predict_features

REPO_ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixture(root):
    rng = np.random.default_rng(42)
    manifests, bases = {}, {}
    for split, generators in GENERATOR_SPLITS.items():
        base = root / split
        (base / "images").mkdir(parents=True)
        rows = []
        groups = [(0, "", "Real")] + [(1, generator, "GANs" if generator in
                  {"taming_transformer_VQGAN", "stylegan3", "AttGAN", "STARGAN"} else "DMs")
                  for generator in sorted(generators)]
        for label, generator, family in groups:
            for i in range(4):
                image_id = f"{split}-{generator or 'real'}-{i}"
                values = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
                Image.fromarray(values).save(base / "images" / f"{image_id}.png")
                rows.append({"image_id": image_id, "image_path": f"images/{image_id}.png", "label": label,
                             "generator_id": generator, "family": family, "split": split,
                             "real_source": "imdb_wiki" if label == 0 else ""})
        path = base / "manifest.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        manifests[split], bases[split] = path, base
    (root / "subset_metadata.json").write_text('{"fixture": true}', encoding="utf-8")
    return manifests, bases


class SpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fft_matches_numpy_reference(self):
        values = np.random.default_rng(3).random((2, 3, 16, 16), dtype=np.float32)
        transform = RadialFFT(16, 8)
        actual = transform(torch.from_numpy(values)).numpy().reshape(2, 3, 8)
        window = np.outer(np.hanning(16), np.hanning(16))
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(values * window), axes=(-2, -1))))
        axis = np.arange(16) - 8
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        # Independent NumPy reference: exact integer annular masks, including
        # points ON a boundary in its higher bin. Float sqrt/floor can differ
        # across platforms even at float64 precision (e.g. diagonal (3, 3)).
        squared = xx**2 + yy**2
        scaled_squared = squared * 8**2
        masks = [
            (scaled_squared >= index**2 * squared.max()) &
            ((scaled_squared < (index + 1)**2 * squared.max()) if index < 7 else True)
            for index in range(8)
        ]
        expected = np.stack([spectrum[:, :, mask].mean(-1) for mask in masks], axis=-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(parameter_count(transform), 0)

    def test_exact_radial_boundaries_and_integer_reference(self):
        transform = RadialFFT(16, 8)
        bins = transform.bin_index.reshape(16, 16)
        # Diagonal (k,k) has exactly radius k/8 * r_max for this geometry.
        self.assertEqual([int(bins[8 + k, 8 + k]) for k in range(8)], list(range(8)))
        self.assertEqual(int(bins[0, 0]), 7)  # Include the maximum-radius corner.
        for size, count in ((16, 8), (224, 64), (15, 8), (16, 1)):
            center = size // 2
            maximum_squared = 2 * center**2
            expected = [
                min(count - 1, math.isqrt(((y - center)**2 + (x - center)**2) * count**2 // maximum_squared))
                for y in range(size) for x in range(size)
            ]
            self.assertEqual(RadialFFT(size, count).bin_index.tolist(), expected)

    def test_window_shapes_finiteness_and_batch_invariance(self):
        transform = RadialFFT()
        self.assertEqual(tuple(transform.window.shape), (224, 224))
        self.assertGreaterEqual(float(transform.window.min()), 0)
        self.assertLessEqual(float(transform.window.max()), 1)
        self.assertTrue((transform.window[0] == 0).all())
        image = torch.rand(1, 3, 224, 224)
        a, b = transform(image), transform(image.expand(3, -1, -1, -1))
        self.assertEqual(tuple(b.shape), (3, 192))
        torch.testing.assert_close(a[0], b[2])
        self.assertTrue(torch.isfinite(transform(torch.zeros_like(image))).all())
        with self.assertRaises(ValueError):
            transform(torch.full_like(image, float("nan")))
        with self.assertRaises(ValueError):
            RadialFFT(16, 100)

    def test_mlp_normalization_train_only_and_probabilities(self):
        model = SpectralMLP()
        features = torch.rand(8, 192)
        model.fit_normalization(features)
        torch.testing.assert_close(model.feature_mean, features.mean(0))
        probabilities = model.predict(features)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(parameter_count(model), 24962)
        with self.assertRaises(RuntimeError):
            model.fit_normalization(features + 100)

    def test_threshold_matches_baseline_policy_with_ties(self):
        rng = np.random.default_rng(123)
        for _ in range(10):
            labels = np.r_[0, 1, rng.integers(0, 2, 18)]
            probabilities = rng.choice([0.0, 0.2, 0.5, 0.8, 1.0], 20)
            expected, metrics = select_threshold(labels, probabilities)
            actual, actual_metrics = select_validation_threshold(labels, probabilities)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_metrics, metrics)

    def test_alignment_and_rescue_harm(self):
        frame = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "label": [0, 1, 0, 1],
                              "generator_id": ["", "STARGAN", "", "STARGAN"], "split": "test",
                              "probability": [0.2, 0.2, 0.8, 0.8]})
        candidate = frame.copy()
        candidate.probability = [0.8, 0.8, 0.2, 0.8]
        aligned = align_predictions(frame, candidate.iloc[::-1])
        self.assertEqual(aligned.image_id.tolist(), frame.image_id.tolist())
        cases, counts = script_module("fuse_aiface_spectral").rescue_harm(frame, aligned, 0.5, 0.5)
        self.assertEqual(cases.group.tolist(), ["harm", "rescue", "rescue", "both_correct"])
        self.assertEqual(int(counts.iloc[0].net_gain), 1)
        with self.assertRaises(ValueError):
            align_predictions(frame, candidate.iloc[:-1])
        candidate.loc[0, "label"] = 1
        with self.assertRaises(ValueError):
            align_predictions(frame, candidate)

    def test_dataset_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests, bases = make_fixture(Path(directory))
            validate_pilot(manifests, bases, require_32k=False)
            with self.assertRaisesRegex(ValueError, "counts do not match"):
                validate_pilot(manifests, bases, require_32k=True)
            dataset = SpectralDataset(manifests["train"], bases["train"], 16)
            self.assertEqual(dataset[0]["image_id"], dataset.frame.iloc[0].image_id)
            self.assertEqual(dataset[0]["label"].item(), int(dataset.frame.iloc[0].label))
            self.assertEqual(tuple(dataset[0]["pixel_values"].shape), (3, 16, 16))
            frame = pd.read_csv(manifests["test"])
            frame.loc[frame.label == 1, "generator_id"] = "AttGAN"
            frame.to_csv(manifests["test"], index=False)
            with self.assertRaises(ValueError):
                validate_pilot(manifests, bases, require_32k=False)

    def test_archive_nested_root_and_traversal(self):
        runner = script_module("run_aiface_spectral_kaggle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            make_fixture(source)
            archive = root / "pilot.bin"
            with tarfile.open(archive, "w") as handle:
                handle.add(source, arcname="pilot")
            extracted = runner.unpack_dataset(archive, root / "work")
            self.assertTrue((extracted / "train/manifest.csv").is_file())
            self.assertEqual(extracted, runner.unpack_dataset(archive, root / "work"))
            unsafe = root / "unsafe.bin"
            with tarfile.open(unsafe, "w") as handle:
                member = tarfile.TarInfo("../escape")
                handle.addfile(member)
            with self.assertRaises(ValueError):
                runner.unpack_dataset(unsafe, root / "work2")
            symlink = root / "link.bin"
            with tarfile.open(symlink, "w") as handle:
                member = tarfile.TarInfo("evil")
                member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
                handle.addfile(member)
            with self.assertRaises(ValueError):
                runner.unpack_dataset(symlink, root / "work3")

    def test_discovery_does_not_require_metadata_at_root(self):
        runner = script_module("run_aiface_spectral_kaggle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "pilot"
            make_fixture(package)
            metadata = package / "subset_metadata.json"
            (package / "audit").mkdir()
            metadata.rename(package / "audit/subset_metadata.json")
            self.assertEqual(runner.find_dataset_root(root), package.resolve())
            self.assertEqual(find_subset_metadata(package), package / "audit/subset_metadata.json")
            archive = root / "pilot.bin"
            with tarfile.open(archive, "w") as handle:
                handle.add(package, arcname="pilot")
            extracted = runner.unpack_dataset(archive, root / "working")
            self.assertEqual(find_subset_metadata(extracted), extracted / "audit/subset_metadata.json")
            # Absence is a metadata error, not proof that manifests are absent.
            missing_metadata = package / "missing_metadata"
            metadata_path = package / "audit/subset_metadata.json"
            metadata_path.rename(missing_metadata)
            self.assertEqual(runner.find_dataset_root(package), package.resolve())
            self.assertIsNone(find_subset_metadata(package))

    def test_end_to_end_training_and_checkpoint_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests, bases = make_fixture(root / "dataset")
            # Match the actual published single-bin: no subset_metadata.json.
            fixture_metadata = root / "dataset/subset_metadata.json"
            fixture_metadata.rename(root / "original_fixture_metadata.json")
            config = yaml.safe_load((REPO_ROOT / "configs/spectral.yaml").read_text())
            config["data"].update(image_size=16, num_workers=0, require_32k=False, batch_size=8)
            config["model"].update(radial_bins=8, hidden_dim=8)
            config["training"].update(epochs=2, device="cpu")
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "output"
            command = [sys.executable, str(REPO_ROOT / "scripts/train_spectral.py"),
                       "--config", str(path), "--data-root", str(root / "dataset"), "--output-dir", str(output)]
            process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            expected = pd.read_csv(output / "predictions_test.csv")
            self.assertEqual(expected.image_id.tolist(), pd.read_csv(manifests["test"]).image_id.tolist())
            predictor = SpectralPredictor(output / "checkpoint.pt", "cpu")
            actual = predictor.predict_manifest(manifests["test"], bases["test"], root / "predicted")
            np.testing.assert_allclose(actual.probability, expected.probability, rtol=1e-6)
            probability = predictor.predict(bases["test"] / pd.read_csv(manifests["test"]).iloc[0].image_path)
            self.assertAlmostEqual(probability, expected.iloc[0].probability, places=6)
            metrics = json.loads((output / "test_metrics_overall.json").read_text())
            self.assertIn("macro", metrics)
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            train_features = torch.load(output / "features_train.pt", weights_only=True)["features"]
            torch.testing.assert_close(checkpoint["state_dict"]["feature_mean"], train_features.mean(0))
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["rows_used"]["test"], len(expected))
            recorded = json.loads((output / "run_metadata.json").read_text())
            self.assertFalse(recorded["original_subset_metadata_available"])
            self.assertIsNone(recorded["subset_metadata"])
            self.assertIsNone(recorded["manifest_derived_evidence"]["original_sampling_seed"])
            self.assertEqual(recorded["manifest_derived_evidence"]["rows"]["test"], len(expected))

            # Simulate the exact current AI-Face baseline artifact format, with
            # no validation CSVs. Verify reconstruction, fusion and rescue/harm.
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            caches = {split: torch.load(output / f"features_{split}.pt", weights_only=True)
                      for split in ("train", "val", "test")}
            all_features = torch.cat([cache["features"] for cache in caches.values()])
            all_labels = torch.cat([cache["labels"] for cache in caches.values()])
            all_ids = sum([cache["image_ids"] for cache in caches.values()], [])
            split_indices, cursor = {}, 0
            for split, cache in caches.items():
                split_indices[split] = list(range(cursor, cursor + len(cache["labels"])))
                cursor += len(cache["labels"])
            torch.save({"global": all_features, "local": all_features * 0.9, "labels": all_labels,
                        "image_ids": all_ids, "split_indices": split_indices}, baseline_root / "feature_cache.pt")
            metadata = json.loads((output / "run_metadata.json").read_text())
            baseline_metadata = dict(metadata)
            baseline_metadata["preprocessing"] = {"image_size": 16}
            write_json(baseline_root / "run_metadata.json", baseline_metadata)
            frames = {split: pd.read_csv(manifests[split], keep_default_na=False) for split in manifests}
            predictions = {}
            for branch, scale in (("global_only", 1.0), ("local_only", 0.9)):
                torch.manual_seed(17 if branch == "global_only" else 23)
                head = MLPClassifier(24, 8, 2, 0.0)
                val_p = predict_features(head, caches["val"]["features"] * scale, torch.device("cpu"))
                test_p = predict_features(head, caches["test"]["features"] * scale, torch.device("cpu"))
                threshold, _ = select_validation_threshold(frames["val"].label, val_p)
                branch_root = baseline_root / branch
                branch_root.mkdir()
                torch.save({"classifier": head.state_dict(), "threshold": threshold,
                            "config": {"model": {"hidden_dim": 8, "dropout": 0.0}}}, branch_root / "checkpoint.pt")
                predictions[branch] = {"val": val_p, "test": test_p}
                save_predictions(branch_root, frames["test"], test_p, threshold, "test")
                write_json(branch_root / "val_metrics.json", {"threshold": threshold})
            gl_val = (predictions["global_only"]["val"] + predictions["local_only"]["val"]) / 2
            gl_test = (predictions["global_only"]["test"] + predictions["local_only"]["test"]) / 2
            gl_threshold, _ = select_validation_threshold(frames["val"].label, gl_val)
            save_predictions(baseline_root / "global_local", frames["test"], gl_test, gl_threshold, "test")
            write_json(baseline_root / "global_local/val_metrics.json", {"threshold": gl_threshold})
            fusion_module = script_module("fuse_aiface_spectral")
            restored = fusion_module.load_baseline_validation(baseline_root, "global_only", frames["val"])
            np.testing.assert_allclose(restored.probability, predictions["global_only"]["val"])
            fused_root = root / "fused"
            fusion_command = [sys.executable, str(REPO_ROOT / "scripts/fuse_aiface_spectral.py"),
                              "--baseline-dir", str(baseline_root), "--forensic-dir", str(output),
                              "--output-dir", str(fused_root)]
            process = subprocess.run(fusion_command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            triple = pd.read_csv(fused_root / "global_local_forensic/predictions_test.csv")
            np.testing.assert_allclose(triple.probability,
                (predictions["global_only"]["test"] + predictions["local_only"]["test"] + expected.probability.to_numpy()) / 3)
            cases = pd.read_csv(fused_root / "local_forensic/error_cases.csv")
            counts = pd.read_csv(fused_root / "local_forensic/rescue_harm.csv")
            self.assertEqual(len(cases), len(expected))
            self.assertEqual(int(counts.iloc[0].net_gain), int((cases.group == "rescue").sum() - (cases.group == "harm").sum()))
            baseline_metadata["seed"] = 43
            write_json(baseline_root / "run_metadata.json", baseline_metadata)
            mismatch = subprocess.run(fusion_command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("disagree on seed", mismatch.stderr)

            # Smoke must create separate output, never replace the full run.
            smoke = subprocess.run(command + ["--smoke"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(smoke.returncode, 0, smoke.stdout + smoke.stderr)
            smoke_json = json.loads((root / "output_smoke/smoke_test.json").read_text())
            self.assertTrue(smoke_json["passed"])
            self.assertFalse(smoke_json["scientific_result"])


if __name__ == "__main__":
    unittest.main()
