"""Artifact ZIP/directory discovery for the CPU fusion notebook."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fusion_kaggle", REPO_ROOT / "scripts/run_aiface_fusion_kaggle.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_artifact(root, role, smoke=False):
    root.mkdir(parents=True, exist_ok=True)
    for name in runner.required_files(role):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (root / "run_metadata.json").write_text(json.dumps({
        "split_mode": "generator_disjoint_fixed", "smoke": smoke,
    }))
    if role == "baseline":
        for branch in ("global_only", "local_only"):
            (root / branch / "predictions_val.csv").write_text("fixture")


class FusionInputTests(unittest.TestCase):
    def test_zip_with_full_and_smoke_resolves_only_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_artifact(root / "source/forensic_only", "forensic")
            make_artifact(root / "source/forensic_only_smoke", "forensic", smoke=True)
            zip_path = root / "inputs/artifact.zip"
            zip_path.parent.mkdir()
            with zipfile.ZipFile(zip_path, "w") as handle:
                for path in (root / "source").rglob("*"):
                    if path.is_file():
                        handle.write(path, path.relative_to(root / "source"))
            found, provenance = runner.resolve_artifact_input(zip_path.parent, "forensic", root / "working")
            self.assertEqual(found.name, "forensic_only")
            self.assertEqual(provenance["kind"], "zip")
            self.assertEqual(found, runner.resolve_artifact_input(zip_path, "forensic", root / "working")[0])

    def test_unpacked_baseline_and_missing_validation_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_artifact(root / "baseline/outputs/aiface_phase_e", "baseline")
            found, provenance = runner.resolve_artifact_input(root / "baseline", "baseline", root / "work")
            self.assertEqual(found, (root / "baseline/outputs/aiface_phase_e").resolve())
            self.assertEqual(provenance["kind"], "unpacked_notebook_output")
            # Only test predictions are insufficient for validation thresholds.
            (found / "local_only/predictions_val.csv").rename(found / "local_only/saved_val.csv")
            with self.assertRaisesRegex(FileNotFoundError, "feature_cache.pt"):
                runner.resolve_artifact_input(root / "baseline", "baseline", root / "work")
            (found / "feature_cache.pt").write_text("fixture")
            (found / "local_only/checkpoint.pt").write_text("fixture")
            self.assertEqual(runner.resolve_artifact_input(root / "baseline", "baseline", root / "work")[0], found)

    def test_ambiguous_runs_and_smoke_only_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_artifact(root / "a", "forensic", smoke=True)
            with self.assertRaisesRegex(FileNotFoundError, "Smoke"):
                runner.resolve_artifact_input(root, "forensic", root / "work")
            make_artifact(root / "b", "forensic")
            make_artifact(root / "c", "forensic")
            with self.assertRaisesRegex(ValueError, "Multiple"):
                runner.resolve_artifact_input(root, "forensic", root / "work")

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside", "not allowed")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                runner.extract_artifact_zip(archive, root / "work", "forensic")
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
