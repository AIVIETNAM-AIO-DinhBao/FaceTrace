import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.evaluation.error_analysis import compare_predictions, run_error_analysis


class ErrorAnalysisTests(unittest.TestCase):
    def _prediction(self, ids, labels, probabilities):
        return pd.DataFrame({"image_id": ids, "label": labels, "prob_fake": probabilities})

    def _loaded(self, frame, name):
        return frame.rename(columns={"label": f"label__{name}", "prob_fake": f"prob_fake__{name}"})

    def test_rescue_harm_groups_have_expected_counts(self):
        baseline = self._prediction(["a", "b", "c", "d"], [0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])
        candidate = self._prediction(["a", "b", "c", "d"], [0, 1, 0, 1], [0.1, 0.8, 0.8, 0.1])
        cases, result = compare_predictions(
            self._loaded(baseline, "baseline"),
            self._loaded(candidate, "candidate"),
            "baseline",
            "candidate",
        )
        counts = cases["group"].value_counts().to_dict()
        self.assertEqual(counts, {"both_correct": 1, "rescue": 1, "harm": 1, "both_wrong": 1})
        self.assertEqual(result["summary"]["rescue_count"], 1)
        self.assertEqual(result["summary"]["harm_count"], 1)
        self.assertEqual(result["summary"]["net_gain"], 0)
        self.assertEqual(sum(counts.values()), 4)

    def test_mismatched_ids_raise(self):
        baseline = self._prediction(["a"], [0], [0.1])
        candidate = self._prediction(["b"], [0], [0.1])
        with self.assertRaises(ValueError):
            compare_predictions(
                self._loaded(baseline, "baseline"),
                self._loaded(candidate, "candidate"),
                "baseline",
                "candidate",
            )

    def test_mismatched_labels_raise(self):
        baseline = self._prediction(["a"], [0], [0.1])
        candidate = self._prediction(["a"], [1], [0.9])
        with self.assertRaises(ValueError):
            compare_predictions(
                self._loaded(baseline, "baseline"),
                self._loaded(candidate, "candidate"),
                "baseline",
                "candidate",
            )

    def test_run_exports_all_comparisons_on_manifest_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "images"
            image_dir.mkdir()
            rows = []
            ids = [f"train/{i:04d}.jpg" for i in range(4)]
            labels = [0, 1, 0, 1]
            for image_id, label in zip(ids, labels):
                image_path = image_dir / Path(image_id).name
                Image.new("RGB", (12, 12), (30 + label * 180, 40, 50)).save(image_path)
                rows.append({"image_id": image_id, "image_path": str(image_path), "label": label})
            manifest = root / "val.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            paths = {}
            predictions = {
                "global": [0.1, 0.2, 0.8, 0.9],
                "local": [0.1, 0.8, 0.2, 0.9],
                "residual": [0.1, 0.8, 0.2, 0.1],
                "local_residual": [0.1, 0.8, 0.8, 0.9],
            }
            for name, probabilities in predictions.items():
                path = root / f"{name}.csv"
                pd.DataFrame({"image_id": ids, "label": labels, "prob_fake": probabilities}).to_csv(path, index=False)
                paths[name] = path

            output = root / "analysis"
            result = run_error_analysis(paths, manifest, root, output, max_per_group=2)
            self.assertEqual(result["n_cases"], 12)
            groups = pd.read_csv(output / "error_rescue_harm.csv")
            self.assertEqual(groups.groupby("comparison")["n_images"].sum().to_dict(), {
                "local_residual_vs_local": 4,
                "residual_vs_global": 4,
                "residual_vs_local": 4,
            })
            self.assertTrue((output / "montages/residual_vs_global.png").is_file())


if __name__ == "__main__":
    unittest.main()
