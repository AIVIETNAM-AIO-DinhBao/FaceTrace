"""Run train-only Logistic Regression shortcut diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.shortcut import run_shortcut_diagnostic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default="outputs/splits/train.csv")
    parser.add_argument("--validation-manifest", default="outputs/splits/val.csv")
    parser.add_argument("--image-path-base", default=".")
    parser.add_argument("--output-dir", default="outputs/week03_shortcut_diagnostic")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = run_shortcut_diagnostic(
        train_manifest=REPO_ROOT / args.train_manifest,
        validation_manifest=REPO_ROOT / args.validation_manifest,
        image_path_base=REPO_ROOT / args.image_path_base,
        output_dir=REPO_ROOT / args.output_dir,
        threshold=args.threshold,
    )
    print(f"Saved shortcut diagnostic to {result['output_dir']}")
    print(f"Rows: train={result['n_train']} validation={result['n_validation']}")
    for row in result["metrics"]:
        if row["split"] == "val":
            print(
                f"{row['model']}: AUROC={row['auroc']:.5f} "
                f"BAcc={row['balanced_accuracy']:.5f} "
                f"Accuracy={row['accuracy']:.5f} F1={row['f1']:.5f}"
            )


if __name__ == "__main__":
    main()
