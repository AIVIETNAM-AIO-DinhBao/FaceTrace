"""Export Week 3 residual rescue/harm analysis from saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.error_analysis import run_error_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", default="outputs/week03_kaggle_20261001/analysis/clean_seed42")
    parser.add_argument("--validation-manifest", default="outputs/splits/val.csv")
    parser.add_argument("--image-path-base", default=".")
    parser.add_argument("--output-dir", default="outputs/week03_kaggle_20261001/analysis/clean_seed42/error_rescue_harm")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = REPO_ROOT / args.analysis_root
    prediction_paths = {
        "global": root / "global_only.csv",
        "local": root / "local_only.csv",
        "residual": root / "residual.csv",
        "local_residual": root / "evaluation/LR_mean.csv",
    }
    result = run_error_analysis(
        prediction_paths=prediction_paths,
        validation_manifest=REPO_ROOT / args.validation_manifest,
        image_path_base=REPO_ROOT / args.image_path_base,
        output_dir=REPO_ROOT / args.output_dir,
        threshold=args.threshold,
        seed=args.seed,
    )
    print(f"Saved error analysis to {result['output_dir']}")
    for summary in result["summaries"]:
        print(
            f"{summary['comparison']}: rescue={summary['rescue_count']} "
            f"harm={summary['harm_count']} net={summary['net_gain']}"
        )


if __name__ == "__main__":
    main()
