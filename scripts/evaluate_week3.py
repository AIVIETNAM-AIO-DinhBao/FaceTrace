"""Evaluate Week 3 branch predictions and fixed mean-probability fusions.

Example:
    python3 scripts/evaluate_week3.py \
      --prediction local=outputs/week03/seed42/local/validation_predictions.csv \
      --prediction residual=outputs/week03/seed42/residual_only/validation_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.fusion import (  # noqa: E402
    evaluate_probability_frame,
    load_prediction_csv,
    predefined_fusions,
)


def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Prediction must use NAME=PATH syntax")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Prediction NAME and PATH cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction_spec)
    parser.add_argument("--output-dir", default="outputs/week03/analysis")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for name, path in args.prediction:
        if name in predictions:
            raise ValueError(f"Duplicate prediction branch: {name}")
        predictions[name] = load_prediction_csv(path)

    metrics: list[dict] = []
    for name, frame in predictions.items():
        metrics.append({"representation": name, **evaluate_probability_frame(frame, args.threshold)})

    for fusion_name, frame in predefined_fusions(predictions).items():
        frame.to_csv(output_dir / f"{fusion_name}.csv", index=False)
        metrics.append({"representation": fusion_name, **evaluate_probability_frame(frame, args.threshold)})

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "representations": list(predictions),
                "metrics": metrics,
                "warning": "Metrics are only as independent as the supplied validation protocol.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metrics_path)
    for row in metrics:
        print(
            f"{row['representation']}: AUROC={row['auroc']:.5f} "
            f"BAcc={row['balanced_accuracy']:.5f}"
        )


if __name__ == "__main__":
    main()
