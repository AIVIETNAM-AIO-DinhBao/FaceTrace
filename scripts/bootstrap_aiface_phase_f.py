"""Bootstrap AI-Face Phase F test uncertainty from frozen predictions.

The script never loads images or retrains models.  It resamples the shared real
pool and each held-out fake generator, then reuses those indices for every
model so paired late-fusion differences are meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_f import MODELS, paired_bootstrap  # noqa: E402


COMPARISONS = (
    ("local_only", "local_forensic"),
    ("global_local", "global_local_forensic"),
    ("local_only", "forensic_only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-f-dir", action="append", required=True,
        help="Completed Phase F directory; repeat for seeds 42/43/44.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+", default=["clean", "jpeg_q70", "resize_112_224", "blur_r1", "center_crop_80"])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026, help="Bootstrap RNG seed.")
    return parser.parse_args()


def load_metadata(root: Path) -> dict:
    path = root / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase F metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("smoke"):
        raise ValueError(f"Smoke Phase F output is not valid for CI: {root}")
    if metadata.get("split_mode") != "generator_disjoint_fixed":
        raise ValueError(f"Unexpected split mode in {root}")
    return metadata


def load_prediction(root: Path, condition: str, model: str) -> pd.DataFrame:
    path = root / f"predictions_{condition}_{model}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    return pd.read_csv(path, keep_default_na=False, dtype={"image_id": str})


def main() -> int:
    args = parse_args()
    roots = [Path(value).expanduser().resolve() for value in args.phase_f_dir]
    metadata = [(root, load_metadata(root)) for root in roots]
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ci_rows: list[dict] = []
    delta_rows: list[dict] = []
    seed_metric_rows: list[pd.DataFrame] = []
    for root, run in metadata:
        seed = int(run["seed"])
        metrics_path = root / "metrics_by_condition_model.csv"
        if metrics_path.is_file():
            metrics = pd.read_csv(metrics_path)
            if "seed" in metrics.columns:
                metrics["seed"] = seed
            else:
                metrics.insert(0, "seed", seed)
            seed_metric_rows.append(metrics)
        for condition in args.conditions:
            frames = {model: load_prediction(root, condition, model) for model in MODELS}
            thresholds = {
                model: float(frames[model]["threshold"].iloc[0])
                for model in MODELS
                if "threshold" in frames[model]
            }
            for metric in ("macro_auroc", "macro_balanced_accuracy"):
                result = paired_bootstrap(
                    frames,
                    metric=metric,
                    comparisons=COMPARISONS,
                    replicates=args.replicates,
                    seed=args.seed,
                    threshold_by_model=thresholds,
                )
                model_rows = result[result["comparison"].isin(MODELS)].copy()
                model_rows.insert(0, "seed", seed)
                model_rows.insert(1, "condition", condition)
                ci_rows.extend(model_rows.to_dict(orient="records"))
                delta = result[~result["comparison"].isin(MODELS)].copy()
                delta.insert(0, "seed", seed)
                delta.insert(1, "condition", condition)
                delta_rows.extend(delta.to_dict(orient="records"))

    ci_frame = pd.DataFrame(ci_rows)
    delta_frame = pd.DataFrame(delta_rows)
    ci_frame.to_csv(output / "ci_by_model_condition.csv", index=False)
    delta_frame.to_csv(output / "paired_delta_ci.csv", index=False)
    if seed_metric_rows:
        all_metrics = pd.concat(seed_metric_rows, ignore_index=True)
        summary = all_metrics.groupby(["condition", "model"], as_index=False).agg(
            seeds=("seed", lambda values: ",".join(str(int(value)) for value in sorted(values))),
            n_seeds=("seed", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_sd=("macro_auroc", "std"),
            macro_balanced_accuracy_mean=("macro_balanced_accuracy", "mean"),
            macro_balanced_accuracy_sd=("macro_balanced_accuracy", "std"),
        )
        for column in summary.columns:
            if column.endswith("_sd"):
                summary[column] = summary[column].fillna(0.0)
        summary.to_csv(output / "seed_summary_mean_sd.csv", index=False)
    (output / "bootstrap_config.yaml").write_text(yaml.safe_dump({
        "replicates": args.replicates,
        "seed": args.seed,
        "confidence_level": 0.95,
        "resampling": "generator_id_x_label_cells; shared indices across models",
        "conditions": list(args.conditions),
        "comparisons": [f"{candidate}-{baseline}" for baseline, candidate in COMPARISONS],
    }, sort_keys=False), encoding="utf-8")
    (output / "bootstrap_metadata.json").write_text(json.dumps({
        "phase_f_dirs": [str(root) for root, _ in metadata],
        "seeds": [int(run["seed"]) for _, run in metadata],
        "split_mode": "generator_disjoint_fixed",
        "manifest_sha256": [run.get("manifest_sha256") for _, run in metadata],
        "replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "confidence_level": 0.95,
        "scientific_claim_boundary": "conditional on fixed split and selected checkpoints",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"AI-Face bootstrap artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
