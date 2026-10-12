"""Select an AI-Face configuration using validation metrics only.

Test predictions are intentionally never read.  Repeat the three root options
for each completed seed; roots are matched by their recorded training seed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_f import validation_selection_table  # noqa: E402


MODEL_ROOTS = {
    "baseline": ("global_only", "local_only", "global_local"),
    "forensic": ("forensic_only",),
    "fusion": ("local_forensic", "global_local_forensic"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", action="append", required=True)
    parser.add_argument("--forensic-root", action="append", required=True)
    parser.add_argument("--fusion-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metadata(root: Path) -> dict:
    path = root / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run_metadata.json: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("smoke"):
        raise ValueError(f"Smoke run cannot be used for model selection: {root}")
    if value.get("split_mode") != "generator_disjoint_fixed":
        raise ValueError(f"Unexpected split mode in {root}")
    return value


def validation_metrics(directory: Path) -> dict[str, float]:
    metrics_path = directory / "val_metrics.json"
    by_generator_path = directory / "val_metrics_by_generator.csv"
    if not metrics_path.is_file() or not by_generator_path.is_file():
        raise FileNotFoundError(f"Validation artifacts missing under {directory}")
    overall = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_generator = pd.read_csv(by_generator_path)
    if by_generator.empty:
        raise ValueError(f"Validation per-generator table is empty: {by_generator_path}")
    auroc = pd.to_numeric(by_generator["auroc"], errors="coerce")
    bacc = pd.to_numeric(by_generator["balanced_accuracy"], errors="coerce")
    return {
        "macro_auroc": float(auroc.mean()),
        "macro_balanced_accuracy": float(bacc.mean()),
        "worst_generator_auroc": float(auroc.min()),
        "threshold": float(overall["threshold"]),
    }


def main() -> int:
    args = parse_args()
    roots = {
        "baseline": [Path(value).expanduser().resolve() for value in args.baseline_root],
        "forensic": [Path(value).expanduser().resolve() for value in args.forensic_root],
        "fusion": [Path(value).expanduser().resolve() for value in args.fusion_root],
    }
    by_seed: dict[int, dict[str, Path]] = {}
    for role, candidates in roots.items():
        for root in candidates:
            run = metadata(root)
            seed = int(run["seed"])
            if seed in by_seed and role in by_seed[seed]:
                raise ValueError(f"Duplicate {role} root for seed {seed}")
            by_seed.setdefault(seed, {})[role] = root
    if not by_seed or any(set(roles) != set(roots) for roles in by_seed.values()):
        raise ValueError("Each seed must have exactly one baseline, forensic and fusion root")

    records: list[dict] = []
    for seed, role_roots in sorted(by_seed.items()):
        for role, names in MODEL_ROOTS.items():
            for model in names:
                directory = role_roots[role] if role == "forensic" else role_roots[role] / model
                metrics = validation_metrics(directory)
                records.append({"seed": seed, "model": model, **metrics})
    records_frame = pd.DataFrame(records)
    selection = validation_selection_table(records_frame)
    selection.insert(0, "selection_rank", np.arange(1, len(selection) + 1))
    selected = str(selection.iloc[0]["model"])

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records_frame.to_csv(output / "validation_records.csv", index=False)
    selection.to_csv(output / "model_selection.csv", index=False)
    payload = {
        "selected_model": selected,
        "selection_metric": "validation macro AUROC mean, then macro BAcc mean, then AUROC SD ascending, then worst-generator AUROC",
        "test_metrics_used_for_selection": False,
        "seeds": sorted(by_seed),
        "roots": {str(seed): {role: str(path) for role, path in values.items()} for seed, values in by_seed.items()},
        "candidates": sorted(records_frame.model.unique()),
        "selection_table": selection.to_dict(orient="records"),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "model_selection.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Selected validation-only model: {selected}")
    print(f"Selection artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
