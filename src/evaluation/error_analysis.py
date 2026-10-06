"""ID-safe rescue/harm analysis for Week 3 branch and fusion predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.evaluation.metrics import compute_metrics


GROUPS = ("both_correct", "rescue", "harm", "both_wrong")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: str | Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"image_id", "label", "prob_fake"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} prediction is missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{name} prediction has duplicate image_id values")
    frame = frame[["image_id", "label", "prob_fake"]].copy()
    frame["label"] = frame["label"].astype(int)
    frame["prob_fake"] = frame["prob_fake"].astype(float)
    if not frame["prob_fake"].between(0.0, 1.0).all():
        raise ValueError(f"{name} prediction has probabilities outside [0, 1]")
    frame = frame.rename(columns={"label": f"label__{name}", "prob_fake": f"prob_fake__{name}"})
    return frame


def compare_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = baseline.merge(candidate, on="image_id", how="outer", validate="one_to_one")
    if merged.isna().any().any():
        raise ValueError(f"{baseline_name}/{candidate_name} do not cover identical IDs")
    if not merged[f"label__{baseline_name}"].equals(merged[f"label__{candidate_name}"]):
        raise ValueError(f"{baseline_name}/{candidate_name} labels disagree")
    labels = merged[f"label__{baseline_name}"].astype(int)
    base_prob = merged[f"prob_fake__{baseline_name}"].astype(float)
    cand_prob = merged[f"prob_fake__{candidate_name}"].astype(float)
    base_pred = (base_prob >= threshold).astype(int)
    cand_pred = (cand_prob >= threshold).astype(int)
    base_correct = base_pred.eq(labels)
    cand_correct = cand_pred.eq(labels)
    group = np.select(
        [base_correct & cand_correct, ~base_correct & cand_correct,
         base_correct & ~cand_correct, ~base_correct & ~cand_correct],
        ["both_correct", "rescue", "harm", "both_wrong"],
        default="both_wrong",
    )
    cases = pd.DataFrame({
        "comparison": f"{candidate_name}_vs_{baseline_name}",
        "image_id": merged["image_id"],
        "label": labels,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "baseline_prob_fake": base_prob,
        "candidate_prob_fake": cand_prob,
        "baseline_predicted_label": base_pred,
        "candidate_predicted_label": cand_pred,
        "baseline_correct": base_correct,
        "candidate_correct": cand_correct,
        "group": group,
    })
    rows = []
    for group_name in GROUPS:
        selected = cases[cases["group"] == group_name]
        rows.append({
            "comparison": cases["comparison"].iloc[0],
            "baseline": baseline_name,
            "candidate": candidate_name,
            "group": group_name,
            "n_images": int(len(selected)),
            "n_real": int((selected["label"] == 0).sum()),
            "n_fake": int((selected["label"] == 1).sum()),
        })
    rescue_count = int((cases["group"] == "rescue").sum())
    harm_count = int((cases["group"] == "harm").sum())
    base_errors = int((~base_correct).sum())
    summary = {
        "comparison": cases["comparison"].iloc[0],
        "baseline": baseline_name,
        "candidate": candidate_name,
        "n_images": int(len(cases)),
        "baseline_errors": base_errors,
        "candidate_errors": int((~cand_correct).sum()),
        "rescue_count": rescue_count,
        "harm_count": harm_count,
        "net_gain": rescue_count - harm_count,
        "rescue_rate": rescue_count / base_errors if base_errors else None,
        "baseline_metrics": compute_metrics(labels.to_numpy(), base_prob.to_numpy(), threshold),
        "candidate_metrics": compute_metrics(labels.to_numpy(), cand_prob.to_numpy(), threshold),
    }
    return cases, {"summary": summary, "groups": rows}


def _resolve_images(val_manifest: Path, image_path_base: Path) -> dict[str, Path]:
    manifest = pd.read_csv(val_manifest, keep_default_na=False)
    result = {}
    for _, row in manifest.iterrows():
        candidates = []
        for value in (row.get("image_path", ""), row.get("original_path", "")):
            if not value:
                continue
            path = Path(str(value))
            candidates.extend([path, image_path_base / path])
            if row.get("original_path", ""):
                candidates.append(image_path_base / "train" / Path(str(row["original_path"])))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Could not resolve validation image {row['image_id']}")
        result[str(row["image_id"])] = path.resolve()
    return result


def _save_montage(cases: pd.DataFrame, image_paths: Mapping[str, Path], path: Path, seed: int, max_per_group: int) -> None:
    chosen = []
    rng = np.random.default_rng(seed)
    for group_name in GROUPS:
        group = cases[cases["group"] == group_name]
        if len(group) > max_per_group:
            group = group.iloc[np.sort(rng.choice(len(group), max_per_group, replace=False))]
        chosen.append(group)
    selected = pd.concat(chosen, ignore_index=True)
    if selected.empty:
        return
    columns = min(max_per_group, 4)
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.6 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        with Image.open(image_paths[str(row["image_id"])]) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(
            f"{row['group']}\n{row['image_id']}\n"
            f"B={row['baseline_prob_fake']:.3f}, C={row['candidate_prob_fake']:.3f}",
            fontsize=8,
        )
    fig.suptitle(f"{cases['candidate'].iloc[0]} vs {cases['baseline'].iloc[0]} - error groups", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_error_analysis(
    prediction_paths: Mapping[str, str | Path],
    validation_manifest: str | Path,
    image_path_base: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    seed: int = 42,
    max_per_group: int = 8,
) -> dict[str, object]:
    """Run all planned Week 3 rescue/harm comparisons and export artifacts."""

    output = Path(output_dir)
    montage_dir = output / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_prediction(path, name) for name, path in prediction_paths.items()}
    expected = {
        "global",
        "local",
        "residual",
        "local_residual",
        "global_local",
        "global_local_residual",
    }
    missing = expected - set(loaded)
    if missing:
        raise ValueError(f"Missing prediction branches: {sorted(missing)}")
    comparisons = [
        ("global", "residual"),
        ("local", "residual"),
        ("local", "local_residual"),
        ("global_local", "global_local_residual"),
    ]
    image_paths = _resolve_images(Path(validation_manifest), Path(image_path_base).resolve())
    expected_ids = set(image_paths)
    for name, frame in loaded.items():
        observed_ids = set(frame["image_id"])
        if observed_ids != expected_ids:
            missing_ids = sorted(expected_ids - observed_ids)[:5]
            extra_ids = sorted(observed_ids - expected_ids)[:5]
            raise ValueError(
                f"{name} prediction IDs do not match validation manifest "
                f"(missing examples={missing_ids}, extra examples={extra_ids})"
            )
    all_cases = []
    all_groups = []
    summaries = []
    for baseline_name, candidate_name in comparisons:
        cases, result = compare_predictions(
            loaded[baseline_name], loaded[candidate_name], baseline_name, candidate_name, threshold
        )
        all_cases.append(cases)
        all_groups.extend(result["groups"])
        summaries.append(result["summary"])
        _save_montage(
            cases,
            image_paths,
            montage_dir / f"{candidate_name}_vs_{baseline_name}.png",
            seed,
            max_per_group,
        )
    cases_frame = pd.concat(all_cases, ignore_index=True)
    groups_frame = pd.DataFrame(all_groups)
    cases_frame["image_path"] = cases_frame["image_id"].map(lambda image_id: str(image_paths[image_id]))
    groups_frame.to_csv(output / "error_rescue_harm.csv", index=False)
    cases_frame.to_csv(output / "error_cases.csv", index=False)
    metrics_rows = []
    for summary in summaries:
        for side in ("baseline", "candidate"):
            metrics_rows.append({
                "comparison": summary["comparison"],
                "role": side,
                "representation": summary[side],
                **summary[f"{side}_metrics"],
            })
    pd.DataFrame(metrics_rows).to_csv(output / "metrics_by_comparison.csv", index=False)
    input_manifest = {
        "threshold": threshold,
        "seed": seed,
        "n_validation": len(image_paths),
        "validation_manifest": str(Path(validation_manifest)),
        "inputs": {name: {"path": str(path), "sha256": _hash_file(Path(path))} for name, path in prediction_paths.items()},
        "comparisons": comparisons,
        "selection_rule": "fixed threshold 0.5; no retuning",
    }
    (output / "run_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")
    (output / "error_rescue_harm_summary.json").write_text(
        json.dumps({"protocol": input_manifest, "comparisons": summaries}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"summaries": summaries, "output_dir": str(output), "n_cases": len(cases_frame)}
