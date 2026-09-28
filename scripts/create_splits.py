"""Create the fixed in-domain split used while source metadata is unavailable.

Run from the repository root with::

    python3 scripts/create_splits.py

The output CSV paths are relative to the repository root so the manifests remain
portable. An empty ``source`` field means that source identity is unknown; these
splits must not be described as an unseen-source evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data/Who_is_AI/data/train/manifest.csv")
DEFAULT_OUTPUT = Path("outputs/splits")
SEED = 42
VALIDATION_FRACTION = 0.20


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "class_name"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def create_splits(
    rows: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
    train_images_root: Path | None = None,
    image_path_base: Path = REPO_ROOT,
):
    train_images_root = (train_images_root or (REPO_ROOT / "data/Who_is_AI/data/train")).resolve()
    image_path_base = image_path_base.resolve()
    grouped: dict[str, list[dict[str, str]]] = {}
    seen_paths: set[str] = set()

    for row in rows:
        original_path = row["path"].strip().replace("\\", "/")
        if not original_path:
            raise ValueError("Manifest contains an empty image path")
        if original_path in seen_paths:
            raise ValueError(f"Duplicate manifest path: {original_path}")
        seen_paths.add(original_path)

        try:
            label = str(int(row["label"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected integer label, got {row['label']!r}") from exc

        absolute_path = train_images_root / original_path
        if not absolute_path.is_file():
            raise FileNotFoundError(f"Image listed in manifest does not exist: {absolute_path}")
        try:
            image_path_value = absolute_path.relative_to(image_path_base).as_posix()
        except ValueError:
            image_path_value = absolute_path.as_posix()

        normalized = {
            "image_id": f"train/{Path(original_path).name}",
            "image_path": image_path_value,
            "label": label,
            "class_name": row["class_name"].strip(),
            "source": "",
            "source_status": "unavailable",
            "original_path": original_path,
        }
        grouped.setdefault(label, []).append(normalized)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for label in sorted(grouped):
        class_rows = sorted(grouped[label], key=lambda r: r["image_id"])
        rng.shuffle(class_rows)
        val_count = round(len(class_rows) * validation_fraction)
        if not 0 < val_count < len(class_rows):
            raise ValueError(f"Invalid validation size for label {label}: {val_count}")
        val_rows.extend(class_rows[:val_count])
        train_rows.extend(class_rows[val_count:])

    for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
        for row in split_rows:
            row["split"] = split_name
        split_rows.sort(key=lambda r: r["image_id"])

    train_ids = {row["image_id"] for row in train_rows}
    val_ids = {row["image_id"] for row in val_rows}
    if train_ids & val_ids:
        raise AssertionError("Train and validation contain overlapping image IDs")
    if train_ids | val_ids != {row["image_id"] for rows_for_label in grouped.values() for row in rows_for_label}:
        raise AssertionError("Split manifests do not cover the full labeled dataset")

    return train_rows, val_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "image_id",
        "image_path",
        "label",
        "class_name",
        "source",
        "source_status",
        "split",
        "original_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--train-images-root", type=Path, default=REPO_ROOT / "data/Who_is_AI/data/train")
    parser.add_argument("--image-path-base", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    rows = read_manifest(args.manifest)
    train_rows, val_rows = create_splits(
        rows, args.validation_fraction, args.seed, args.train_images_root, args.image_path_base
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "val.csv", val_rows)

    counts = Counter((r["split"], r["label"], r["class_name"]) for r in train_rows + val_rows)
    summary_rows = [
        {"split": split, "label": label, "class_name": class_name, "source": "unknown", "n_images": count}
        for (split, label, class_name), count in sorted(counts.items())
    ]
    with (args.output_dir / "counts_by_split_label_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "label", "class_name", "source", "n_images"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path = args.manifest.resolve()
    try:
        manifest_value = manifest_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        manifest_value = manifest_path.as_posix()
    protocol = {
        "dataset_manifest": manifest_value,
        "seed": args.seed,
        "strategy": "stratified random split by label",
        "validation_fraction_per_class": args.validation_fraction,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "source_metadata_available": False,
        "unseen_source_test_available": False,
        "evaluation_claim": "in-domain/protocol sanity check only",
        "image_path_base": args.image_path_base.resolve().as_posix(),
        "preprocessing": {
            "image_size": 224,
            "transform": "Use the official processor for the configured DINOv3 checkpoint; no extra grayscale conversion or augmentation.",
            "threshold": 0.5,
        },
        "model_selection": "Choose classifier/checkpoint using validation AUROC; report validation Balanced Accuracy at the fixed 0.5 threshold.",
        "source_note": "Blank source values mean source/generator identity is unavailable, not that all images share one source.",
        "limitation": "This split cannot establish generalization to an unseen source or generator. Public/private test images are unlabeled and excluded.",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} validation rows to {args.output_dir}")
    for row in summary_rows:
        print(f"{row['split']:5} label={row['label']} ({row['class_name']}): {row['n_images']}")
    print("Source/generator metadata: unavailable; no unseen-source test was created.")


if __name__ == "__main__":
    main()
