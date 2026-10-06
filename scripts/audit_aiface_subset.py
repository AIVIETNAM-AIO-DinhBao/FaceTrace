"""Run deterministic integrity checks on an extracted AI-Face subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = REPO_ROOT / "data/AI_Face/aiface_pilot_32k"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = args.subset_dir.resolve()
    with (subset / "manifests" / "all.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    issues: list[str] = []
    hash_groups: dict[str, list[str]] = defaultdict(list)
    generator_splits: dict[str, set[str]] = defaultdict(set)
    identities_by_split: dict[str, set[str]] = defaultdict(set)
    path_seen: set[str] = set()
    mode_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        image_path = row["image_path"]
        if image_path in path_seen:
            issues.append(f"duplicate_manifest_path:{image_path}")
        path_seen.add(image_path)
        path = subset / image_path
        if not path.is_file():
            issues.append(f"missing_file:{image_path}")
            continue
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                mode_counts[image.mode] += 1
        except Exception as exc:  # pragma: no cover - diagnostic output
            issues.append(f"decode_error:{image_path}:{type(exc).__name__}")
        digest = sha256(path)
        hash_groups[digest].append(image_path)
        if row["family"] != "Real":
            generator_splits[row["generator_id"]].add(row["split"])
        elif row.get("identity_hint"):
            identities_by_split[row["split"]].add(row["identity_hint"])

    for generator, splits in sorted(generator_splits.items()):
        if len(splits) != 1:
            issues.append(f"generator_overlap:{generator}:{','.join(sorted(splits))}")

    duplicate_groups = [paths for paths in hash_groups.values() if len(paths) > 1]
    for paths in duplicate_groups:
        issues.append("duplicate_content:" + ",".join(sorted(paths)))

    split_counts = defaultdict(int)
    label_counts = defaultdict(int)
    generator_counts = defaultdict(int)
    for row in rows:
        split_counts[row["split"]] += 1
        label_counts[(row["split"], row["label"])] += 1
        generator_counts[(row["split"], row["family"], row["generator_id"], row["label"])] += 1

    identity_overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        identity_overlap[f"{left}/{right}"] = len(
            identities_by_split[left] & identities_by_split[right]
        )

    expected = {
        "n_rows": 32000,
        "n_unique_sha256": len(hash_groups),
        "n_duplicate_groups": len(duplicate_groups),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {f"{split}/{label}": count for (split, label), count in sorted(label_counts.items())},
        "generator_splits": {key: sorted(value) for key, value in sorted(generator_splits.items())},
        "real_identity_counts": {
            key: len(value) for key, value in sorted(identities_by_split.items())
        },
        "real_identity_overlap": identity_overlap,
        "mode_counts": dict(sorted(mode_counts.items())),
        "issues": issues,
        "passed": not issues and len(rows) == 32000,
    }
    audit_dir = subset / "audit"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "quality_gate.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    with (audit_dir / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "n_images", "image_paths"])
        writer.writeheader()
        for digest, paths in sorted(hash_groups.items()):
            if len(paths) > 1:
                writer.writerow({"sha256": digest, "n_images": len(paths), "image_paths": "|".join(sorted(paths))})
    print(json.dumps(expected, indent=2))
    if issues or len(rows) != 32000:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
