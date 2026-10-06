"""Repair exact duplicate real images in an extracted AI-Face subset.

The pilot subset is already extracted, so this script replaces only duplicate
real rows with deterministic, unused candidates from the same annotation
pool. It then rewrites manifests and emits a bounded integrity report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/AI_Face"
DEFAULT_SUBSET = DEFAULT_DATA_ROOT / "aiface_pilot_32k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_annotations(data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    root = data_root / "AI_Face_annotationsV2"
    for filename in ("train_data.csv", "test_data.csv"):
        with (root / filename).open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_member(annotation_path: str) -> str:
    parts = [part for part in annotation_path.replace("\\", "/").split("/") if part]
    real_index = parts.index("Real")
    return "/".join(["imdb_wiki", *parts[real_index + 2 :]])


def extract_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        result = subprocess.run(
            ["unzip", "-p", str(archive), member],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0 or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Could not extract {member}: {result.stderr.decode(errors='replace')}"
        )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "image_id", "image_path", "label", "class_name", "split",
        "family", "generator_id", "real_source", "original_path",
        "annotation_split", "identity_hint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["image_id"]))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    subset = args.subset_dir.resolve()
    manifest_path = subset / "manifests" / "all.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    annotations = read_annotations(data_root)
    by_path = {row["Image Path"].replace("\\", "/"): row for row in annotations}

    digest_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest_groups[sha256(subset / row["image_path"])].append(row)
    duplicate_groups = [group for group in digest_groups.values() if len(group) > 1]
    if not duplicate_groups:
        print("No exact duplicate groups found; nothing to repair.")
        return

    selected_paths = {row["original_path"] for row in rows}
    selected_identity_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["identity_hint"]:
            selected_identity_by_split[row["split"]].add(row["identity_hint"])

    real_candidates = [
        row for row in annotations
        if row.get("Target") == "0"
        and "/Real/imdb_wiki/" in row.get("Image Path", "")
    ]
    archive = data_root / "imdb_wiki.zip"
    replacements: list[dict[str, str]] = []

    # Replace one row per duplicate group, preferring an unused identity in the
    # same split and matching the original broad demographic strata.
    for group_index, group in enumerate(sorted(duplicate_groups, key=lambda g: g[0]["image_path"])):
        victim = sorted(group, key=lambda row: (row["split"], row["image_path"]))[-1]
        old_path = victim["original_path"]
        old_annotation = by_path.get(old_path)
        if old_annotation is None:
            raise KeyError(f"Missing annotation for {old_path}")

        candidates = []
        for candidate in real_candidates:
            path = candidate["Image Path"].replace("\\", "/")
            if path in selected_paths:
                continue
            identity = path.rsplit("/", 1)[-1].split("_", 1)[0]
            if identity and identity in selected_identity_by_split[victim["split"]]:
                continue
            if candidate.get("Predicted Gender", "") != old_annotation.get("Predicted Gender", ""):
                continue
            if candidate.get("Skin Tone Group", "") != old_annotation.get("Skin Tone Group", ""):
                continue
            candidates.append((path, candidate, identity))
        if not candidates:
            # Demographic matching is a preference, not a hard requirement.
            for candidate in real_candidates:
                path = candidate["Image Path"].replace("\\", "/")
                if path in selected_paths:
                    continue
                identity = path.rsplit("/", 1)[-1].split("_", 1)[0]
                if identity and identity in selected_identity_by_split[victim["split"]]:
                    continue
                candidates.append((path, candidate, identity))
        if not candidates:
            raise RuntimeError(f"No unused real candidate for {victim['image_path']}")

        candidate_path, candidate, identity = sorted(candidates)[group_index % len(candidates)]
        destination = subset / victim["image_path"]
        with tempfile.TemporaryDirectory(prefix="aiface_repair_") as temp_dir:
            temporary = Path(temp_dir) / Path(candidate_path).name
            extract_member(archive, archive_member(candidate_path), temporary)
            os.replace(temporary, destination)

        selected_paths.remove(old_path)
        selected_paths.add(candidate_path)
        selected_identity_by_split[victim["split"]].discard(victim.get("identity_hint", ""))
        if identity:
            selected_identity_by_split[victim["split"]].add(identity)
        victim["original_path"] = candidate_path
        victim["identity_hint"] = identity
        replacements.append({
            "image_path": victim["image_path"],
            "split": victim["split"],
            "old_original_path": old_path,
            "new_original_path": candidate_path,
            "new_sha256": sha256(destination),
        })

    for split in ("train", "val", "test"):
        write_manifest(subset / "manifests" / f"{split}.csv", [row for row in rows if row["split"] == split])
    write_manifest(manifest_path, rows)

    audit_dir = subset / "audit"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "repair_report.json").write_text(
        json.dumps({"seed": args.seed, "replacements": replacements}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"repaired_groups": len(replacements), "replacements": replacements}, indent=2))


if __name__ == "__main__":
    main()
