"""Create the small generator-disjoint AI-Face pilot subset.

The AI-Face download contains outer ZIP files with generator-specific ZIP files
inside them. This script streams those archives and extracts only the sampled
members needed for the 32k pilot; it never unpacks the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/AI_Face"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "aiface_pilot_32k"
SEED = 42
PER_GENERATOR = 2_000
REAL_TOTAL = 16_000

GENERATOR_ROLES = {
    ("GANs", "taming_transformer_VQGAN"): "train",
    ("GANs", "stylegan3"): "train",
    ("GANs", "AttGAN"): "val",
    ("GANs", "STARGAN"): "test",
    ("DMs", "StableDiffusion1.5"): "train",
    ("DMs", "latent_diffusion"): "train",
    ("DMs", "Palette"): "val",
    ("DMs", "StableDiffusion_Inpainting"): "test",
}

INNER_ARCHIVES = {
    ("GANs", "taming_transformer_VQGAN"): "taming_transformer_VQGAN.zip",
    ("GANs", "stylegan3"): "stylegan3.zip",
    ("GANs", "AttGAN"): "AttGAN.zip",
    ("GANs", "STARGAN"): "STARGAN.zip",
    ("DMs", "StableDiffusion1.5"): "StableDiffusion1.5.zip",
    ("DMs", "latent_diffusion"): "latent_diffusion.zip",
    ("DMs", "Palette"): "Palette.zip",
    ("DMs", "StableDiffusion_Inpainting"): "StableDiffusion_Inpainting.zip",
}

OUTER_ARCHIVES = {
    "GANs": "OneDrive_1_10-7-2026.zip",
    "DMs": "OneDrive_2_10-7-2026.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-generator", type=int, default=PER_GENERATOR)
    parser.add_argument("--real-total", type=int, default=REAL_TOTAL)
    return parser.parse_args()


def normalize_record(row: dict[str, str], source_split: str) -> dict[str, object]:
    path = (row.get("Image Path") or "").replace("\\", "/")
    parts = [part for part in path.split("/") if part]
    family = next((name for name in ("GANs", "DMs", "deepfakes", "Real") if name in parts), "<unknown>")
    index = parts.index(family) if family in parts else -1
    generator = parts[index + 1] if index >= 0 and index + 1 < len(parts) else "<unknown>"
    if family == "DMs" and generator == "CommercialTools" and index + 2 < len(parts):
        generator = f"CommercialTools/{parts[index + 2]}"

    basename = Path(path).name
    identity_match = re.match(r"(nm\d+|\d+)_", basename)
    return {
        "path": path,
        "source_split": source_split,
        "family": family,
        "generator": generator,
        "family_generator": (family, generator),
        "target": row.get("Target", ""),
        "identity_hint": identity_match.group(1) if identity_match else "",
        "row": row,
    }


def read_annotation(data_root: Path) -> list[dict[str, object]]:
    annotation_root = data_root / "AI_Face_annotationsV2"
    records: list[dict[str, object]] = []
    for filename, source_split in (("train_data.csv", "train_data"), ("test_data.csv", "test_data")):
        path = annotation_root / filename
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                records.append(normalize_record(row, source_split))
    return records


def sample_rows(rows: list[dict[str, object]], count: int, seed: int) -> list[dict[str, object]]:
    if len(rows) < count:
        raise ValueError(f"Need {count} rows but only found {len(rows)}")
    selected = sorted(rows, key=lambda item: str(item["path"]))
    random.Random(seed).shuffle(selected)
    return selected[:count]


def stratified_sample(
    rows: list[dict[str, object]], count: int, seed: int
) -> list[dict[str, object]]:
    """Sample real images while roughly preserving gender/skin-tone groups."""
    if len(rows) < count:
        raise ValueError(f"Need {count} real rows but only found {len(rows)}")
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in rows:
        row = item["row"]
        key = (row.get("Predicted Gender", ""), row.get("Skin Tone Group", ""))
        groups[key].append(item)

    total = sum(len(group) for group in groups.values())
    quotas: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    for key, group in groups.items():
        raw = count * len(group) / total
        quota = min(len(group), int(raw))
        quotas[key] = quota
        fractions.append((raw - quota, key))

    remaining = count - sum(quotas.values())
    for _, key in sorted(fractions, reverse=True):
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1

    selected: list[dict[str, object]] = []
    for offset, key in enumerate(sorted(groups)):
        group = sorted(groups[key], key=lambda item: str(item["path"]))
        random.Random(seed + offset).shuffle(group)
        selected.extend(group[: quotas[key]])
    random.Random(seed + 10_000).shuffle(selected)
    return selected[:count]


def suffix_for_record(item: dict[str, object]) -> str:
    path = str(item["path"])
    parts = [part for part in path.split("/") if part]
    family = str(item["family"])
    generator = str(item["generator"])
    family_index = parts.index(family)
    if family == "DMs" and generator.startswith("CommercialTools/"):
        generator_parts = generator.split("/")
        rest = parts[family_index + 3 :]
    else:
        rest = parts[family_index + 2 :]
    return "/".join(rest)


def run_pipeline(command: list[str], *, stdin=None, stdout=None) -> subprocess.Popen:
    return subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, text=stdout is None)


def list_nested_members(outer: Path, inner: str) -> list[str]:
    unzip = subprocess.Popen(["unzip", "-p", str(outer), inner], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert unzip.stdout is not None
    tar = subprocess.Popen(["bsdtar", "-tf", "-"], stdin=unzip.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    unzip.stdout.close()
    assert tar.stdout is not None
    members = [line.strip().lstrip("./") for line in tar.stdout if line.strip()]
    tar_stdout, tar_stderr = tar.communicate()
    unzip_stderr = unzip.communicate()[1]
    if tar.returncode != 0 or unzip.returncode != 0:
        raise RuntimeError(f"Could not list {inner}: {tar_stderr.decode(errors='replace') if isinstance(tar_stderr, bytes) else tar_stderr} {unzip_stderr.decode(errors='replace') if isinstance(unzip_stderr, bytes) else unzip_stderr}")
    return members


def extract_nested_members(outer: Path, inner: str, members: list[str], destination: Path) -> None:
    list_path = destination.parent / f"{inner}.members.txt"
    list_path.write_text("\n".join(members) + "\n", encoding="utf-8")
    try:
        unzip = subprocess.Popen(["unzip", "-p", str(outer), inner], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert unzip.stdout is not None
        tar = subprocess.Popen(
            ["bsdtar", "-xpf", "-", "-T", str(list_path), "-C", str(destination)],
            stdin=unzip.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        unzip.stdout.close()
        _, tar_stderr = tar.communicate()
        unzip_stderr = unzip.communicate()[1]
        if tar.returncode != 0 or unzip.returncode != 0:
            raise RuntimeError(
                f"Could not extract {inner}: {tar_stderr.decode(errors='replace')} {unzip_stderr.decode(errors='replace')}"
            )
    finally:
        list_path.unlink(missing_ok=True)


def extract_regular_members(archive: Path, members: list[str], destination: Path) -> None:
    list_path = destination.parent / "real.members.txt"
    list_path.write_text("\n".join(members) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["bsdtar", "-xpf", str(archive), "-T", str(list_path), "-C", str(destination)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not extract {archive}: {result.stderr}")
    finally:
        list_path.unlink(missing_ok=True)


def match_members(records: list[dict[str, object]], members: list[str]) -> dict[str, str]:
    by_basename: dict[str, list[str]] = defaultdict(list)
    for member in members:
        if member.lower().endswith((".jpg", ".jpeg", ".png")):
            by_basename[Path(member).name].append(member)

    matched: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for item in records:
        suffix = suffix_for_record(item)
        candidates = [member for member in by_basename.get(Path(suffix).name, []) if member == suffix or member.endswith("/" + suffix)]
        if len(candidates) == 1:
            matched[str(item["path"])] = candidates[0]
        elif not candidates:
            missing.append(suffix)
        else:
            ambiguous.append(f"{suffix}: {candidates[:3]}")
    if missing or ambiguous:
        raise RuntimeError(f"Archive/member mapping failed; missing={missing[:5]}, ambiguous={ambiguous[:5]}")
    return matched


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "image_id", "image_path", "label", "class_name", "split",
        "family", "generator_id", "real_source", "original_path",
        "annotation_split", "identity_hint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    manifest_root = output_dir / "manifests"
    audit_root = output_dir / "audit"
    image_root.mkdir()
    manifest_root.mkdir()
    audit_root.mkdir()

    records = read_annotation(data_root)
    selected_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for index, (key, role) in enumerate(GENERATOR_ROLES.items()):
        candidates = [item for item in records if item["family_generator"] == key and item["target"] == "1"]
        selected_by_key[key] = sample_rows(candidates, args.per_generator, args.seed + index * 1_000)

    real_candidates = [item for item in records if item["family_generator"] == ("Real", "imdb_wiki") and item["target"] == "0"]
    selected_real = stratified_sample(real_candidates, args.real_total, args.seed)
    random.Random(args.seed + 99).shuffle(selected_real)
    real_roles = [("train", 8_000), ("val", 4_000), ("test", 4_000)]
    selected_by_role: dict[str, list[dict[str, object]]] = defaultdict(list)
    cursor = 0
    for role, count in real_roles:
        selected_by_role[role].extend(selected_real[cursor : cursor + count])
        cursor += count
    real_role_by_path = {
        str(item["path"]): role
        for role, rows_for_role in selected_by_role.items()
        for item in rows_for_role
    }

    selected_records: list[dict[str, object]] = []
    for key, generator_records in selected_by_key.items():
        selected_records.extend(generator_records)
    selected_records.extend(selected_real)

    with tempfile.TemporaryDirectory(prefix="aiface_subset_") as temp_name:
        temp_root = Path(temp_name)
        extracted_by_path: dict[str, Path] = {}

        for key, generator_records in selected_by_key.items():
            family, generator = key
            outer = data_root / OUTER_ARCHIVES[family]
            inner = INNER_ARCHIVES[key]
            print(f"Listing {inner} ...")
            members = list_nested_members(outer, inner)
            matched = match_members(generator_records, members)
            selected_members = sorted(set(matched.values()))
            extract_root = temp_root / slug(generator)
            extract_root.mkdir(parents=True)
            print(f"Extracting {len(selected_members)} members from {inner} ...")
            extract_nested_members(outer, inner, selected_members, extract_root)
            for item in generator_records:
                extracted_by_path[str(item["path"])] = extract_root / matched[str(item["path"])]

        real_members = []
        real_match_by_path = {}
        for item in selected_real:
            path = str(item["path"])
            parts = [part for part in path.split("/") if part]
            real_index = parts.index("Real")
            member = "/".join(["imdb_wiki", *parts[real_index + 2 :]])
            real_members.append(member)
            real_match_by_path[path] = member
        real_root = temp_root / "imdb_wiki"
        real_root.mkdir()
        print(f"Extracting {len(real_members)} real members ...")
        extract_regular_members(data_root / "imdb_wiki.zip", sorted(set(real_members)), real_root)
        for item in selected_real:
            extracted_by_path[str(item["path"])] = real_root / real_match_by_path[str(item["path"])]

        manifest_rows: list[dict[str, str]] = []
        counters = Counter()
        role_by_key = GENERATOR_ROLES
        for item in selected_records:
            key = item["family_generator"]
            role = real_role_by_path[str(item["path"])] if item["family"] == "Real" else role_by_key[key]
            source_path = extracted_by_path[str(item["path"])]
            if not source_path.is_file():
                raise FileNotFoundError(f"Extracted member missing: {source_path}")
            family = str(item["family"])
            generator = str(item["generator"])
            image_id = f"{role}/{slug(family)}_{slug(generator)}_{counters[(role, family, generator)]:05d}"
            counters[(role, family, generator)] += 1
            extension = source_path.suffix.lower() or ".jpg"
            relative_path = Path("images") / role / f"{slug(family)}_{slug(generator)}_{counters[(role, family, generator)] - 1:05d}{extension}"
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            row = item["row"]
            manifest_rows.append({
                "image_id": image_id,
                "image_path": relative_path.as_posix(),
                "label": "0" if item["target"] == "0" else "1",
                "class_name": "real" if item["target"] == "0" else "fake",
                "split": role,
                "family": family,
                "generator_id": generator,
                "real_source": "imdb_wiki" if family == "Real" else "",
                "original_path": str(item["path"]),
                "annotation_split": str(item["source_split"]),
                "identity_hint": str(item["identity_hint"]),
            })

    for role in ("train", "val", "test"):
        write_csv(manifest_root / f"{role}.csv", sorted((row for row in manifest_rows if row["split"] == role), key=lambda row: row["image_id"]))
    write_csv(manifest_root / "all.csv", sorted(manifest_rows, key=lambda row: row["image_id"]))

    counts = Counter((row["split"], row["label"], row["family"], row["generator_id"]) for row in manifest_rows)
    with (audit_root / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "family", "generator_id", "n_images"])
        writer.writeheader()
        for (split, label, family, generator), count in sorted(counts.items()):
            writer.writerow({"split": split, "label": label, "family": family, "generator_id": generator, "n_images": count})

    metadata = {
        "dataset": "AI-Face v2",
        "subset_name": "aiface_pilot_32k",
        "seed": args.seed,
        "per_generator_fake": args.per_generator,
        "real_total": args.real_total,
        "total_images": len(manifest_rows),
        "selected_generators": [f"{family}/{generator}" for family, generator in GENERATOR_ROLES],
        "split_mapping": {f"{family}/{generator}": role for (family, generator), role in GENERATOR_ROLES.items()},
        "real_source": "imdb_wiki",
        "identity_disjointness_verified": False,
        "generator_disjoint": True,
        "preprocessing_note": "Original resolutions vary; resize must be fixed by the training pipeline.",
        "limitation": "Pilot uses imdb_wiki only for real images and does not establish identity-disjointness.",
    }
    (output_dir / "subset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Created {len(manifest_rows)} images under {output_dir}")
    print("Split counts:", Counter(row["split"] for row in manifest_rows))
    print("Label counts:", Counter(row["label"] for row in manifest_rows))


if __name__ == "__main__":
    main()
