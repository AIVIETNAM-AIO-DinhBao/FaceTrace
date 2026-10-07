"""Locate the AI-Face .bin/.tar package, safely unpack it and run spectral training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_e import read_manifest, sha256_file
from src.evaluation.spectral import resolve_manifests, write_json


def find_dataset_root(root: Path) -> Path | None:
    candidates = set()
    if (root / "train/manifest.csv").is_file() or (root / "manifests/train.csv").is_file():
        candidates.add(root.resolve())
    for path in root.rglob("*.csv"):
        if path.name == "manifest.csv" and path.parent.name == "train":
            candidates.add(path.parent.parent.resolve())
        elif path.name == "train.csv" and path.parent.name == "manifests":
            candidates.add(path.parent.parent.resolve())
    valid = []
    for candidate in sorted(candidates):
        try:
            manifests, bases = resolve_manifests(candidate)
            # Identify an image package from its manifests and image files.
            # Metadata placement is checked separately by the trainer; it must
            # not hide valid train/val/test manifests during root discovery.
            for split, manifest in manifests.items():
                frame = read_manifest(manifest)
                sample = Path(str(frame.iloc[0]["image_path"]))
                image = sample if sample.is_absolute() else bases[split] / sample
                if not image.is_file():
                    raise FileNotFoundError(f"Sample {split} image is missing: {image}")
            valid.append(candidate)
        except FileNotFoundError:
            pass
    if len(valid) > 1:
        raise ValueError(f"Multiple dataset roots found; select one using --input-root: {valid}")
    return valid[0] if valid else None


def unpack_dataset(input_root: Path, work_root: Path) -> Path:
    direct = find_dataset_root(input_root)
    if direct:
        return direct
    archives = [input_root] if input_root.is_file() else sorted(
        path for path in input_root.rglob("*") if path.is_file() and
        (path.suffix.lower() in {".bin", ".tar"} or path.name.lower().endswith((".tar.gz", ".tgz")))
    )
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected one AI-Face archive; use --input-root to select it. Found: {archives}")
    archive = archives[0]
    digest = sha256_file(archive)
    target = (work_root / f"aiface_spectral_dataset_{digest[:12]}").resolve()
    marker = target / ".archive_metadata.json"
    if marker.is_file():
        if json.loads(marker.read_text())["sha256"] != digest:
            raise ValueError("Extraction marker checksum mismatch")
    else:
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            # Reject links and special files, as well as traversal/absolute paths.
            for member in members:
                name = Path(member.name)
                destination = (target / name).resolve()
                if name.is_absolute() or not destination.is_relative_to(target) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            # data filter adds platform-specific link/path protection on Python
            # versions supporting it. The checks above cover Python 3.10 too.
            if hasattr(tarfile, "data_filter"):
                handle.extractall(target, members=members, filter="data")
            else:
                handle.extractall(target, members=members)
        write_json(marker, {"archive": str(archive), "sha256": digest})
        print(f"Extracted dataset archive: {archive} -> {target}", flush=True)
    dataset_root = find_dataset_root(target)
    if dataset_root is None:
        inventory = sorted(str(path.relative_to(target)) for path in target.rglob("*")
                           if path.is_file() and path.suffix.lower() in {".csv", ".json"})
        raise FileNotFoundError(
            f"No readable image dataset with fixed train/val/test manifests under {target}. "
            f"CSV/JSON files found: {inventory[:40]}"
        )
    return dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/spectral.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--working-root", default="/kaggle/temp")
    parser.add_argument("--output-dir", default="/kaggle/working/outputs/aiface_phase_e/forensic_only")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = unpack_dataset(Path(args.input_root), Path(args.working_root))
    command = [sys.executable, str(REPO_ROOT / "scripts/train_spectral.py"), "--config", args.config,
               "--data-root", str(root), "--output-dir", args.output_dir]
    if args.smoke:
        command.append("--smoke")
    if args.device:
        command.extend(["--device", args.device])
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
