"""Kaggle entrypoint for the AI-Face Global/Local Phase E experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_unpacked_root(input_root: Path) -> Path | None:
    candidates = []
    for root in sorted(input_root.rglob("*")):
        if not root.is_dir():
            continue
        if all((root / split / "manifest.csv").is_file() for split in ("train", "val", "test")):
            candidates.append(root)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple AI-Face package roots found: {candidates}")
    return None


def extract_single_bin(input_root: Path, working_root: Path) -> Path:
    archives = sorted(input_root.rglob("*.bin"))
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected exactly one AI-Face .bin archive, found: {archives}")
    archive = archives[0]
    target = (working_root / "aiface_phase_e_dataset").resolve()
    marker = target / ".extracted"
    if not marker.is_file():
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                destination = (target / member.name).resolve()
                if destination != target and target not in destination.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
            handle.extractall(target)
        marker.write_text("ok\n", encoding="utf-8")
        print(f"Extracted dataset archive: {archive} -> {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aiface_phase_e.yaml")
    parser.add_argument("--output-dir", default="outputs/aiface_phase_e")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_root = Path("/kaggle/input")
    working_root = Path("/kaggle/working")
    data_root = find_unpacked_root(input_root)
    if data_root is None:
        data_root = extract_single_bin(input_root, working_root)
    print(f"Dataset root: {data_root}")
    print("Split mode: generator_disjoint_fixed")
    print("Random fallback: disabled")

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_aiface_phase_e.py"),
        "--config", args.config,
        "--data-root", str(data_root),
        "--output-dir", args.output_dir,
    ]
    if args.smoke:
        command.append("--smoke")
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
