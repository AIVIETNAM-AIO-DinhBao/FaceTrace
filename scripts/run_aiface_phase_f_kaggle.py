"""Kaggle wrapper for AI-Face Phase F robustness inference.

The attached image dataset may be either an already-unpacked package or a
single `.bin`/`.tar` archive.  Baseline and forensic inputs must be complete,
non-smoke outputs from the same training seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_dataset_root(input_root: Path) -> Path:
    candidates = []
    for root in sorted(input_root.rglob("*")):
        if root.is_dir() and all((root / split / "manifest.csv").is_file() for split in ("train", "val", "test")):
            candidates.append(root)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No unpacked AI-Face root with train/val/test manifests was found")
    raise RuntimeError(f"Multiple AI-Face roots found; pass --data-root explicitly: {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--forensic-root", required=True)
    parser.add_argument("--fusion-root", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/outputs/aiface_phase_f")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve() if args.data_root else find_dataset_root(Path("/kaggle/input"))
    command = [
        sys.executable, "-u", str(REPO_ROOT / "scripts/run_aiface_phase_f.py"),
        "--data-root", str(data_root), "--baseline-root", args.baseline_root,
        "--forensic-root", args.forensic_root, "--output-dir", args.output_dir,
        "--device", args.device, "--batch-size", str(args.batch_size),
    ]
    if args.fusion_root:
        command.extend(["--fusion-root", args.fusion_root])
    if args.smoke:
        command.append("--smoke")
    if args.conditions:
        command.extend(["--conditions", *args.conditions])
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
