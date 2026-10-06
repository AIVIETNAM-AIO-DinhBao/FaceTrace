"""Prepare the attached Kaggle dataset and run the Week 3 residual baseline.

This entrypoint is designed for KRun. It keeps the dataset outside the source
package, discovers ``train/manifest.csv`` under ``/kaggle/input``, creates the
fixed in-domain split, and then invokes the normal Week 3 trainer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _csv_has_manifest_columns(path: Path) -> bool:
    """Accept a renamed manifest as long as its schema is compatible."""

    try:
        header = path.open(newline="", encoding="utf-8-sig").readline().strip().split(",")
    except (OSError, UnicodeDecodeError):
        return False
    return {"path", "label", "class_name"}.issubset(set(header))


def discover_data_root(explicit: str | None = None) -> Path:
    """Find the unique attached dataset containing a compatible train manifest.

    Kaggle preserves the directory layout used when a dataset is published, so
    the manifest may be nested below an extra folder or have been renamed from
    ``manifest.csv``. Search compatible CSVs rather than relying on one exact
    archive layout.
    """

    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "train" / "manifest.csv").is_file():
            return root
        raise FileNotFoundError(f"WEEK3_DATA_ROOT does not contain train/manifest.csv: {root}")

    candidates = []
    for input_root in (Path("/kaggle/input"), REPO_ROOT / "data"):
        if not input_root.exists():
            continue
        for manifest in sorted(input_root.rglob("*.csv")):
            if not _csv_has_manifest_columns(manifest):
                continue
            # Normal layout: <root>/train/manifest.csv and paths like images/x.jpg.
            roots_to_try = [manifest.parent.parent, manifest.parent]
            for root in roots_to_try:
                train_dir = root / "train"
                if (train_dir / "images").is_dir():
                    candidates.append(root.resolve())
                    break
            # Also accept a manifest directly inside a train directory when the
            # image folder is present but the parent folder is named differently.
            if (manifest.parent / "images").is_dir():
                candidates.append(manifest.parent.parent.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        detail = ", ".join(str(path) for path in candidates) or "none"
        visible = []
        input_root = Path("/kaggle/input")
        if input_root.exists():
            visible = [str(path.relative_to(input_root)) for path in sorted(input_root.rglob("*")) if path.is_file()][:40]
        raise RuntimeError(
            "Expected exactly one Who Is AI dataset root with a compatible train CSV "
            f"and train/images; found {len(candidates)}: {detail}. "
            "Set WEEK3_DATA_ROOT explicitly if multiple datasets are attached. "
            f"Visible input files: {visible}"
        )
    return candidates[0]


def load_split_module():
    path = REPO_ROOT / "scripts" / "create_splits.py"
    spec = importlib.util.spec_from_file_location("week3_create_splits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load split generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_runtime_splits(data_root: Path, split_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    split_module = load_split_module()
    rows = split_module.read_manifest(data_root / "train" / "manifest.csv")
    train_rows, val_rows = split_module.create_splits(
        rows,
        validation_fraction=0.20,
        seed=seed,
        train_images_root=data_root / "train",
        image_path_base=REPO_ROOT,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    split_module.write_csv(train_path, train_rows)
    split_module.write_csv(val_path, val_rows)
    return train_path, val_path


def write_runtime_config(train_manifest: Path, val_manifest: Path, output_dir: Path) -> Path:
    base_path = REPO_ROOT / "configs" / "week03.yaml"
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config["data"]["train_manifest"] = str(train_manifest)
    config["data"]["validation_manifest"] = str(val_manifest)
    config["data"]["image_path_base"] = str(REPO_ROOT)
    config["experiment"]["output_dir"] = str(output_dir)
    config_path = output_dir / "resolved_week03.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("WEEK3_DATA_ROOT"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/week03")
    args = parser.parse_args()

    data_root = discover_data_root(args.data_root)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    split_dir = output_dir / "splits"
    train_manifest, val_manifest = create_runtime_splits(data_root, split_dir, args.split_seed)
    config_path = write_runtime_config(train_manifest, val_manifest, output_dir)
    (output_dir / "input.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "train_manifest": str(train_manifest),
                "validation_manifest": str(val_manifest),
                "split_seed": args.split_seed,
                "evaluation_claim": "in-domain/protocol sanity check only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Dataset root: {data_root}")
    print(f"Train manifest: {train_manifest}")
    print(f"Validation manifest: {val_manifest}")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_residual.py"), "--config", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
