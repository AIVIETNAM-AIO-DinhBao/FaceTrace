"""Resolve saved Global/Local and spectral artifacts, then run CPU fusion.

Inputs can be saved Notebook Output directories or ZIPs uploaded as datasets.
No original images, FFT extraction, classifier training or HF_TOKEN is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aiface_phase_e import sha256_file
from src.evaluation.spectral import write_json


def required_files(role: str) -> list[str]:
    if role == "forensic":
        return ["run_metadata.json", "config_resolved.yaml", "environment.json", "checkpoint.pt",
                "predictions_val.csv", "predictions_test.csv", "val_metrics.json", "test_metrics_overall.json"]
    if role == "baseline":
        return ["run_metadata.json"] + [f"{branch}/{name}" for branch in
                ("global_only", "local_only", "global_local") for name in
                ("predictions_test.csv", "val_metrics.json", "test_metrics_overall.json", "test_metrics_by_generator.csv")]
    raise ValueError(f"Unknown artifact role: {role}")


def metadata_matches(metadata, role):
    if metadata.get("split_mode") != "generator_disjoint_fixed" or metadata.get("smoke") is not False:
        return False
    return role != "forensic" or metadata.get("model_name") in (None, "forensic_only")


def find_artifact_root(search_root: Path, role: str) -> Path | None:
    if not search_root.is_dir():
        return None
    roots = set()
    for path in search_root.rglob("run_metadata.json"):
        candidate = path.parent
        if not all((candidate / name).is_file() for name in required_files(role)):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata_matches(metadata, role):
            roots.add(candidate.resolve())
    if len(roots) > 1:
        raise ValueError(f"Multiple full {role} runs found; set --{role}-input explicitly: {sorted(roots)}")
    return next(iter(roots)) if roots else None


def archive_matches(path: Path, role: str) -> bool:
    with zipfile.ZipFile(path) as handle:
        files = set(handle.namelist())
        for name in files:
            if not name.endswith("run_metadata.json"):
                continue
            prefix = name[:-len("run_metadata.json")]
            if not all(prefix + required in files for required in required_files(role)):
                continue
            metadata = json.loads(handle.read(name))
            if metadata_matches(metadata, role):
                return True
    return False


def extract_artifact_zip(archive: Path, work_root: Path, role: str) -> Path:
    digest = sha256_file(archive)
    target = (work_root / f"{role}_{digest[:12]}").resolve()
    marker = target / ".artifact_archive.json"
    if not marker.is_file():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                destination = (target / member.filename).resolve()
                mode = member.external_attr >> 16
                if not destination.is_relative_to(target) or stat.S_ISLNK(mode):
                    raise ValueError(f"Unsafe artifact ZIP member: {member.filename}")
            handle.extractall(target)
        write_json(marker, {"archive": str(archive), "sha256": digest})
    else:
        if json.loads(marker.read_text())["sha256"] != digest:
            raise ValueError("Artifact extraction checksum mismatch")
    return target


def validate_validation_inputs(root: Path, role: str):
    if role != "baseline":
        return
    for branch in ("global_only", "local_only"):
        if (root / branch / "predictions_val.csv").is_file():
            continue
        if not (root / "feature_cache.pt").is_file() or not (root / branch / "checkpoint.pt").is_file():
            raise FileNotFoundError(
                f"Baseline {branch} needs predictions_val.csv, OR feature_cache.pt plus {branch}/checkpoint.pt. "
                "Ask the baseline owner for the full AI-Face output, not only test predictions/metrics."
            )


def resolve_artifact_input(search_root: Path, role: str, work_root: Path):
    search_root = search_root.resolve()
    direct = find_artifact_root(search_root, role)
    if direct:
        validate_validation_inputs(direct, role)
        return direct, {"kind": "unpacked_notebook_output", "input": str(search_root), "root": str(direct)}
    if search_root.is_file():
        archives = [search_root] if zipfile.is_zipfile(search_root) and archive_matches(search_root, role) else []
    else:
        archives = [path for path in sorted(search_root.rglob("*.zip"))
                    if zipfile.is_zipfile(path) and archive_matches(path, role)]
    if len(archives) != 1:
        raise FileNotFoundError(
            f"Expected one full {role} artifact directory/ZIP under {search_root}; found {archives}. "
            f"Set --{role}-input to the correct dataset directory or ZIP. Smoke artifacts do not qualify."
        )
    archive = archives[0]
    extracted = extract_artifact_zip(archive, work_root, role)
    root = find_artifact_root(extracted, role)
    if root is None:
        raise FileNotFoundError(f"No full {role} run in {archive}")
    validate_validation_inputs(root, role)
    return root, {"kind": "zip", "input": str(archive), "sha256": sha256_file(archive), "root": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forensic-input", default="/kaggle/input")
    parser.add_argument("--baseline-input", default="/kaggle/input")
    parser.add_argument("--working-root", default="/kaggle/temp/aiface_fusion_inputs")
    parser.add_argument("--output-dir", default="/kaggle/working/outputs/aiface_phase_e_fusion")
    args = parser.parse_args()
    work = Path(args.working_root)
    forensic, forensic_source = resolve_artifact_input(Path(args.forensic_input), "forensic", work)
    baseline, baseline_source = resolve_artifact_input(Path(args.baseline_input), "baseline", work)
    print(f"FORENSIC_DIR={forensic}", flush=True)
    print(f"BASELINE_DIR={baseline}", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "fusion_inputs.json", {"forensic": forensic_source, "baseline": baseline_source})
    command = [sys.executable, "-u", str(REPO_ROOT / "scripts/fuse_aiface_spectral.py"),
               "--forensic-dir", str(forensic), "--baseline-dir", str(baseline), "--output-dir", str(output)]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
