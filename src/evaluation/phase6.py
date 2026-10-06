"""Validate saved probes, inspect errors and evaluate fixed image corruptions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


REPRESENTATIONS = ("global_only", "local_only", "global_local")
CONDITIONS = {
    "clean": {"operation": "original RGB image"},
    "jpeg_q70": {"operation": "JPEG encode/decode before processor", "quality": 70, "subsampling": 2},
    "resize_112_224": {"operation": "bilinear resize before processor", "down_size": 112, "up_size": 224},
    "blur_r1": {"operation": "Gaussian blur before processor", "radius_in_original_image_pixels": 1.0},
}
ARTIFACT_FILES = {
    "phase4": ("cache_manifest.json", "processor.json", "metadata.csv", "global_features.pt",
               "patch_features.pt", "labels.pt", "train.csv", "val.csv"),
    "phase5": ("results.json", "config.yaml", "cache_manifest.json", "validation_predictions.csv",
               *(f"{name}/best_classifier.pt" for name in REPRESENTATIONS)),
    "phase3": ("results.json", "processor.json", "mlp/best_classifier.pt", "environment.json"),
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_roots(path: Path, required: tuple[str, ...]) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    marker = required[0]
    prefixes = {name[:-len(marker)] for name in names if name == marker or name.endswith("/" + marker)}
    return sorted(prefix for prefix in prefixes if all(prefix + name in names for name in required))


def extract_zip(path: Path, destination: Path) -> None:
    """Extract a selected artifact without allowing entries to escape its directory."""
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError(f"Invalid ZIP member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP contains a symlink: {member.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)


def find_artifact(input_root: Path, working_root: Path, kind: str, explicit=None, required=True):
    """Find a complete directory or ZIP; ambiguous inputs require an explicit path."""
    files = ARTIFACT_FILES[kind]
    if explicit is not None:
        path = Path(explicit)
        candidates = [path] if path.is_dir() else []
        archives = [path] if path.is_file() and path.suffix.lower() == ".zip" else []
    else:
        candidates = [p.parent for p in sorted(input_root.rglob(files[0]))]
        archives = sorted(input_root.rglob("*.zip"))
    directories = [p for p in candidates if all((p / name).is_file() for name in files)]
    if len(directories) == 1:
        return directories[0].resolve()
    if len(directories) > 1:
        raise ValueError(f"Multiple {kind} artifacts found. Set its ROOT explicitly: {directories}")
    matches = []
    for archive in archives:
        if zipfile.is_zipfile(archive):
            matches.extend((archive, prefix) for prefix in _zip_roots(archive, files))
    if len(matches) > 1:
        raise ValueError(f"Multiple {kind} ZIP artifacts found. Set its ROOT explicitly: {matches}")
    if matches:
        archive, prefix = matches[0]
        destination = working_root / f"{kind}_{file_hash(archive)[:12]}"
        extract_zip(archive, destination)
        return (destination / prefix).resolve()
    if required or explicit is not None:
        raise FileNotFoundError(f"No complete {kind} artifact at {explicit or input_root}; needs {files}")
    return None


def validate_saved_rows(metadata: list[dict], manifests: dict[str, list[dict]]) -> list[int]:
    ids = [row["image_id"] for row in metadata]
    if len(ids) != 2000 or len(set(ids)) != len(ids):
        raise ValueError("Cache must contain exactly 2000 unique image IDs")
    if {row["split"] for row in metadata} != {"train", "val"}:
        raise ValueError("Unexpected split in cache metadata")
    for name, count in (("train", 1600), ("val", 400)):
        rows = [row for row in metadata if row["split"] == name]
        expected = manifests[name]
        if len(rows) != count or len(expected) != count:
            raise ValueError(f"Wrong {name} sample count")
        if [(r["image_id"], int(r["label"])) for r in rows] != [
            (r["image_id"], int(r["label"])) for r in expected
        ]:
            raise ValueError(f"Cached {name} ID/label order differs from saved manifest")
        labels = [int(r["label"]) for r in rows]
        if labels.count(0) != count // 2 or labels.count(1) != count // 2:
            raise ValueError(f"{name} labels do not match the locked balanced split")
    return [i for i, row in enumerate(metadata) if row["split"] == "val"]


def resolve_validation_images(validation_rows: list[dict], train_root: Path) -> list[Path]:
    """Use original_path so a new Kaggle mount can differ from the extraction run."""
    original_labels = {r["path"].replace("\\", "/"): int(r["label"])
                       for r in read_rows(train_root / "manifest.csv")}
    paths = []
    for row in validation_rows:
        relative = PurePosixPath(row["original_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid original_path: {relative}")
        if original_labels.get(str(relative)) != int(row["label"]):
            raise ValueError(f"Mounted dataset label/path mismatch: {relative}")
        path = train_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def load_phase4(root: Path) -> dict:
    import pandas as pd
    import torch

    manifest = json.loads((root / "cache_manifest.json").read_text())
    if (manifest["model_id"] != "facebook/dinov3-vitb16-pretrain-lvd1689m"
            or not manifest.get("model_revision") or manifest["seed"] != 42
            or manifest["image_size"] != 224 or manifest["feature_dtype"] != "float16"
            or not manifest.get("patch_tokens_exclude_cls_and_registers")
            or manifest.get("source_metadata_available") is not False
            or manifest.get("split_counts") != {"train": 1600, "val": 400}):
        raise ValueError("Phase 4 cache does not match the locked DINOv3 protocol")
    rows = read_rows(root / "metadata.csv")
    manifests = {name: read_rows(root / f"{name}.csv") for name in ("train", "val")}
    indices = validate_saved_rows(rows, manifests)
    tensors = {name: torch.load(root / filename, map_location="cpu", weights_only=True, mmap=True)
               for name, filename in (("global", "global_features.pt"), ("patches", "patch_features.pt"),
                                      ("labels", "labels.pt"))}
    for name, shape_key, expected in (("global", "global_shape", [2000, 768]),
                                      ("patches", "patch_shape", [2000, 196, 768]),
                                      ("labels", "labels_shape", [2000])):
        if list(tensors[name].shape) != manifest[shape_key] or manifest[shape_key] != expected:
            raise ValueError(f"Wrong cached {name} shape")
    if [int(r["label"]) for r in rows] != tensors["labels"].tolist():
        raise ValueError("Persisted labels do not match metadata")
    if tensors["global"].dtype != torch.float16 or tensors["patches"].dtype != torch.float16:
        raise ValueError("Cache tensor dtype differs from declared float16")
    global_values = tensors["global"][indices].float()
    patches = tensors["patches"][indices].float()
    if not torch.isfinite(global_values).all() or not torch.isfinite(patches).all():
        raise ValueError("Validation cache contains NaN or infinity")
    local_values = patches.mean(dim=1)
    return {"manifest": manifest, "processor": json.loads((root / "processor.json").read_text()),
            "metadata": pd.DataFrame([rows[i] for i in indices]).assign(label=lambda x: x.label.astype(int)),
            "validation_rows": manifests["val"],
            "features": {"global": global_values, "local": local_values},
            "counts": {"train": 1600, "val": 400}}


def load_classifier(path: Path, input_dim: int, device):
    import torch
    from src.models.classifier import MLPClassifier

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # The frozen MLP protocol is shared across training seeds; seed is recorded
    # for provenance but must not prevent replay of a valid alternate-seed head.
    if checkpoint["classifier_name"] != "mlp" or checkpoint["threshold"] != 0.5:
        raise ValueError(f"Checkpoint does not match the frozen MLP protocol: {path}")
    settings = checkpoint.get("classifier_config", checkpoint)
    if (int(settings["hidden_dim"]) != 256 or float(settings["dropout"]) != 0.0
            or int(settings.get("num_classes", 2)) != 2):
        raise ValueError(f"Checkpoint architecture differs from the locked MLP: {path}")
    if checkpoint["classifier"]["layers.0.weight"].shape[1] != input_dim:
        raise ValueError(f"Wrong classifier input dimension: {path}")
    model = MLPClassifier(input_dim, hidden_dim=int(settings["hidden_dim"]),
                          num_classes=2, dropout=float(settings["dropout"]))
    model.load_state_dict(checkpoint["classifier"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def load_phase5(root: Path, cache: dict, device):
    import yaml

    saved_cache = json.loads((root / "cache_manifest.json").read_text())
    for key in ("model_id", "model_revision", "seed", "image_size", "feature_dtype",
                "global_shape", "patch_shape", "labels_shape", "patch_tokens_exclude_cls_and_registers"):
        if saved_cache[key] != cache["manifest"][key]:
            raise ValueError(f"Phase 5 used a different cache: {key}")
    config = yaml.safe_load((root / "config.yaml").read_text())
    if (config["model"].get("local_pooling") != "mean"
            or config["model"].get("fusion") != "concat"
            or config["model"].get("normalize_branches") is not False):
        raise ValueError("Phase 5 pooling/fusion protocol differs")
    if (config["experiment"]["seed"] != 42 or config["data"]["image_size"] != 224
            or config["model"]["backbone"] != cache["manifest"]["model_id"]
            or config["model"]["freeze_backbone"] is not True
            or config["model"]["classifier"] != "mlp"
            or config["model"]["hidden_dim"] != 256 or config["model"]["dropout"] != 0.0
            or config["model"]["num_classes"] != 2 or config["evaluation"]["threshold"] != 0.5):
        raise ValueError("Phase 5 config differs from the locked model/split protocol")
    results = json.loads((root / "results.json").read_text())
    heads, checkpoints = {}, {}
    for name in REPRESENTATIONS:
        heads[name], checkpoints[name] = load_classifier(
            root / name / "best_classifier.pt", 1536 if name == "global_local" else 768, device
        )
        if (checkpoints[name].get("representation") != name
                or checkpoints[name]["best_epoch"] != results[name]["best_epoch"]
                or checkpoints[name].get("local_pooling") != "mean"):
            raise ValueError(f"Phase 5 result/checkpoint mismatch: {name}")
    return heads, checkpoints, results, config


def predict_features(heads: dict, features: dict, metadata, device, condition: str, batch_size=32):
    import pandas as pd
    import torch

    tensors = {"global_only": features["global"], "local_only": features["local"],
               "global_local": torch.cat([features["global"], features["local"]], dim=1)}
    if not all(len(tensor) == len(metadata) for tensor in tensors.values()):
        raise ValueError("Feature and metadata lengths differ")
    frames = []
    with torch.inference_mode():
        for name, model in heads.items():
            model.eval()
            outputs = [model(batch.to(device)).softmax(dim=1).cpu() for batch in tensors[name].split(batch_size)]
            probabilities = torch.cat(outputs)
            if not torch.isfinite(probabilities).all():
                raise ValueError(f"Non-finite probabilities: {name}/{condition}")
            frame = metadata.copy()
            frame["representation"] = name
            frame["condition"] = condition
            frame["prob_real"] = probabilities[:, 0].numpy()
            frame["prob_fake"] = probabilities[:, 1].numpy()
            frame["prediction"] = (frame.prob_fake >= 0.5).astype(int)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_predictions(predictions):
    import pandas as pd
    from src.evaluation.metrics import compute_metrics

    rows = []
    for (condition, name), group in predictions.groupby(["condition", "representation"], sort=False):
        values = compute_metrics(group.label.to_numpy(), group.prob_fake.to_numpy(), threshold=0.5)
        rows.append({"condition": condition, "representation": name, "n_images": len(group), **values})
    return pd.DataFrame(rows)


def verify_phase5_replay(root: Path, replay, results: dict):
    import numpy as np
    import pandas as pd

    saved = pd.read_csv(root / "validation_predictions.csv", keep_default_na=False)
    if (len(saved) != 1200 or set(saved.representation) != set(REPRESENTATIONS)
            or saved.duplicated(["representation", "image_id"]).any()):
        raise ValueError("Phase 5 predictions must have one row per image and representation")
    rows = []
    for name in REPRESENTATIONS:
        old = saved[saved.representation == name].sort_values("image_id").reset_index(drop=True)
        new = replay[replay.representation == name].sort_values("image_id").reset_index(drop=True)
        if old.image_id.tolist() != new.image_id.tolist() or not np.array_equal(old.label, new.label):
            raise ValueError(f"Phase 5 validation IDs/labels differ: {name}")
        if not np.isfinite(old[["prob_real", "prob_fake"]].to_numpy()).all():
            raise ValueError(f"Non-finite Phase 5 predictions: {name}")
        if not np.array_equal(old.prediction, (old.prob_fake >= 0.5).astype(int)):
            raise ValueError(f"Phase 5 saved predictions use a different threshold: {name}")
        probability_delta = float(np.max(np.abs(
            old[["prob_real", "prob_fake"]].to_numpy() - new[["prob_real", "prob_fake"]].to_numpy()
        )))
        if probability_delta > 1e-4:
            raise ValueError(f"Cannot reproduce Phase 5 saved checkpoint: {name}, max delta={probability_delta}")
        metrics = score_predictions(new).iloc[0]
        for metric in ("auroc", "balanced_accuracy", "accuracy", "f1"):
            if abs(float(metrics[metric]) - results[name]["metrics"][f"val_{metric}"]) > 1e-6:
                raise ValueError(f"Phase 5 replay metric differs: {name}/{metric}")
        rows.append({"representation": name, "max_probability_difference": probability_delta, "status": "passed"})
    return pd.DataFrame(rows)


def error_groups(predictions):
    import pandas as pd

    base = predictions[predictions.representation == "global_only"].set_index("image_id")
    frames = []
    for peer in ("global_local", "local_only"):
        other = predictions[predictions.representation == peer].set_index("image_id").loc[base.index]
        frame = base[["label", "source", "split"]].copy()
        frame["comparison"] = f"global_only_vs_{peer}"
        frame["global_prob_fake"] = base.prob_fake
        frame["peer_prob_fake"] = other.prob_fake
        frame["global_prediction"] = base.prediction
        frame["peer_prediction"] = other.prediction
        frame["global_confidence"] = base[["prob_real", "prob_fake"]].max(axis=1)
        frame["peer_confidence"] = other[["prob_real", "prob_fake"]].max(axis=1)
        global_correct = base.prediction == base.label
        peer_correct = other.prediction == other.label
        frame["group"] = ["both_correct" if g and p else "both_wrong" if not g and not p
                          else "global_wrong_peer_correct" if p else "global_correct_peer_wrong"
                          for g, p in zip(global_correct, peer_correct)]
        frames.append(frame.reset_index())
    cases = pd.concat(frames, ignore_index=True)
    counts = cases.groupby(["comparison", "group"]).size().rename("n_images").reset_index()
    return cases, counts


def save_error_montages(cases, image_paths: dict, output_dir: Path, seed=42, per_group=8):
    import pandas as pd
    from PIL import Image, ImageDraw, ImageOps

    output_dir.mkdir(parents=True, exist_ok=True)
    samples, files = [], []
    for (comparison, group), rows in cases.groupby(["comparison", "group"]):
        if group == "both_correct":
            continue
        selected = rows.sort_values("image_id").sample(n=min(per_group, len(rows)), random_state=seed).copy()
        canvas = Image.new("RGB", (4 * 224, ((len(selected) + 3) // 4) * 266), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(selected.itertuples()):
            with Image.open(image_paths[row.image_id]) as image:
                tile = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
            x, y = (i % 4) * 224, (i // 4) * 266
            canvas.paste(tile, (x, y))
            draw.text((x + 3, y + 225), f"{row.image_id} label={row.label}\nG={row.global_prob_fake:.3f} peer={row.peer_prob_fake:.3f}", fill="black")
        path = output_dir / f"{comparison}__{group}.png"
        canvas.save(path)
        files.append(path)
        selected["montage"] = path.name
        selected["observation"] = ""
        samples.append(selected)
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(output_dir / "sampled_error_cases.csv", index=False)
    return files


def corrupt_image(image, condition: str):
    from PIL import Image, ImageFilter

    if condition == "clean":
        return image.copy()
    if condition == "jpeg_q70":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if condition == "resize_112_224":
        return image.resize((112, 112), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
    if condition == "blur_r1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    raise ValueError(f"Unknown condition: {condition}")


def extract_validation(extractor, processor, image_paths, metadata, device,
                       condition="clean", precision="amp_float16", batch_size=32):
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    if precision not in ("amp_float16", "float32"):
        raise ValueError(f"Unsupported precision: {precision}")

    class ValidationImages(Dataset):
        def __len__(self):
            return len(image_paths)

        def __getitem__(self, i):
            with Image.open(image_paths[i]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image = corrupt_image(image, condition)
                return processor(images=image, return_tensors="pt")["pixel_values"][0]

    loader = DataLoader(ValidationImages(), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    extractor.eval()
    global_parts, local_parts = [], []
    use_amp = precision == "amp_float16"
    with torch.inference_mode():
        for pixels in tqdm(loader, desc=f"{condition}/{precision}"):
            if tuple(pixels.shape[1:]) != (3, 224, 224):
                raise ValueError(f"Processor output shape changed: {tuple(pixels.shape)}")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = extractor(pixels.to(device, non_blocking=True))
            global_values, patch_values = features["global"].cpu(), features["patches"].cpu()
            if use_amp:
                global_values, patch_values = global_values.half(), patch_values.half()
            if tuple(patch_values.shape[1:]) != (196, 768):
                raise ValueError("Patch dimensions differ from Phase 4")
            global_parts.append(global_values.float())
            local_parts.append(patch_values.float().mean(dim=1))
    result = {"global": torch.cat(global_parts), "local": torch.cat(local_parts)}
    if len(metadata) != len(result["global"]) or not all(torch.isfinite(t).all() for t in result.values()):
        raise ValueError("Invalid extracted validation features")
    return result


def compare_features(reference: dict, regenerated: dict, name: str):
    import pandas as pd
    import torch

    rows = []
    for branch in ("global", "local"):
        old, new = reference[branch], regenerated[branch]
        difference = (old - new).abs()
        similarity = torch.nn.functional.cosine_similarity(old, new, dim=1)
        rows.append({"comparison": name, "branch": branch,
                     "mean_absolute_difference": float(difference.mean()),
                     "max_absolute_difference": float(difference.max()),
                     "mean_cosine_similarity": float(similarity.mean()),
                     "min_cosine_similarity": float(similarity.min())})
    return pd.DataFrame(rows)


def robustness_deltas(metrics):
    base = metrics[metrics.condition == "clean"].set_index("representation")
    output = metrics.copy()
    for metric in ("auroc", "balanced_accuracy", "accuracy", "f1"):
        output[f"delta_{metric}_vs_clean"] = [float(row[metric] - base.loc[row.representation, metric])
                                              for _, row in output.iterrows()]
    return output


def save_robustness_plot(metrics, output_path: Path):
    import matplotlib.pyplot as plt

    conditions = list(CONDITIONS)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("auroc", "balanced_accuracy")):
        for name in REPRESENTATIONS:
            rows = metrics[metrics.representation == name].set_index("condition")
            axis.plot(conditions, rows.loc[conditions, metric], marker="o", label=name)
        axis.set_ylabel(metric)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
