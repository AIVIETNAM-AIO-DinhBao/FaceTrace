"""Feature caching, MLP fitting and portable spectral prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_metrics
from src.evaluation.spectral import save_predictions, select_validation_threshold
from src.input_data.spectral_dataset import SpectralDataset, SpectralImageTransform
from src.models.spectral import RadialFFT, SpectralMLP
from src.utils.seed import set_seed


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


@torch.inference_mode()
def extract_spectral_features(dataset, transform, device, batch_size=32, num_workers=0):
    transform = transform.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    features, labels, ids = [], [], []
    for batch in tqdm(loader, desc="Radial FFT features"):
        features.append(transform(batch["pixel_values"].to(device)).cpu())
        labels.append(batch["label"])
        ids.extend(str(value) for value in batch["image_id"])
    if not features:
        raise ValueError("Empty feature dataset")
    result = {"features": torch.cat(features), "labels": torch.cat(labels), "image_ids": ids}
    if not torch.isfinite(result["features"]).all():
        raise ValueError("Non-finite spectral features")
    return result


@torch.inference_mode()
def predict_features(model, features, device, batch_size=256) -> np.ndarray:
    model.eval()
    probabilities = []
    for batch in features.split(batch_size):
        probabilities.append(model(batch.to(device)).softmax(1)[:, 1].cpu())
    return torch.cat(probabilities).numpy()


def fit_spectral_probe(train, val, model_config, training_config, seed=42, device=None):
    device = device or select_device(training_config.get("device", "auto"))
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = SpectralMLP(train["features"].shape[1], model_config["hidden_dim"], model_config["dropout"])
    model.fit_normalization(train["features"])
    model.to(device)
    loader = DataLoader(TensorDataset(train["features"], train["labels"]),
                        batch_size=training_config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"],
                                 weight_decay=training_config["weight_decay"])
    best_score, best_state, best_epoch = (-np.inf, -np.inf), None, None
    history = []
    for epoch in range(1, training_config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(values.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        probabilities = predict_features(model, val["features"], device)
        metrics = compute_metrics(val["labels"].numpy(), probabilities, 0.5)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train["labels"]),
                        **{f"val_{key}": value for key, value in metrics.items()}})
        score = metrics["auroc"], metrics["balanced_accuracy"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch}: loss={history[-1]['train_loss']:.6f}, val AUROC={metrics['auroc']:.6f}")
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint selected")
    model.load_state_dict(best_state)
    probabilities = predict_features(model, val["features"], device)
    threshold, metrics = select_validation_threshold(val["labels"].numpy(), probabilities)
    return {"model": model, "state_dict": best_state, "history": history, "best_epoch": best_epoch,
            "threshold": threshold, "val_metrics": metrics, "val_probabilities": probabilities}


class SpectralPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = select_device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = payload["config"]
        self.metadata = payload["run_metadata"]
        self.threshold = float(payload["threshold"])
        data, model = self.config["data"], self.config["model"]
        self.image_transform = SpectralImageTransform(data["image_size"])
        self.fft = RadialFFT(data["image_size"], model["radial_bins"]).to(self.device)
        self.model = SpectralMLP(self.fft.output_dim, model["hidden_dim"], model["dropout"])
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path) -> float:
        if isinstance(image, (str, Path)):
            with Image.open(image) as loaded:
                pixels = self.image_transform(loaded)
        else:
            pixels = self.image_transform(image)
        return float(self.model.predict(self.fft(pixels.unsqueeze(0).to(self.device)))[0])

    def predict_manifest(self, manifest, image_path_base, output_dir, split="test", num_workers=0):
        dataset = SpectralDataset(manifest, image_path_base, self.config["data"]["image_size"])
        if set(dataset.frame["split"]) != {split}:
            raise ValueError("Manifest split does not match requested prediction split")
        cache = extract_spectral_features(dataset, self.fft, self.device,
                                          self.config["data"]["batch_size"], num_workers)
        if cache["image_ids"] != dataset.frame.image_id.astype(str).tolist():
            raise ValueError("Prediction ID order does not match manifest")
        probabilities = predict_features(self.model, cache["features"], self.device)
        return save_predictions(Path(output_dir), dataset.frame, probabilities, self.threshold, split)
