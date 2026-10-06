"""Training, checkpoint selection and prediction export for Week 3 image models."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_metrics


def _make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_image_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    mixed_precision: bool = False,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(mixed_precision and device.type == "cuda")
    total_loss = 0.0
    num_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pixels)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.size(0)
        num_samples += labels.size(0)
    return total_loss / max(num_samples, 1)


@torch.inference_mode()
def predict_image_model(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    row_ids, labels, probabilities, image_ids = [], [], [], []
    for batch in loader:
        logits = model(batch["pixel_values"].to(device, non_blocking=True))
        row_ids.append(torch.arange(len(batch["label"]), dtype=torch.long))
        labels.append(batch["label"].cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
        image_ids.extend(batch["image_id"])
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return {
        "labels": torch.cat(labels),
        "probabilities": torch.cat(probabilities),
        "image_id": image_ids,
    }


@torch.inference_mode()
def evaluate_image_model(model: nn.Module, loader, device: torch.device, threshold: float = 0.5):
    prediction = predict_image_model(model, loader, device)
    metrics = compute_metrics(
        prediction["labels"].numpy(), prediction["probabilities"].numpy(), threshold
    )
    return metrics, prediction


def fit_image_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    threshold: float,
    mixed_precision: bool,
    history_path: str | Path,
):
    """Fit an image model and select by clean validation AUROC then BAcc."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_amp = bool(mixed_precision and device.type == "cuda")
    scaler = _make_scaler(use_amp)
    history = []
    best_score = (float("-inf"), float("-inf"))
    best_state = None
    best_epoch = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_image_epoch(
            model, train_loader, optimizer, device, scaler, mixed_precision=mixed_precision
        )
        metrics, _ = evaluate_image_model(model, validation_loader, device, threshold)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.5f} "
            f"val_auroc={metrics['auroc']:.5f} val_bacc={metrics['balanced_accuracy']:.5f}"
        )
        score = (metrics["auroc"], metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = metrics

    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint was selected")
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    model.load_state_dict(copy.deepcopy(best_state))
    return best_state, best_epoch, best_metrics, history


def write_predictions(prediction: dict, path: str | Path, threshold: float = 0.5) -> None:
    """Write stable image-ID keyed predictions for later fusion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = prediction["probabilities"].numpy()
    labels = prediction["labels"].numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "label", "prob_real", "prob_fake", "predicted_label"]
        )
        writer.writeheader()
        for image_id, label, probs in zip(prediction["image_id"], labels, probabilities):
            writer.writerow(
                {
                    "image_id": image_id,
                    "label": int(label),
                    "prob_real": float(probs[0]),
                    "prob_fake": float(probs[1]),
                    "predicted_label": int(probs[1] >= threshold),
                }
            )
