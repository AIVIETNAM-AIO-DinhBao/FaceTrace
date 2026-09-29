"""Training and evaluation for classifiers on cached tensor features."""

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


def train_feature_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool = False,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(mixed_precision and device.type == "cuda")
    scaler = _make_scaler(use_amp)
    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.size(0)
        num_samples += labels.size(0)
    return total_loss / max(num_samples, 1)


@torch.inference_mode()
def evaluate_feature_model(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        logits = model(features)
        probabilities.append(logits.softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return compute_metrics(torch.cat(labels).numpy(), torch.cat(probabilities).numpy(), threshold)


@torch.inference_mode()
def predict_feature_model(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    row_indices, labels, probabilities, image_ids = [], [], [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        logits = model(features)
        row_indices.append(torch.as_tensor(batch["row_index"]).cpu())
        labels.append(batch["label"].cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
        image_ids.extend(batch["image_id"])
    if not labels:
        raise ValueError("Prediction loader is empty")
    return {
        "row_index": torch.cat(row_indices),
        "labels": torch.cat(labels),
        "probabilities": torch.cat(probabilities),
        "image_id": image_ids,
    }


def fit_feature_probe(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_score = (float("-inf"), float("-inf"))
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        train_loss = train_feature_epoch(
            model, train_loader, optimizer, device, mixed_precision=mixed_precision
        )
        metrics = evaluate_feature_model(model, validation_loader, device, threshold)
        row = {"epoch": epoch, "train_loss": train_loss, **{
            f"val_{key}": value for key, value in metrics.items()
        }}
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.5f} "
            f"val_auroc={metrics['auroc']:.5f} "
            f"val_bacc={metrics['balanced_accuracy']:.5f}"
        )
        score = (metrics["auroc"], metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(copy.deepcopy(best_state))
    best_metrics = evaluate_feature_model(model, validation_loader, device, threshold)
    return best_state, best_epoch, {f"val_{key}": value for key, value in best_metrics.items()}, history
