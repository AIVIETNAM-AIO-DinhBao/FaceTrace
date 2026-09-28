"""Training and validation for frozen-feature classification heads."""

import copy
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from src.evaluation.evaluate import evaluate_model


def train_one_epoch(model, loader, optimizer, device, mixed_precision: bool = False):
    model.train()
    total_loss = 0.0
    num_samples = 0
    use_amp = bool(mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="train", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.size(0)
        num_samples += labels.size(0)
    return total_loss / max(num_samples, 1)


def fit_probe(
    model,
    train_loader,
    validation_loader,
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    threshold: float,
    mixed_precision: bool,
    history_path: str | Path,
):
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    best = {"val_auroc": float("-inf"), "val_balanced_accuracy": float("-inf")}
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, mixed_precision=mixed_precision
        )
        metrics = evaluate_model(model, validation_loader, device, threshold)
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
        if score > (best["val_auroc"], best["val_balanced_accuracy"]):
            best = {f"val_{key}": value for key, value in metrics.items()}
            best_state = copy.deepcopy(model.classifier.state_dict())
            best_epoch = epoch

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    return best_state, best_epoch, best, history
