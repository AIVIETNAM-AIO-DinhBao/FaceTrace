"""Evaluate image batches and compute the configured binary metrics."""

import torch

from src.evaluation.metrics import compute_metrics


@torch.inference_mode()
def evaluate_model(model, loader, device, threshold: float = 0.5):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits = model(batch["pixel_values"].to(device, non_blocking=True))
        probabilities.append(logits.softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return compute_metrics(torch.cat(labels).numpy(), torch.cat(probabilities).numpy(), threshold)
