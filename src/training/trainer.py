"""Minimal classifier training loop scaffold."""

import torch
from tqdm.auto import tqdm

from src.training.losses import classification_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = classification_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        num_samples += labels.size(0)
    return total_loss / max(num_samples, 1)
