"""Configuration and checkpoint I/O."""

from pathlib import Path

import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
