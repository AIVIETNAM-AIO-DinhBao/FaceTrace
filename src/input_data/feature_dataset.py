"""Dataset wrapper for cached tensor features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Expose selected rows from a cached feature tensor for probe training."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        indices: Sequence[int] | torch.Tensor,
        image_ids: Sequence[str],
    ) -> None:
        if features.ndim != 2:
            raise ValueError("Expected pooled features with shape [samples, channels]")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("Feature and label lengths do not match")
        if len(image_ids) != features.shape[0]:
            raise ValueError("Feature and image ID lengths do not match")
        self.features = features
        self.labels = labels.to(dtype=torch.long)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.image_ids = list(image_ids)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict:
        row_index = int(self.indices[index])
        return {
            "features": self.features[row_index],
            "label": self.labels[row_index],
            "row_index": row_index,
            "image_id": self.image_ids[row_index],
        }
