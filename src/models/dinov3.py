"""Frozen DINOv3 feature extractor."""

import torch
from torch import nn
from transformers import AutoModel


class DINOv3FeatureExtractor(nn.Module):
    """Expose CLS/global and patch representations from a pretrained DINOv3 ViT."""

    def __init__(
        self, model_name: str, freeze: bool = True, revision: str | None = None
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision)
        self.freeze = freeze
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        # DINOv3 may place register tokens after CLS and before spatial patch tokens.
        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        patch_start = 1 + num_register_tokens
        return {"global": tokens[:, 0], "patches": tokens[:, patch_start:]}
