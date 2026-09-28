"""DINOv3 feature extractor wrapper."""

import torch
from torch import nn
from transformers import AutoModel


class DINOv3FeatureExtractor(nn.Module):
    """Expose image-level and patch-level DINOv3 representations."""

    def __init__(self, model_name: str, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
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
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        # DINO ViT convention: first token is the CLS/global representation.
        global_feature = tokens[:, 0]
        patch_features = tokens[:, 1:]
        return {"global": global_feature, "patches": patch_features}
