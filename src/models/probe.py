"""Frozen-backbone global-feature probe."""

from torch import nn


class GlobalProbe(nn.Module):
    def __init__(self, extractor: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.classifier = classifier

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.extractor, "freeze", False):
            self.extractor.eval()
        return self

    def forward(self, pixel_values):
        features = self.extractor(pixel_values)["global"]
        return self.classifier(features)
