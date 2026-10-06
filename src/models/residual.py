"""Small image models for the Week 3 residual experiments.

The residual filter is deliberately fixed.  Keeping it outside the trainable
CNN makes the Residual-vs-RGB control interpretable and avoids silently
changing the representation during training.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def box_blur_residual(
    images: torch.Tensor, kernel_size: int = 3, residual_gain: float = 1.0
) -> torch.Tensor:
    """Return ``images - reflected_box_blur(images)``.

    ``images`` must be a floating-point ``[B, C, H, W]`` tensor.  Reflect
    padding avoids the artificial border residual caused by zero-padding.
    """

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B,C,H,W], got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("Residual filtering requires floating-point image tensors")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if images.shape[-2] <= kernel_size // 2 or images.shape[-1] <= kernel_size // 2:
        raise ValueError("Image is too small for reflect padding with this kernel")
    if not torch.isfinite(images).all():
        raise ValueError("Images contain non-finite values")

    pad = kernel_size // 2
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    blurred = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
    return (images - blurred) * float(residual_gain)


class TinyResidualCNN(nn.Module):
    """The small, from-scratch CNN specified for the Week 3 baseline."""

    def __init__(self, input_channels: int = 3, num_classes: int = 2, groups: int = 8):
        super().__init__()
        if input_channels < 1 or num_classes < 2:
            raise ValueError("input_channels must be positive and num_classes must be at least 2")
        if any(channels % groups for channels in (32, 64, 128)):
            raise ValueError("groups must divide all CNN channel widths")
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, 128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B,C,H,W], got {tuple(images.shape)}")
        features = self.features(images)
        return self.pool(features).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


class ResidualImageModel(nn.Module):
    """Wrap a fixed residual transform and the TinyCNN classifier.

    Set ``use_residual=False`` for the matched RGB-control.  Both branches
    then share exactly the same trainable architecture.
    """

    def __init__(
        self,
        use_residual: bool = True,
        kernel_size: int = 3,
        residual_gain: float = 1.0,
        input_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.use_residual = bool(use_residual)
        self.kernel_size = int(kernel_size)
        self.residual_gain = float(residual_gain)
        self.encoder = TinyResidualCNN(input_channels=input_channels, num_classes=num_classes)

    def representation(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return box_blur_residual(images, self.kernel_size, self.residual_gain)
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.representation(images))


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters for run metadata."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
