"""Local patch and low-level residual representation helpers."""

import torch


def pool_patch_features(patches: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate patch tokens to a fixed-size local descriptor."""
    if patches.ndim != 3:
        raise ValueError("Expected patch features with shape [batch, patches, channels]")
    if method == "mean":
        return patches.mean(dim=1)
    if method == "max":
        return patches.max(dim=1).values
    raise ValueError(f"Unsupported patch pooling method: {method}")


def high_pass_residual(images: torch.Tensor) -> torch.Tensor:
    """Return a simple high-pass residual; input is [B,C,H,W]."""
    blurred = torch.nn.functional.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
    return images - blurred
