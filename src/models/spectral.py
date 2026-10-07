"""Fixed RGB radial FFT features and a small trainable MLP probe."""

from __future__ import annotations

import torch
from torch import nn


class RadialFFT(nn.Module):
    """Symmetric Hann -> FFT2 -> fftshift -> log1p(abs) -> annular mean.

    Equal-width bins cover the entire square spectrum, including its corners.
    Channels remain separate; the output is RGB channel-major [B, 3 * bins].
    FFT and windowing always operate in float32 (including on CUDA).
    """

    def __init__(self, image_size: int = 224, radial_bins: int = 64):
        super().__init__()
        if image_size < 4 or radial_bins < 1:
            raise ValueError("image_size >= 4 and radial_bins >= 1 are required")
        self.image_size = image_size
        self.radial_bins = radial_bins
        window = torch.hann_window(image_size, periodic=False)
        self.register_buffer("window", torch.outer(window, window))
        # Construct bin ownership in float64 so pixels on annular boundaries
        # do not move to a neighbouring bin through float32 rounding.
        axis = torch.arange(image_size, dtype=torch.float64) - image_size // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        bins = (radius / radius.max() * radial_bins).long().clamp(max=radial_bins - 1)
        counts = torch.bincount(bins.flatten(), minlength=radial_bins).float()
        if (counts == 0).any():
            raise ValueError("Too many radial bins: at least one annulus is empty")
        self.register_buffer("bin_index", bins.flatten())
        self.register_buffer("bin_counts", counts)
        self.register_buffer("radial_edges", torch.linspace(0, radius.max(), radial_bins + 1))

    @property
    def output_dim(self) -> int:
        return 3 * self.radial_bins

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected [B, 3, {self.image_size}, {self.image_size}]")
        if not torch.isfinite(images).all():
            raise ValueError("Images contain NaN/Inf")
        with torch.autocast(device_type=images.device.type, enabled=False):
            spectrum = torch.fft.fft2(images.float() * self.window, norm="backward")
            log_magnitude = torch.log1p(torch.fft.fftshift(spectrum, dim=(-2, -1)).abs())
            flat = log_magnitude.flatten(2)
            indices = self.bin_index.view(1, 1, -1).expand_as(flat)
            sums = flat.new_zeros(images.shape[0], 3, self.radial_bins)
            sums.scatter_add_(2, indices, flat)
            return (sums / self.bin_counts).flatten(1)


class SpectralMLP(nn.Module):
    """Train-only feature standardization followed by Linear/GELU/Linear."""

    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid MLP dimensions/dropout")
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )

    @torch.no_grad()
    def fit_normalization(self, train_features: torch.Tensor) -> None:
        if bool(self.normalization_fitted):
            raise RuntimeError("Normalization has already been fitted; create a new model to refit")
        if train_features.ndim != 2 or len(train_features) == 0 or not torch.isfinite(train_features).all():
            raise ValueError("Expected non-empty finite training features")
        self.feature_mean.copy_(train_features.mean(0))
        self.feature_scale.copy_(train_features.std(0, unbiased=False).clamp_min(1e-6))
        self.normalization_fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers((features - self.feature_mean) / self.feature_scale)

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(features).softmax(1)[:, 1]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
