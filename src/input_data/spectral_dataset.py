"""Deterministic RGB preprocessing and manifest-order spectral data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.evaluation.aiface_phase_e import read_manifest


class SpectralImageTransform:
    """Match baseline DINOv3 fast-processor geometry: square bilinear resize.

    The current Phase E runner overrides processor.size to height/width=224;
    it does not add a crop. EXIF transpose and RGB conversion match its loader.
    Keep RGB in [0, 1] for FFT, without ImageNet channel normalization.
    Tensor resize with antialias=True mirrors the fast processor's resize path.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixels = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        # DINOv3 fast processor explicitly rescales before resizing. Resizing
        # uint8 first would quantize interpolated values and change the spectrum.
        pixels = pixels.float() * (1.0 / 255)
        pixels = TF.resize(
            pixels, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        return pixels


class SpectralDataset(Dataset):
    def __init__(self, manifest: str | Path, image_path_base: str | Path, image_size: int = 224):
        self.frame = read_manifest(manifest)
        self.base = Path(image_path_base).resolve()
        self.transform = SpectralImageTransform(image_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.base / path
        with Image.open(path) as image:
            pixels = self.transform(image)
        return {
            "pixel_values": pixels, "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": str(row["image_id"]), "generator_id": str(row["generator_id"]),
            "split": str(row["split"]),
        }
