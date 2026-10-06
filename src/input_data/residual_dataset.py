"""Image dataset and preprocessing for Week 3 residual/RGB controls."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


def _resize_shortest_edge_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    """Match the usual processor pattern without applying mean/std normalization."""

    if image_size < 1:
        raise ValueError("image_size must be positive")
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize(
        (max(image_size, round(width * scale)), max(image_size, round(height * scale))),
        resample=Image.Resampling.BICUBIC,
    )
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


def apply_image_corruption(image: Image.Image, corruption: dict[str, Any] | None) -> Image.Image:
    """Apply a corruption to RGB pixels before the shared spatial preprocessing."""

    if not corruption or corruption.get("name", "clean") in (None, "clean"):
        return image
    name = str(corruption["name"]).lower()
    if name == "jpeg":
        quality = int(corruption.get("quality", 70))
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    if name == "resize":
        target = int(corruption.get("target_size", 112))
        if target < 1:
            raise ValueError("Resize target_size must be positive")
        return image.resize((target, target), resample=Image.Resampling.BICUBIC)
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(corruption.get("radius", 1.0))))
    raise ValueError(f"Unsupported corruption: {name}")


class ResidualImageDataset(Dataset):
    """Read a split manifest and return unnormalized RGB tensors in ``[0, 1]``."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        image_column: str = "image_path",
        label_column: str = "label",
        image_path_base: str | Path = ".",
        corruption: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.image_path_base = Path(image_path_base).resolve()
        self.image_size = int(image_size)
        self.image_column = image_column
        self.label_column = label_column
        self.corruption = dict(corruption or {"name": "clean"})
        self.rows = pd.read_csv(self.manifest_path).to_dict(orient="records")
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        required = {image_column, label_column, "image_id"}
        missing = required - set(self.rows[0])
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(str(row[self.image_column]))
        if not image_path.is_absolute():
            image_path = self.image_path_base / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = apply_image_corruption(rgb, self.corruption)
            rgb = _resize_shortest_edge_center_crop(rgb, self.image_size)
            pixels = torch.from_numpy(np.asarray(rgb, dtype="float32"))
            pixel_values = pixels.permute(2, 0, 1).div_(255.0).contiguous()
        source_value = row.get("source", "")
        if pd.isna(source_value):
            source_value = ""
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
            "source": str(source_value or ""),
        }
