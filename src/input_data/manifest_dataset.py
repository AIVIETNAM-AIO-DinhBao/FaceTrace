"""CSV manifest dataset using the checkpoint's official image processor."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        processor,
        image_column: str = "image_path",
        label_column: str = "label",
        image_path_base: str | Path = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.processor = processor
        self.image_path_base = Path(image_path_base).resolve()
        self.rows = pd.read_csv(self.manifest_path).to_dict(orient="records")
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        for name in (image_column, label_column, "image_id"):
            if name not in self.rows[0]:
                raise ValueError(f"Manifest {self.manifest_path} is missing column {name!r}")
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(str(row[self.image_column]))
        if not image_path.is_absolute():
            image_path = self.image_path_base / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
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
