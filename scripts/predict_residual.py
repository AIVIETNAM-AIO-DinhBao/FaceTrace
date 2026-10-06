"""Run a saved Week 3 residual/RGB checkpoint on clean or corrupted images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.input_data.residual_dataset import ResidualImageDataset  # noqa: E402
from src.models.residual import ResidualImageModel  # noqa: E402
from src.training.residual_trainer import evaluate_image_model, write_predictions  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_corruption(value: str) -> dict:
    if value == "clean":
        return {"name": "clean"}
    if value == "jpeg70":
        return {"name": "jpeg", "quality": 70}
    if value == "resize112":
        return {"name": "resize", "target_size": 112}
    if value == "blur1":
        return {"name": "blur", "radius": 1.0}
    raise ValueError(f"Unknown corruption: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-path-base", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--corruption", choices=("clean", "jpeg70", "resize112", "blur1"), default="clean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = select_device(args.device)
    model = ResidualImageModel(
        use_residual=bool(checkpoint["use_residual"]),
        kernel_size=int(checkpoint["residual_kernel_size"]),
        residual_gain=float(checkpoint["residual_gain"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    dataset = ResidualImageDataset(
        args.manifest,
        image_size=int(checkpoint["image_size"]),
        image_path_base=args.image_path_base,
        corruption=parse_corruption(args.corruption),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    threshold = float(checkpoint.get("threshold", 0.5))
    metrics, prediction = evaluate_image_model(model, loader, device, threshold)
    output = Path(args.output)
    write_predictions(prediction, output, threshold)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint)),
                "corruption": parse_corruption(args.corruption),
                "metrics": metrics,
                "n_images": len(dataset),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote predictions to {output}; AUROC={metrics['auroc']:.5f}")


if __name__ == "__main__":
    main()
