"""Predict a fixed manifest from a saved spectral checkpoint; no fitting."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.spectral_trainer import SpectralPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-path-base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    predictor = SpectralPredictor(args.checkpoint, args.device)
    predictor.predict_manifest(args.manifest, args.image_path_base, args.output_dir, args.split, args.num_workers)
    print(f"Predictions: {Path(args.output_dir) / ('predictions_' + args.split + '.csv')}")


if __name__ == "__main__":
    main()
