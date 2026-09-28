"""Entry point for the Frozen DINOv3 global-feature baseline."""

import argparse

from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["data"].get("manifest"):
        parser.error("Set data.manifest in the config after completing the dataset audit.")
    raise NotImplementedError("Wire the trainer after confirming the audited manifest and model checkpoint.")


if __name__ == "__main__":
    main()
