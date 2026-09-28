"""Entry point for the DINOv3 global + local experiment."""

import argparse

from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["data"].get("manifest"):
        parser.error("Set data.manifest in the config after completing the dataset audit.")
    raise NotImplementedError("Wire the local-feature experiment after baseline protocol is fixed.")


if __name__ == "__main__":
    main()
