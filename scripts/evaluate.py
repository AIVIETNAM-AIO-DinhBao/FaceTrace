"""Entry point for source-disjoint evaluation."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.parse_args()
    raise NotImplementedError("Implement evaluation once the training checkpoint format is defined.")


if __name__ == "__main__":
    main()
