"""Train both models on the 80% split, evaluate on the 20% hold-out, persist everything.

Usage: python scripts/train.py [--strategy user_wise|random]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.exceptions import DataLoadError  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=config.SPLIT_STRATEGY, choices=["user_wise", "random"])
    args = parser.parse_args()
    try:
        run_pipeline(strategy=args.strategy)
    except DataLoadError as exc:
        print("ERROR:", exc)
        sys.exit(1)
