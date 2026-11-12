"""Download MovieLens ml-latest-small into data/raw/. Usage: python scripts/download_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import download_dataset  # noqa: E402
from src.exceptions import DataLoadError  # noqa: E402

if __name__ == "__main__":
    try:
        print("Dataset ready at", download_dataset())
    except DataLoadError as exc:
        print("ERROR:", exc)
        sys.exit(1)
