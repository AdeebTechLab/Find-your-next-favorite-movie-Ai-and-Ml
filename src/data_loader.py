"""Load, clean and split MovieLens data.

Order of operations matters for leakage prevention:
    load -> clean -> SPLIT -> (only then) fit models / compute any statistic.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.exceptions import DataLoadError
from src.utils import get_logger

log = get_logger(__name__)


def download_dataset(dest_dir: Path = config.DATA_RAW_DIR, url: str = config.DATASET_URL) -> Path:
    """Download and extract ml-latest-small. Returns the extracted folder."""
    dest_dir = Path(dest_dir)
    target = dest_dir / config.DATASET_NAME
    if (target / "ratings.csv").exists():
        log.info("Dataset already present at %s", target)
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # network errors are varied; wrap them all
        raise DataLoadError(
            f"Could not download {url}: {exc}. Download the zip manually from "
            f"https://grouplens.org/datasets/movielens/latest/ and extract it into {dest_dir}."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest_dir)
    log.info("Dataset extracted to %s", target)
    return target


def load_raw(
    ratings_path: Path = config.RATINGS_FILE, movies_path: Path = config.MOVIES_FILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ratings.csv and movies.csv exactly as shipped."""
    for path in (ratings_path, movies_path):
        if not Path(path).exists():
            raise DataLoadError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(ratings_path), pd.read_csv(movies_path)


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate movieIds and fill missing genres."""
    out = movies.dropna(subset=["movieId", "title"]).copy()
    out["movieId"] = out["movieId"].astype("int64")
    out["genres"] = out["genres"].fillna("(no genres listed)")
    return out.drop_duplicates(subset="movieId", keep="first").reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate ratings. Returns (clean_ratings, report_of_what_was_dropped)."""
    required = ["userId", "movieId", "rating"]
    missing_cols = [c for c in required if c not in ratings.columns]
    if missing_cols:
        raise ValueError(f"ratings is missing columns: {missing_cols}")

    report = {"rows_in": int(len(ratings))}
    df = ratings.copy()

    # 1. missing values
    before = len(df)
    df = df.dropna(subset=required)
    report["dropped_missing"] = int(before - len(df))

    # 2. dtypes
    df["userId"] = df["userId"].astype("int64")
    df["movieId"] = df["movieId"].astype("int64")
    df["rating"] = df["rating"].astype("float64")

    # 3. rating range and 0.5 granularity
    scaled = df["rating"] / config.RATING_STEP
    valid = df["rating"].between(config.MIN_RATING, config.MAX_RATING) & np.isclose(scaled, np.round(scaled))
    report["dropped_invalid_rating"] = int((~valid).sum())
    df = df[valid]

    # 4. duplicate (user, movie) pairs: keep the most recent rating
    before = len(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")
    report["dropped_duplicates"] = int(before - len(df))

    # 5. referential integrity: every rated movie must exist in movies.csv
    before = len(df)
    df = df[df["movieId"].isin(movies["movieId"])]
    report["dropped_unknown_movie"] = int(before - len(df))

    df = df.sort_values(["userId", "movieId"]).reset_index(drop=True)
    report.update(rows_out=int(len(df)), n_users=int(df["userId"].nunique()), n_movies=int(df["movieId"].nunique()))
    return df, report


def split_train_test(
    ratings: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    strategy: str = config.SPLIT_STRATEGY,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split of INTERACTIONS (rows), before any model is fitted.

    random    : every row has the same chance of being held out. Simple, but a user with few
                ratings can end up with no training data (cold user in evaluation).
    user_wise : hold out ~test_size of EACH user's ratings, always leaving >= 1 in train, so
                every user can be personalised and evaluated.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    df = ratings.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(df), dtype=bool)

    if strategy == "random":
        n_test = int(round(len(df) * test_size))
        is_test[rng.permutation(len(df))[:n_test]] = True
    elif strategy == "user_wise":
        for positions in df.groupby("userId").indices.values():
            if len(positions) < 2:
                continue  # a single rating must stay in train
            n_test = min(max(1, int(round(len(positions) * test_size))), len(positions) - 1)
            is_test[rng.choice(positions, size=n_test, replace=False)] = True
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def build_catalog(movies: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Movie catalog indexed by movieId with title, genres and TRAIN-only rating stats.

    Computing avg_rating from the full data would leak test information into what we display.
    """
    stats = train.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
    catalog = movies.set_index("movieId")[["title", "genres"]].join(stats)
    catalog["n_ratings"] = catalog["n_ratings"].fillna(0).astype(int)
    return catalog
