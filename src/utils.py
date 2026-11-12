"""Small shared helpers: logging, top-N selection and result formatting."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure the root handler only once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def top_n_positions(scores: np.ndarray, n: int) -> np.ndarray:
    """Positions of the n highest non-NaN scores, best first (stable for ties)."""
    valid = np.flatnonzero(~np.isnan(scores))
    if valid.size == 0 or n <= 0:
        return np.array([], dtype=int)
    order = np.argsort(-scores[valid], kind="stable")[:n]
    return valid[order]


RESULT_COLUMNS = ["movieId", "title", "genres", "avg_rating", "n_ratings"]


def build_result_frame(
    catalog: pd.DataFrame,
    positions: np.ndarray,
    scores: np.ndarray,
    score_name: str = "predicted_rating",
) -> pd.DataFrame:
    """Turn catalog positions + scores into the display DataFrame.

    `catalog` must be aligned (row i == matrix column i) and indexed by movieId.
    Returns an empty frame with the right columns when `positions` is empty.
    """
    sub = catalog.iloc[np.asarray(positions, dtype=int)]
    out = pd.DataFrame(
        {
            "movieId": sub.index.to_numpy(),
            "title": sub["title"].to_numpy(),
            "genres": sub["genres"].to_numpy(),
            "avg_rating": sub["avg_rating"].round(2).to_numpy(),
            "n_ratings": sub["n_ratings"].fillna(0).astype(int).to_numpy(),
            score_name: np.round(np.asarray(scores, dtype=float), 2),
        }
    )
    return out.reset_index(drop=True)
