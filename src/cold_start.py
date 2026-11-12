"""Cold-start workflow for a brand-new user (no userId is ever invented).

The user rates a few movies; those ratings form a TEMPORARY profile that is scored with the
item-item similarities learned from training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from src import config
from src.exceptions import InsufficientRatingsError, InvalidRatingsError
from src.item_based_cf import ItemBasedCF
from src.utils import build_result_frame, top_n_positions


@dataclass
class ColdStartResult:
    recommendations: pd.DataFrame
    used_ratings: dict[int, float]
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False


def validate_new_user_ratings(
    model: ItemBasedCF,
    ratings: Mapping,
    min_ratings: int = config.COLD_START_MIN_RATINGS,
) -> tuple[dict[int, float], list[str]]:
    """Keep valid {movieId: rating} pairs; report what was dropped and why."""
    warnings: list[str] = []
    clean: dict[int, float] = {}
    for raw_id, raw_rating in ratings.items():
        try:
            movie_id, rating = int(raw_id), float(raw_rating)
        except (TypeError, ValueError):
            warnings.append(f"Ignored unreadable entry: {raw_id!r} -> {raw_rating!r}")
            continue
        scaled = rating / config.RATING_STEP
        if not (config.MIN_RATING <= rating <= config.MAX_RATING) or not np.isclose(scaled, round(scaled)):
            warnings.append(f"Ignored movie {movie_id}: rating {rating} must be {config.MIN_RATING}-{config.MAX_RATING} in steps of {config.RATING_STEP}")
            continue
        if movie_id not in model.movie_index_:
            warnings.append(f"Ignored movie {movie_id}: not in the training catalog")
            continue
        clean[movie_id] = rating  # a repeated movieId keeps the last value

    if not clean:
        raise InvalidRatingsError("None of the supplied ratings were usable. " + " ".join(warnings))
    if len(clean) < min_ratings:
        raise InsufficientRatingsError(
            f"Need at least {min_ratings} valid ratings, got {len(clean)}. " + " ".join(warnings)
        )

    if len(clean) < config.COLD_START_RECOMMENDED:
        warnings.append(f"Only {len(clean)} ratings: recommendations will be less personal than with {config.COLD_START_RECOMMENDED}.")
    if len(set(clean.values())) == 1:
        warnings.append("All ratings are identical, which says little about what you dislike.")
    thin = [m for m in clean if model.n_ratings_[model.movie_index_.get_loc(m)] < config.MIN_RATINGS_TO_RECOMMEND]
    if thin:
        warnings.append(f"{len(thin)} chosen movie(s) have very few ratings in the data, so they carry little signal.")
    return clean, warnings


def popular_movies(model: ItemBasedCF, n: int = config.TOP_N, min_ratings: int = config.MIN_RATINGS_TO_RECOMMEND,
                   exclude: frozenset = frozenset()) -> pd.DataFrame:
    """Non-personalised fallback: damped average rating (popularity-aware)."""
    scores = model.item_means_.astype(float).copy()
    scores[model.n_ratings_ < min_ratings] = np.nan
    for movie_id in exclude:
        pos = model.movie_index_.get_indexer([movie_id])[0]
        if pos >= 0:
            scores[pos] = np.nan
    positions = top_n_positions(scores, n)
    return build_result_frame(model.catalog_, positions, scores[positions])


def recommend_for_new_user(model: ItemBasedCF, ratings: Mapping, n: int = config.TOP_N) -> ColdStartResult:
    """Validate ratings, build a temporary profile and recommend. Falls back to popularity."""
    clean, warnings = validate_new_user_ratings(model, ratings)
    recs = model.recommend_from_ratings(clean, n=n)
    used_fallback = False
    if recs.empty:
        warnings.append("No personalised recommendation was possible; showing popular movies instead.")
        recs = popular_movies(model, n=n, exclude=frozenset(clean))
        used_fallback = True
    return ColdStartResult(recs, clean, warnings, used_fallback)
