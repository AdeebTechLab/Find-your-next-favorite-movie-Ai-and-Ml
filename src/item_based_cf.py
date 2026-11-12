"""Item-based collaborative filtering.

Similarity : adjusted cosine (cosine on user-mean-centred ratings, compared across movies),
             shrunk by overlap. Only the top-K neighbours per movie are stored.
Prediction : r_hat(u,i) = item_mean_i + sum_j sim(i,j)*(r_uj - item_mean_j) / sum_j |sim(i,j)|
             over the K neighbours j of i that user u has rated.
Because the profile is just {movie: rating}, the same scorer serves brand-new users (cold start).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src import config
from src.exceptions import UnknownMovieError, UnknownUserError
from src.matrix_builder import RatingMatrix
from src.utils import build_result_frame, top_n_positions

_CHUNK = 1000  # items processed per block, keeps the dense block small


class ItemBasedCF:
    name = "Item-based CF"

    def __init__(
        self,
        k: int = config.ITEM_NEIGHBORS_K,
        shrinkage: float = config.SIMILARITY_SHRINKAGE,
        item_mean_damping: float = config.ITEM_MEAN_DAMPING,
        min_neighbors: int = config.MIN_NEIGHBORS,
        min_rating: float = config.MIN_RATING,
        max_rating: float = config.MAX_RATING,
    ):
        self.k = k
        self.shrinkage = shrinkage
        self.item_mean_damping = item_mean_damping
        self.min_neighbors = min_neighbors
        self.min_rating = min_rating
        self.max_rating = max_rating

    # ------------------------------------------------------------------ training
    def fit(self, matrix: RatingMatrix, catalog: pd.DataFrame) -> "ItemBasedCF":
        """Learn movie-movie similarities from TRAINING ratings only."""
        if matrix.n_items < 2:
            raise ValueError("Item-based CF needs at least 2 movies")
        ratings = matrix.values.astype(np.float32)
        self.user_index_ = pd.Index(matrix.user_ids)
        self.movie_index_ = pd.Index(matrix.movie_ids)

        self.mask_ = (~np.isnan(ratings)).astype(np.float32)
        self.filled_ = np.nan_to_num(ratings, nan=0.0)
        user_means = self.filled_.sum(axis=1) / self.mask_.sum(axis=1)
        centered = (self.filled_ - user_means[:, None]) * self.mask_

        self.global_mean_ = float(self.filled_.sum() / self.mask_.sum())
        self.n_ratings_ = self.mask_.sum(axis=0)
        # damped item mean: movies with few ratings are pulled toward the global mean
        self.item_means_ = (
            (self.filled_.sum(axis=0) + self.item_mean_damping * self.global_mean_)
            / (self.n_ratings_ + self.item_mean_damping)
        ).astype(np.float32)

        unit_items = normalize(centered.T, axis=1).astype(np.float32)   # (items, users), unit length rows
        mask_items = np.ascontiguousarray(self.mask_.T)                  # (items, users)
        n_items = matrix.n_items
        k = min(self.k, n_items - 1)
        self.neighbor_idx_ = np.empty((n_items, k), dtype=np.int32)
        self.neighbor_sim_ = np.empty((n_items, k), dtype=np.float32)

        for start in range(0, n_items, _CHUNK):
            stop = min(start + _CHUNK, n_items)
            sim = unit_items[start:stop] @ unit_items.T                  # cosine block
            overlap = mask_items[start:stop] @ mask_items.T              # users who rated both movies
            sim *= overlap / np.maximum(overlap + self.shrinkage, 1e-9)
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0   # exclude self
            sim[sim < 0] = 0.0
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            sims = np.take_along_axis(sim, idx, axis=1)
            order = np.argsort(-sims, axis=1)
            self.neighbor_idx_[start:stop] = np.take_along_axis(idx, order, axis=1)
            self.neighbor_sim_[start:stop] = np.take_along_axis(sims, order, axis=1)

        self.catalog_ = catalog.reindex(matrix.movie_ids)
        return self

    # ------------------------------------------------------------------ core scorer
    def _score(self, rating_vec: np.ndarray, rated_mask: np.ndarray) -> np.ndarray:
        """Predicted rating for every movie given one user's (possibly temporary) profile."""
        deviation = (rating_vec - self.item_means_) * rated_mask         # how the user differs from the crowd
        weights = self.neighbor_sim_ * rated_mask[self.neighbor_idx_]     # only neighbours the user has rated
        numerator = (weights * deviation[self.neighbor_idx_]).sum(axis=1)
        denominator = weights.sum(axis=1)                                 # sims are >= 0 so == sum |sim|
        support = (weights > 0).sum(axis=1)
        ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-9)
        pred = self.item_means_ + ratio
        pred[support < self.min_neighbors] = np.nan
        return np.clip(pred, self.min_rating, self.max_rating)

    def predict_pairs(self, user_ids, movie_ids) -> np.ndarray:
        """Predict many (user, movie) pairs; NaN where impossible."""
        u_pos = self.user_index_.get_indexer(np.asarray(user_ids))
        i_pos = self.movie_index_.get_indexer(np.asarray(movie_ids))
        out = np.full(len(u_pos), np.nan, dtype=np.float64)
        valid = (u_pos >= 0) & (i_pos >= 0)
        for u in np.unique(u_pos[valid]):
            sel = np.flatnonzero(valid & (u_pos == u))
            out[sel] = self._score(self.filled_[u], self.mask_[u])[i_pos[sel]]
        return out

    # ------------------------------------------------------------------ recommendation
    def _rank(self, scores: np.ndarray, rated_mask: np.ndarray, n: int, min_ratings: int) -> pd.DataFrame:
        scores = scores.copy()
        scores[rated_mask > 0] = np.nan                  # exclude already-watched movies
        scores[self.n_ratings_ < min_ratings] = np.nan   # exclude movies with thin evidence
        positions = top_n_positions(scores, n)
        return build_result_frame(self.catalog_, positions, scores[positions])

    def recommend(
        self, user_id: int, n: int = config.TOP_N, min_ratings: int = config.MIN_RATINGS_TO_RECOMMEND
    ) -> pd.DataFrame:
        """Top-n unseen movies for an existing user."""
        u = self.user_index_.get_indexer([user_id])[0]
        if u < 0:
            raise UnknownUserError(f"User {user_id} not found in the training data")
        return self._rank(self._score(self.filled_[u], self.mask_[u]), self.mask_[u], n, min_ratings)

    def profile_from_ratings(self, ratings: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """Turn {movieId: rating} into (rating_vector, mask_vector). Raises for unknown movies."""
        positions = self.movie_index_.get_indexer(list(ratings.keys()))
        unknown = [m for m, p in zip(ratings.keys(), positions) if p < 0]
        if unknown:
            raise UnknownMovieError(f"Unknown movie ids: {unknown}")
        vec = np.zeros(len(self.movie_index_), dtype=np.float32)
        mask = np.zeros_like(vec)
        vec[positions] = list(ratings.values())
        mask[positions] = 1.0
        return vec, mask

    def recommend_from_ratings(
        self, ratings: Mapping[int, float], n: int = config.TOP_N, min_ratings: int = config.MIN_RATINGS_TO_RECOMMEND
    ) -> pd.DataFrame:
        """Recommendations for a temporary (cold-start) profile; no userId is created."""
        vec, mask = self.profile_from_ratings(ratings)
        return self._rank(self._score(vec, mask), mask, n, min_ratings)

    def similar_movies(self, movie_id: int, n: int = config.TOP_N) -> pd.DataFrame:
        """The n most similar movies (by rating behaviour). Empty if the movie has no neighbours."""
        pos = self.movie_index_.get_indexer([movie_id])[0]
        if pos < 0:
            raise UnknownMovieError(f"Movie {movie_id} not found in the training catalog")
        sims = self.neighbor_sim_[pos][:n]
        keep = sims > 0
        positions = self.neighbor_idx_[pos][:n][keep]
        frame = build_result_frame(self.catalog_, positions, sims[keep], "similarity")
        frame["similarity"] = np.round(sims[keep], 3)
        return frame
