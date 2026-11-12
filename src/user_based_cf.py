"""User-based collaborative filtering.

Similarity : cosine on user-mean-centred ratings (missing = 0 AFTER centring = "no opinion"),
             shrunk by overlap/(overlap+lambda) so tiny overlaps are not trusted.
Prediction : r_hat(u,i) = mean_u + sum_v sim(u,v)*(r_vi - mean_v) / sum_v |sim(u,v)|
             over the top-K most similar users who rated i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.exceptions import UnknownUserError
from src.matrix_builder import RatingMatrix
from src.utils import build_result_frame, top_n_positions


class UserBasedCF:
    name = "User-based CF"

    def __init__(
        self,
        k: int = config.USER_NEIGHBORS_K,
        shrinkage: float = config.SIMILARITY_SHRINKAGE,
        min_neighbors: int = config.MIN_NEIGHBORS,
        min_rating: float = config.MIN_RATING,
        max_rating: float = config.MAX_RATING,
    ):
        self.k = k
        self.shrinkage = shrinkage
        self.min_neighbors = min_neighbors
        self.min_rating = min_rating
        self.max_rating = max_rating

    # ------------------------------------------------------------------ training
    def fit(self, matrix: RatingMatrix, catalog: pd.DataFrame) -> "UserBasedCF":
        """Learn similarities from TRAINING ratings only."""
        if matrix.n_users < 2:
            raise ValueError("User-based CF needs at least 2 users")
        ratings = matrix.values.astype(np.float32)
        self.user_index_ = pd.Index(matrix.user_ids)
        self.movie_index_ = pd.Index(matrix.movie_ids)

        self.mask_ = (~np.isnan(ratings)).astype(np.float32)           # 1 where rated
        filled = np.nan_to_num(ratings, nan=0.0)
        counts = self.mask_.sum(axis=1)
        self.user_means_ = (filled.sum(axis=1) / counts).astype(np.float32)
        # centre each user's observed ratings; unobserved stay 0 so they add nothing to dot products
        self.centered_ = (filled - self.user_means_[:, None]) * self.mask_
        self.global_mean_ = float(filled.sum() / self.mask_.sum())
        self.n_ratings_ = self.mask_.sum(axis=0)                        # ratings per movie

        similarity = cosine_similarity(self.centered_).astype(np.float32)
        overlap = self.mask_ @ self.mask_.T                              # co-rated movie counts
        similarity *= overlap / np.maximum(overlap + self.shrinkage, 1e-9)
        np.fill_diagonal(similarity, 0.0)                                # a user is not their own neighbour
        similarity[similarity < 0] = 0.0                                 # keep only positively correlated users

        k = min(self.k, matrix.n_users - 1)
        idx = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(similarity, idx, axis=1)
        order = np.argsort(-sims, axis=1)
        self.neighbor_idx_ = np.take_along_axis(idx, order, axis=1)
        self.neighbor_sim_ = np.take_along_axis(sims, order, axis=1)
        self.catalog_ = catalog.reindex(matrix.movie_ids)
        return self

    # ------------------------------------------------------------------ prediction
    def _predict_row(self, user_pos: int) -> np.ndarray:
        """Predicted rating for every movie for one user (NaN where not predictable)."""
        nb, sims = self.neighbor_idx_[user_pos], self.neighbor_sim_[user_pos]
        numerator = sims @ self.centered_[nb]                            # sum sim * deviation
        denominator = np.abs(sims) @ self.mask_[nb]                      # sum |sim| over neighbours who rated i
        support = (sims > 0).astype(np.float32) @ self.mask_[nb]         # how many neighbours rated i
        ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-9)
        pred = self.user_means_[user_pos] + ratio
        pred[support < self.min_neighbors] = np.nan
        return np.clip(pred, self.min_rating, self.max_rating)

    def predict_pairs(self, user_ids, movie_ids) -> np.ndarray:
        """Predict many (user, movie) pairs; NaN for unknown users/movies or unpredictable pairs."""
        u_pos = self.user_index_.get_indexer(np.asarray(user_ids))
        i_pos = self.movie_index_.get_indexer(np.asarray(movie_ids))
        out = np.full(len(u_pos), np.nan, dtype=np.float64)
        valid = (u_pos >= 0) & (i_pos >= 0)
        for u in np.unique(u_pos[valid]):
            sel = np.flatnonzero(valid & (u_pos == u))
            out[sel] = self._predict_row(u)[i_pos[sel]]
        return out

    # ------------------------------------------------------------------ recommendation
    def _user_position(self, user_id: int) -> int:
        pos = self.user_index_.get_indexer([user_id])[0]
        if pos < 0:
            raise UnknownUserError(f"User {user_id} not found in the training data")
        return int(pos)

    def recommend(
        self, user_id: int, n: int = config.TOP_N, min_ratings: int = config.MIN_RATINGS_TO_RECOMMEND
    ) -> pd.DataFrame:
        """Top-n unseen movies by predicted rating. Empty frame if nothing can be predicted."""
        u = self._user_position(user_id)
        scores = self._predict_row(u)
        scores[self.mask_[u] > 0] = np.nan            # never recommend what the user already rated
        scores[self.n_ratings_ < min_ratings] = np.nan  # too little evidence about the movie
        positions = top_n_positions(scores, n)
        return build_result_frame(self.catalog_, positions, scores[positions])

    def similar_users(self, user_id: int, n: int = 5) -> pd.DataFrame:
        """The n nearest neighbours of a user with their similarity."""
        u = self._user_position(user_id)
        sims = self.neighbor_sim_[u][:n]
        keep = sims > 0
        return pd.DataFrame(
            {"userId": self.user_index_[self.neighbor_idx_[u][:n]][keep], "similarity": np.round(sims[keep], 3)}
        )

    def user_summary(self, user_id: int) -> dict:
        """Profile facts for the UI: number of training ratings, mean rating, favourite genres."""
        u = self._user_position(user_id)
        rated = np.flatnonzero(self.mask_[u] > 0)
        genres = self.catalog_["genres"].iloc[rated].str.split("|").explode()
        genres = genres[genres != "(no genres listed)"]
        return {
            "n_ratings": int(len(rated)),
            "mean_rating": float(self.user_means_[u]),
            "top_genres": list(genres.value_counts().head(3).index),
        }

    def user_history(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """The user's own highest-rated training movies (for context in the UI)."""
        u = self._user_position(user_id)
        rated = np.flatnonzero(self.mask_[u] > 0)
        raw = self.centered_[u, rated] + self.user_means_[u]
        top = rated[np.argsort(-raw, kind="stable")[:n]]
        return build_result_frame(self.catalog_, top, self.centered_[u, top] + self.user_means_[u], "your_rating")