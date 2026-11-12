"""User-item rating matrix. Rows = users, columns = movies, NaN = "not rated" (NOT zero)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format ratings into a user x movie table (needs unique user/movie pairs)."""
    return ratings.pivot(index="userId", columns="movieId", values="rating")


@dataclass
class RatingMatrix:
    values: np.ndarray      # float32, shape (n_users, n_items), NaN where unrated
    user_ids: np.ndarray    # row labels
    movie_ids: np.ndarray   # column labels

    @classmethod
    def from_ratings(cls, ratings: pd.DataFrame) -> "RatingMatrix":
        if ratings.empty:
            raise ValueError("Cannot build a rating matrix from an empty ratings table")
        table = build_user_item_matrix(ratings)
        return cls(
            values=table.to_numpy(dtype=np.float32),
            user_ids=table.index.to_numpy(),
            movie_ids=table.columns.to_numpy(),
        )

    @property
    def n_users(self) -> int:
        return self.values.shape[0]

    @property
    def n_items(self) -> int:
        return self.values.shape[1]

    @property
    def sparsity(self) -> float:
        """Fraction of cells with no rating."""
        return float(np.isnan(self.values).mean())

    def to_sparse(self) -> sparse.csr_matrix:
        """CSR copy storing only observed ratings (memory-efficient representation)."""
        filled = np.nan_to_num(self.values, nan=0.0)
        return sparse.csr_matrix(filled)
