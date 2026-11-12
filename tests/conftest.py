"""Synthetic fixture with planted structure (two taste groups) so tests are deterministic.

This data exists ONLY to test code paths. It is never used to report model performance.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_catalog, clean_movies, clean_ratings, split_train_test
from src.item_based_cf import ItemBasedCF
from src.matrix_builder import RatingMatrix
from src.user_based_cf import UserBasedCF


@pytest.fixture(scope="session")
def raw():
    rng = np.random.default_rng(0)
    n_users, n_movies = 60, 40
    movies = pd.DataFrame({
        "movieId": np.arange(100, 100 + n_movies),
        "title": [f"Movie {i}" for i in range(n_movies)],
        "genres": ["Action" if i < 20 else "Romance" for i in range(n_movies)],
    })
    rows = []
    for u in range(1, n_users + 1):
        group_action = u <= 30
        for i in rng.choice(n_movies, size=15, replace=False):
            likes = (i < 20) == group_action
            rating = float(np.clip(round((4.5 if likes else 1.5) + rng.normal(0, 0.5)) , 1, 5))
            rows.append((u, 100 + i, rating, 1_000 + len(rows)))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return ratings, movies


@pytest.fixture(scope="session")
def prepared(raw):
    ratings_raw, movies_raw = raw
    movies = clean_movies(movies_raw)
    ratings, _ = clean_ratings(ratings_raw, movies)
    train, test = split_train_test(ratings)
    matrix = RatingMatrix.from_ratings(train)
    catalog = build_catalog(movies, train)
    return {"movies": movies, "ratings": ratings, "train": train, "test": test, "matrix": matrix, "catalog": catalog}


@pytest.fixture(scope="session")
def user_cf(prepared):
    return UserBasedCF(k=20, min_neighbors=2).fit(prepared["matrix"], prepared["catalog"])


@pytest.fixture(scope="session")
def item_cf(prepared):
    return ItemBasedCF(k=20, min_neighbors=2).fit(prepared["matrix"], prepared["catalog"])
