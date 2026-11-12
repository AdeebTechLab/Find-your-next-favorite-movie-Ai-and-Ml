"""Phase 2 exploratory analysis: prints findings and saves charts to reports/eda/.

Usage: python scripts/explore_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import clean_movies, clean_ratings, load_raw  # noqa: E402

OUT = config.REPORTS_DIR / "eda"
COLOR = "#2f6f9f"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ratings_raw, movies_raw = load_raw()
    print("ratings:", ratings_raw.shape, "| movies:", movies_raw.shape)
    print(ratings_raw.dtypes, "\n")
    print("missing values:\n", ratings_raw.isna().sum(), "\n", movies_raw.isna().sum())
    print("duplicate (user,movie) rows:", ratings_raw.duplicated(["userId", "movieId"]).sum())

    movies = clean_movies(movies_raw)
    ratings, report = clean_ratings(ratings_raw, movies)
    print("cleaning report:", report)

    n_users, n_movies = ratings["userId"].nunique(), ratings["movieId"].nunique()
    print(f"sparsity: {1 - len(ratings) / (n_users * n_movies):.4%}")
    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size()
    print("ratings/user  min/median/max:", per_user.min(), per_user.median(), per_user.max())
    print("ratings/movie min/median/max:", per_movie.min(), per_movie.median(), per_movie.max())
    print("movies rated by only one user:", int((per_movie == 1).sum()))

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=COLOR, width=0.7)
    _style(ax, "Rating distribution", "Rating", "Number of ratings")
    fig.tight_layout(); fig.savefig(OUT / "rating_distribution.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, series, label in ((axes[0], per_user, "user"), (axes[1], per_movie, "movie")):
        ax.hist(series, bins=60, color=COLOR)
        ax.set_yscale("log")
        _style(ax, f"Ratings per {label} (long tail)", f"Ratings per {label}", f"Number of {label}s (log)")
    fig.tight_layout(); fig.savefig(OUT / "activity_long_tail.png", dpi=150); plt.close(fig)

    top = (ratings.groupby("movieId")["rating"].agg(["count", "mean"]).query("count >= 50")
           .sort_values("count", ascending=False).head(15).join(movies.set_index("movieId")["title"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["title"][::-1], top["count"][::-1], color=COLOR)
    _style(ax, "15 most-rated movies", "Number of ratings", "")
    fig.tight_layout(); fig.savefig(OUT / "popular_movies.png", dpi=150); plt.close(fig)
    print("saved charts to", OUT)


if __name__ == "__main__":
    main()
