"""Export the trained recommender as static files for the Vercel site (public/data/).

Vercel serves plain files, so everything the website needs is precomputed here:
  catalog.json    movie table (columnar) incl. damped item means used by the new-user scorer
  neighbors.bin   item-item neighbour lists (uint16 index + uint16 quantised similarity)
  users.json      per-user top-20 lists (user-based and item-based), neighbours and history
  metrics.json    evaluation results from reports/metrics.json

Run after training changes:  python scripts/export_static.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_loader import build_catalog, clean_movies, clean_ratings, load_raw, split_train_test  # noqa: E402
from src.item_based_cf import ItemBasedCF  # noqa: E402
from src.matrix_builder import RatingMatrix  # noqa: E402
from src.user_based_cf import UserBasedCF  # noqa: E402

OUT = ROOT / "public" / "data"
TOP = 20  # recommendations stored per user and method


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_ratings, raw_movies = load_raw(config.RATINGS_FILE, config.MOVIES_FILE)
    movies = clean_movies(raw_movies)
    ratings, _ = clean_ratings(raw_ratings, movies)
    train, _test = split_train_test(ratings, config.TEST_SIZE, config.SPLIT_STRATEGY, config.RANDOM_SEED)
    matrix = RatingMatrix.from_ratings(train)
    catalog = build_catalog(movies, train)
    user_cf = UserBasedCF().fit(matrix, catalog)
    item_cf = ItemBasedCF().fit(matrix, catalog)

    cat = item_cf.catalog_
    pos_of = {int(m): i for i, m in enumerate(item_cf.movie_index_)}
    n_items = len(cat)

    # ---- catalog (columnar, positions are the matrix columns)
    catalog_json = {
        "id": [int(m) for m in cat.index],
        "title": [str(t) for t in cat["title"]],
        "genres": [str(g) for g in cat["genres"]],
        "avg": [round(float(a), 2) if a == a else 0 for a in cat["avg_rating"]],
        "n": [int(x) if x == x else 0 for x in cat["n_ratings"]],
        "mu": [round(float(x), 4) for x in item_cf.item_means_],       # damped item mean
        "nt": [int(x) for x in item_cf.n_ratings_],                     # training ratings per movie
        "k": int(item_cf.neighbor_idx_.shape[1]),
        "minRatings": config.MIN_RATINGS_TO_RECOMMEND,
        "minNeighbors": item_cf.min_neighbors,
        "damping": item_cf.prediction_damping,
        "coldMin": config.COLD_START_MIN_RATINGS,
        "coldRecommended": config.COLD_START_RECOMMENDED,
        "globalMean": round(item_cf.global_mean_, 4),
    }
    (OUT / "catalog.json").write_text(json.dumps(catalog_json, separators=(",", ":")))

    # ---- neighbour lists
    assert n_items < 65536
    sims = item_cf.neighbor_sim_
    assert sims.min() >= 0 and sims.max() <= 1.0
    idx16 = item_cf.neighbor_idx_.astype("<u2")
    sim16 = np.rint(sims * 65535).astype("<u2")
    (OUT / "neighbors.bin").write_bytes(idx16.tobytes() + sim16.tobytes())

    # ---- users
    users = {}
    for uid in (int(u) for u in user_cf.user_index_):
        summary = user_cf.user_summary(uid)
        urec = user_cf.recommend(uid, n=TOP)
        irec = item_cf.recommend(uid, n=TOP)
        hist = user_cf.user_history(uid, n=4)
        sim_users = user_cf.similar_users(uid, n=8)
        hist_col = "your_rating" if "your_rating" in hist.columns else hist.columns[-1]
        users[str(uid)] = {
            "n": summary["n_ratings"],
            "mean": round(summary["mean_rating"], 2),
            "genres": summary["top_genres"],
            "user": [[pos_of[int(m)], float(s)] for m, s in zip(urec.movieId, urec.predicted_rating)],
            "item": [[pos_of[int(m)], float(s)] for m, s in zip(irec.movieId, irec.predicted_rating)],
            "hist": [[pos_of[int(m)], float(s)] for m, s in zip(hist.movieId, hist[hist_col])],
            "peers": [[int(u), float(s)] for u, s in zip(sim_users.userId, sim_users.similarity)],
        }
    (OUT / "users.json").write_text(json.dumps(users, separators=(",", ":")))

    shutil.copy(ROOT / "reports" / "metrics.json", OUT / "metrics.json")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16s}{f.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
