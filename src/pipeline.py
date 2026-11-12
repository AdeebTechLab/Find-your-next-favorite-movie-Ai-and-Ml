"""End-to-end pipeline: load -> clean -> split -> fit (train only) -> evaluate (test) -> persist."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src import config
from src.data_loader import build_catalog, clean_movies, clean_ratings, load_raw, split_train_test
from src.evaluation import ItemMeanBaseline, compare_models, plot_comparison
from src.exceptions import ModelNotTrainedError
from src.item_based_cf import ItemBasedCF
from src.matrix_builder import RatingMatrix
from src.user_based_cf import UserBasedCF
from src.utils import get_logger

log = get_logger(__name__)


def run_pipeline(
    ratings_path: Path = config.RATINGS_FILE,
    movies_path: Path = config.MOVIES_FILE,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    processed_dir: Path = config.DATA_PROCESSED_DIR,
    test_size: float = config.TEST_SIZE,
    strategy: str = config.SPLIT_STRATEGY,
    seed: int = config.RANDOM_SEED,
    save: bool = True,
) -> dict:
    """Run everything and return the metrics dictionary (also written to reports/metrics.json)."""
    raw_ratings, raw_movies = load_raw(ratings_path, movies_path)
    movies = clean_movies(raw_movies)
    ratings, cleaning_report = clean_ratings(raw_ratings, movies)
    log.info("Cleaning report: %s", cleaning_report)

    train, test = split_train_test(ratings, test_size, strategy, seed)   # SPLIT BEFORE FITTING
    log.info("Train %d rows | Test %d rows", len(train), len(test))

    matrix = RatingMatrix.from_ratings(train)                            # train only
    catalog = build_catalog(movies, train)                               # train-only statistics
    user_cf = UserBasedCF().fit(matrix, catalog)
    item_cf = ItemBasedCF().fit(matrix, catalog)
    baseline = ItemMeanBaseline().fit(matrix)

    table, _ = compare_models({user_cf.name: user_cf, item_cf.name: item_cf, baseline.name: baseline}, test)
    log.info("\n%s", table.round(4).to_string(index=False))

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {**cleaning_report, "n_train": int(len(train)), "n_test": int(len(test)),
                    "matrix_shape": list(matrix.values.shape), "sparsity": round(matrix.sparsity, 5)},
        "config": {"split_strategy": strategy, "test_size": test_size, "seed": seed,
                   "user_k": user_cf.k, "item_k": item_cf.k, "shrinkage": user_cf.shrinkage,
                   "min_neighbors": user_cf.min_neighbors},
        "results": table.to_dict("records"),
    }

    if save:
        models_dir, reports_dir, processed_dir = Path(models_dir), Path(reports_dir), Path(processed_dir)
        for directory in (models_dir, reports_dir, processed_dir):
            directory.mkdir(parents=True, exist_ok=True)
        train.to_csv(processed_dir / "train.csv", index=False)
        test.to_csv(processed_dir / "test.csv", index=False)
        joblib.dump(user_cf, models_dir / "user_cf.joblib", compress=3)
        joblib.dump(item_cf, models_dir / "item_cf.joblib", compress=3)
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        table.to_csv(reports_dir / "model_comparison.csv", index=False)
        plot_comparison(table, reports_dir / "model_comparison.png")
    return metrics


def load_models(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load persisted models; raises ModelNotTrainedError with the fix if absent."""
    models_dir = Path(models_dir)
    paths = {"user_cf": models_dir / "user_cf.joblib", "item_cf": models_dir / "item_cf.joblib"}
    if not all(p.exists() for p in paths.values()):
        raise ModelNotTrainedError("Trained models not found. Run `python scripts/train.py` first.")
    return {name: joblib.load(path) for name, path in paths.items()}


def load_metrics(reports_dir: Path = config.REPORTS_DIR) -> dict | None:
    """Return saved metrics, or None if training has not been run (never invent numbers)."""
    path = Path(reports_dir) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None
