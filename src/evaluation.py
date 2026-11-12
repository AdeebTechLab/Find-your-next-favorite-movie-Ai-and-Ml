"""Evaluation: RMSE, MAE and coverage on held-out ratings.

Models are fitted on TRAIN only; here they merely predict the test pairs.
RMSE = sqrt(mean((actual - predicted)^2))     MAE = mean(|actual - predicted|)
Coverage = predicted test pairs / all test pairs (a model that abstains often looks better on RMSE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src import config
from src.matrix_builder import RatingMatrix


def rmse(actual, predicted) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def mae(actual, predicted) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def evaluate_predictions(actual, predicted, n_total: int | None = None) -> dict:
    """Metrics on the predictable subset plus coverage over `n_total` records."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    n_total = len(a) if n_total is None else n_total
    ok = ~np.isnan(p)
    return {
        "rmse": rmse(a[ok], p[ok]),
        "mae": mae(a[ok], p[ok]),
        "coverage": float(ok.sum() / n_total) if n_total else float("nan"),
        "n_predicted": int(ok.sum()),
        "n_test": int(n_total),
    }


class ItemMeanBaseline:
    """Reference model: predict a movie's damped training mean. Any CF model should beat it."""

    name = "Baseline (item mean)"

    def __init__(self, damping: float = config.ITEM_MEAN_DAMPING):
        self.damping = damping

    def fit(self, matrix: RatingMatrix, catalog=None) -> "ItemMeanBaseline":
        observed = ~np.isnan(matrix.values)
        sums = np.nansum(matrix.values, axis=0)
        counts = observed.sum(axis=0)
        global_mean = float(np.nansum(matrix.values) / observed.sum())
        self.means_ = (sums + self.damping * global_mean) / (counts + self.damping)
        self.movie_index_ = pd.Index(matrix.movie_ids)
        return self

    def predict_pairs(self, user_ids, movie_ids) -> np.ndarray:
        pos = self.movie_index_.get_indexer(np.asarray(movie_ids))
        out = np.full(len(pos), np.nan)
        out[pos >= 0] = self.means_[pos[pos >= 0]]
        return out


def compare_models(models: Mapping[str, object], test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on the same test rows.

    Returns (summary_table, per_row_predictions). The summary has two views per model:
      'all predictable' : each model on whatever it can predict (coverage differs)
      'common records'  : only rows EVERY model predicted (the fair head-to-head)
    """
    preds = pd.DataFrame(
        {"userId": test["userId"].to_numpy(), "movieId": test["movieId"].to_numpy(), "actual": test["rating"].to_numpy()}
    )
    for name, model in models.items():
        preds[name] = model.predict_pairs(preds["userId"].to_numpy(), preds["movieId"].to_numpy())

    names = list(models)
    rows = [{"model": n, "subset": "all predictable", **evaluate_predictions(preds["actual"], preds[n])} for n in names]
    common = preds[names].notna().all(axis=1)
    for n in names:
        rows.append(
            {"model": n, "subset": "common records",
             **evaluate_predictions(preds.loc[common, "actual"], preds.loc[common, n], n_total=len(preds))}
        )
    return pd.DataFrame(rows), preds


def plot_comparison(table: pd.DataFrame, path: Path) -> Path:
    """Two small panels (RMSE, MAE) on the common records; lower is better. One hue, direct labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = table[table["subset"] == "common records"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, metric in zip(axes, ["rmse", "mae"]):
        bars = ax.bar(common["model"], common[metric], color="#2f6f9f", width=0.55)
        ax.set_title(f"{metric.upper()} (lower is better)", loc="left", fontsize=11)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, float(common[metric].max()) * 1.2 if len(common) else 1)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    n_common = int(common["n_predicted"].iloc[0]) if len(common) else 0
    fig.suptitle(f"Held-out test ratings, same {n_common} records for every model", x=0.01, ha="left", fontsize=10, color="#555")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
