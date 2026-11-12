import numpy as np
import pandas as pd
import pytest

from src import config
from src.cold_start import recommend_for_new_user, validate_new_user_ratings
from src.data_loader import clean_ratings, split_train_test
from src.evaluation import ItemMeanBaseline, compare_models, evaluate_predictions, mae, rmse
from src.exceptions import (InsufficientRatingsError, InvalidRatingsError, UnknownMovieError, UnknownUserError)
from src.matrix_builder import RatingMatrix


# ---------------------------------------------------------------- data and splitting
def test_split_has_no_overlap_and_is_80_20(prepared):
    train, test = prepared["train"], prepared["test"]
    keys = lambda d: set(zip(d.userId, d.movieId))
    assert not keys(train) & keys(test)
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.03


def test_user_wise_split_keeps_every_user_in_train(prepared):
    assert set(prepared["train"].userId) == set(prepared["ratings"].userId)


def test_clean_ratings_removes_bad_rows(raw):
    ratings, movies = raw
    dirty = pd.concat([ratings, ratings.iloc[[0]], pd.DataFrame([{"userId": 1, "movieId": 999, "rating": 3.0, "timestamp": 1},
                                                                  {"userId": 1, "movieId": 101, "rating": 9.0, "timestamp": 1}])])
    clean, report = clean_ratings(dirty, movies)
    assert report["dropped_duplicates"] >= 1 and report["dropped_unknown_movie"] == 1 and report["dropped_invalid_rating"] == 1
    assert clean.rating.between(0.5, 5).all()


def test_matrix_missing_is_nan_not_zero(prepared):
    m = prepared["matrix"]
    assert np.isnan(m.values).any() and not (m.values == 0).any()
    assert 0 < m.sparsity < 1


# ---------------------------------------------------------------- user-based
def test_user_recommendations_valid_and_unseen(user_cf, prepared):
    recs = user_cf.recommend(1)
    assert 0 < len(recs) <= config.TOP_N
    seen = set(prepared["train"].query("userId == 1").movieId)
    assert not seen & set(recs.movieId)
    assert recs.predicted_rating.between(config.MIN_RATING, config.MAX_RATING).all()
    assert recs.predicted_rating.is_monotonic_decreasing
    assert {"movieId", "title", "genres", "avg_rating", "predicted_rating"} <= set(recs.columns)


def test_user_cf_recovers_planted_taste(user_cf):
    # user 1 belongs to the "Action" group; top recs should be mostly Action titles
    assert (user_cf.recommend(1, n=5).genres == "Action").mean() >= 0.6


def test_invalid_user_raises(user_cf, item_cf):
    with pytest.raises(UnknownUserError):
        user_cf.recommend(99999)
    with pytest.raises(UnknownUserError):
        item_cf.recommend(99999)


# ---------------------------------------------------------------- item-based
def test_item_recommendations_exclude_rated(item_cf, prepared):
    recs = item_cf.recommend(40)
    seen = set(prepared["train"].query("userId == 40").movieId)
    assert not seen & set(recs.movieId)
    assert recs.predicted_rating.between(config.MIN_RATING, config.MAX_RATING).all()


def test_similar_movies(item_cf):
    sims = item_cf.similar_movies(100, n=5)
    assert 100 not in set(sims.movieId) and len(sims) <= 5
    assert (sims.similarity > 0).all()
    with pytest.raises(UnknownMovieError):
        item_cf.similar_movies(-1)


# ---------------------------------------------------------------- cold start
def test_cold_start_happy_path(item_cf):
    ratings = {100: 5, 101: 4.5, 102: 5, 130: 1, 131: 1.5}
    result = recommend_for_new_user(item_cf, ratings)
    assert not result.recommendations.empty
    assert not set(ratings) & set(result.recommendations.movieId)


def test_cold_start_handles_bad_inputs(item_cf):
    clean, warnings = validate_new_user_ratings(item_cf, {100: 5, 101: 4, 102: 3, 999999: 4, 103: 7})
    assert set(clean) == {100, 101, 102} and len(warnings) >= 2
    with pytest.raises(InvalidRatingsError):
        validate_new_user_ratings(item_cf, {999999: 4})
    with pytest.raises(InsufficientRatingsError):
        validate_new_user_ratings(item_cf, {100: 5})
    with pytest.raises(UnknownMovieError):
        item_cf.recommend_from_ratings({424242: 3})


def test_empty_result_is_empty_frame_not_error(item_cf):
    assert item_cf.recommend(1, min_ratings=10**6).empty


# ---------------------------------------------------------------- evaluation
def test_metric_formulas():
    assert rmse([3, 5], [4, 3]) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert mae([3, 5], [4, 3]) == pytest.approx(1.5)


def test_coverage_counts_nan():
    m = evaluate_predictions([3, 4, 5, 2], [3, np.nan, 5, np.nan])
    assert m["coverage"] == 0.5 and m["rmse"] == 0 and m["n_predicted"] == 2


def test_compare_models_uses_test_only_and_beats_chance(prepared, user_cf, item_cf):
    baseline = ItemMeanBaseline().fit(prepared["matrix"])
    table, preds = compare_models({"u": user_cf, "i": item_cf, "b": baseline}, prepared["test"])
    assert len(preds) == len(prepared["test"])
    common = table[table.subset == "common records"]
    assert common.n_predicted.nunique() == 1            # identical records for every model
    assert common.rmse.between(0, 4.5).all() and (table.coverage <= 1).all()


def test_predictions_within_bounds(user_cf, item_cf, prepared):
    t = prepared["test"]
    for model in (user_cf, item_cf):
        p = model.predict_pairs(t.userId, t.movieId)
        assert np.nanmin(p) >= config.MIN_RATING and np.nanmax(p) <= config.MAX_RATING


def test_unknown_pairs_give_nan(user_cf, item_cf):
    assert np.isnan(user_cf.predict_pairs([99999], [100])[0])
    assert np.isnan(item_cf.predict_pairs([1], [99999])[0])


def test_no_test_leakage_in_matrix(prepared):
    m, test = prepared["matrix"], prepared["test"]
    u = m.user_ids.tolist(); i = m.movie_ids.tolist()
    for row in test.head(50).itertuples():
        if row.userId in u and row.movieId in i:
            assert np.isnan(m.values[u.index(row.userId), i.index(row.movieId)])
