import numpy as np
import pandas as pd

from app import components as ui


def _frame(**overrides):
    row = {"movieId": 1, "title": "Matrix, The (1999)", "genres": "Action|Sci-Fi", "avg_rating": 4.2,
           "n_ratings": 120, "predicted_rating": 4.4}
    row.update(overrides)
    return pd.DataFrame([row])


def test_split_title_handles_year_and_article():
    assert ui.split_title("Matrix, The (1999)") == ("The Matrix", "1999")
    assert ui.split_title("Untitled") == ("Untitled", "")


def test_card_escapes_html():
    html = ui.cards_html(_frame(title="<script>alert(1)</script> (2000)"), "predicted_rating")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_card_handles_nan_and_empty():
    html = ui.cards_html(_frame(avg_rating=np.nan, predicted_rating=np.nan), "predicted_rating")
    assert "n/a" in html
    assert "No results" in ui.cards_html(pd.DataFrame(), "predicted_rating")


def test_card_html_has_no_newlines_or_indentation():
    html = ui.cards_html(_frame(), "predicted_rating")
    assert "\n" not in html


def test_similarity_and_bar_bounds():
    html = ui.cards_html(_frame(similarity=0.5), "similarity")
    assert "50%" in html
    assert 'width:100%' in ui.cards_html(_frame(predicted_rating=9.0), "predicted_rating")  # bar is clamped


def test_hue_is_stable():
    assert ui.hue("Drama") == ui.hue("Drama") and 0 <= ui.hue("Drama") < 360