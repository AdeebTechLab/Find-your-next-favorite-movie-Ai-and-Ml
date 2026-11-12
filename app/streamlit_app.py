# """Streamlit UI. Contains NO recommendation logic: it only calls src.* and renders results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src import config  # noqa: E402
from src.cold_start import recommend_for_new_user  # noqa: E402
from src.exceptions import RecommenderError  # noqa: E402
from src.pipeline import load_metrics, load_models  # noqa: E402

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = ["Home", "Recommend", "Similar Movies", "New User", "Model Performance", "About"]
ICONS = {"Home": "🏠", "Recommend": "🎯", "Similar Movies": "🎞️", "New User": "🆕", "Model Performance": "📊", "About": "ℹ️"}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_models()


@st.cache_data(show_spinner=False)
def movie_labels(_catalog: pd.DataFrame, min_ratings: int) -> dict:
    """{movieId: 'Title'} for movies with enough ratings, most popular first."""
    cat = _catalog[_catalog["n_ratings"] >= min_ratings].sort_values("n_ratings", ascending=False)
    return {int(i): t for i, t in cat["title"].items()}


def goto(page: str):
    st.session_state.page = page


def render_cards(df: pd.DataFrame, score_col: str, numbered: bool = True, compact: bool = False):
    st.markdown(ui.cards_html(df, score_col, numbered, compact), unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, name: str):
    if not df.empty:
        st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), f"{name}.csv", "text/csv")


# ------------------------------------------------------------------ pages
def page_home(models, metrics):
    user_cf = models["user_cf"]
    st.markdown(
        ui.hero(
            "Find your next favourite movie",
            "Personalised recommendations built from the rating behaviour of hundreds of viewers, "
            "using user-based and item-based collaborative filtering.",
            ["MovieLens ml-latest-small", "User-based CF", "Item-based CF", "Cold-start support", "RMSE / MAE evaluated"],
        ),
        unsafe_allow_html=True,
    )
    sparsity = f"{metrics['dataset']['sparsity']:.1%}" if metrics else "n/a"
    n_train = f"{metrics['dataset']['n_train']:,}" if metrics else "n/a"
    st.markdown(
        ui.kpi_row(
            [
                ("Users", f"{len(user_cf.user_index_):,}", "in the training data"),
                ("Movies", f"{len(user_cf.movie_index_):,}", "in the rating matrix"),
                ("Training ratings", n_train, "80% of all ratings"),
                ("Matrix sparsity", sparsity, "cells with no rating"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.subheader("How it works")
    st.markdown(
        ui.steps_html(
            [
                ("Learn from ratings", "Each person's ratings are compared with everyone else's to find similar tastes (or similar movies)."),
                ("Predict what you would rate", "Ratings from your closest neighbours are averaged, weighted by how similar they are to you."),
                ("Rank unseen movies", "Movies you already rated are removed and the highest predicted ones become your top picks."),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Start here")
    c1, c2, c3 = st.columns(3)
    c1.button("🎯  Recommendations for a user", on_click=goto, args=("Recommend",), use_container_width=True)
    c2.button("🎞️  Find similar movies", on_click=goto, args=("Similar Movies",), use_container_width=True)
    c3.button("🆕  I'm a new user", on_click=goto, args=("New User",), use_container_width=True)


def page_recommend(models):
    st.title("Recommendations")
    user_cf, item_cf = models["user_cf"], models["item_cf"]
    c1, c2, c3 = st.columns([1, 1, 2])
    user_id = c1.selectbox("Viewer (userId)", [int(u) for u in user_cf.user_index_])
    n = c2.slider("Number of movies", 5, 20, config.TOP_N)
    method = c3.radio("Method", ["User-based", "Item-based", "Side by side"], horizontal=True)

    try:
        summary = user_cf.user_summary(user_id)
        st.markdown(
            ui.kpi_row(
                [
                    ("Movies rated", f"{summary['n_ratings']:,}", "in the training data"),
                    ("Average rating given", f"{summary['mean_rating']:.2f}", "out of 5"),
                    ("Favourite genres", ", ".join(summary["top_genres"]) or "n/a", "by number of ratings"),
                ]
            ),
            unsafe_allow_html=True,
        )
        with st.spinner("Computing recommendations..."):
            user_recs = user_cf.recommend(user_id, n=n)
            item_recs = item_cf.recommend(user_id, n=n)
    except RecommenderError as exc:
        st.error(str(exc))
        return

    if method == "User-based":
        st.subheader(f"Top {n} for user {user_id} · users with similar taste")
        render_cards(user_recs, "predicted_rating")
        csv_button(user_recs, f"user{user_id}_user_based")
    elif method == "Item-based":
        st.subheader(f"Top {n} for user {user_id} · similar to movies they rated")
        render_cards(item_recs, "predicted_rating")
        csv_button(item_recs, f"user{user_id}_item_based")
    else:
        overlap = set(user_recs.movieId) & set(item_recs.movieId)
        st.markdown(ui.callout(f"{len(overlap)} of {n} movies appear in both lists."), unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.subheader("User-based")
            render_cards(user_recs, "predicted_rating", compact=True)
        with right:
            st.subheader("Item-based")
            render_cards(item_recs, "predicted_rating", compact=True)

    with st.expander("Why these? Taste neighbours and this viewer's favourites"):
        a, b = st.columns([1, 2])
        with a:
            st.markdown("**Most similar viewers**")
            st.dataframe(user_cf.similar_users(user_id, n=8), hide_index=True)
        with b:
            st.markdown("**Highest rated by this viewer**")
            render_cards(user_cf.user_history(user_id, n=4), "your_rating", numbered=False, compact=True)


def page_similar(models):
    st.title("Similar movies")
    st.caption("Similarity comes from who rated the movies alike, not from genre labels.")
    item_cf = models["item_cf"]
    labels = movie_labels(item_cf.catalog_, config.MIN_RATINGS_TO_RECOMMEND)
    c1, c2 = st.columns([3, 1])
    movie_id = c1.selectbox("Pick or type a movie", list(labels), format_func=labels.get)
    n = c2.slider("Results", 5, 20, 10)
    try:
        with st.spinner("Finding similar movies..."):
            sims = item_cf.similar_movies(movie_id, n=n)
    except RecommenderError as exc:
        st.error(str(exc))
        return
    if sims.empty:
        st.warning("This movie has no similar movies with enough shared raters.")
        return
    st.subheader(f"If you liked **{labels[movie_id]}**")
    render_cards(sims, "similarity")
    csv_button(sims, f"similar_to_{movie_id}")


def page_new_user(models):
    st.title("New here? Rate a few movies")
    st.caption(
        f"Choose up to {config.COLD_START_RECOMMENDED} movies you know and rate them. "
        f"At least {config.COLD_START_MIN_RATINGS} are required. Nothing is saved and no account is created."
    )
    item_cf = models["item_cf"]
    labels = movie_labels(item_cf.catalog_, 50)  # well-known movies are easier to rate
    chosen = st.multiselect("Search and choose movies", list(labels), format_func=labels.get, max_selections=config.COLD_START_RECOMMENDED)

    ratings = {}
    if chosen:
        st.markdown("**Your ratings**")
        cols = st.columns(min(len(chosen), 3))
        for i, movie_id in enumerate(chosen):
            ratings[movie_id] = cols[i % len(cols)].slider(labels[movie_id], 0.5, 5.0, 4.0, 0.5, key=f"nu_{movie_id}")

    if st.button("Get my recommendations", type="primary", disabled=not chosen):
        try:
            with st.spinner("Building your taste profile..."):
                result = recommend_for_new_user(item_cf, ratings)
        except RecommenderError as exc:
            st.error(str(exc))
            return
        for warning in result.warnings:
            st.warning(warning)
        if result.used_fallback:
            st.info("Showing popular movies because a personalised list was not possible.")
        st.subheader("Picked for you")
        render_cards(result.recommendations, "predicted_rating")
        csv_button(result.recommendations, "new_user_recommendations")


def page_performance(metrics):
    st.title("Model performance")
    if not metrics:
        st.warning("No evaluation results found. Run `python scripts/train.py` first.")
        return
    cfg, ds = metrics["config"], metrics["dataset"]
    st.caption(
        f"{cfg['split_strategy'].replace('_', '-')} split · {ds['n_train']:,} training / {ds['n_test']:,} test ratings · "
        f"seed {cfg['seed']} · models fitted on training data only"
    )
    table = pd.DataFrame(metrics["results"])
    common = table[table["subset"] == "common records"].reset_index(drop=True)

    best = common.loc[common["rmse"].idxmin()]
    cf_only = common[~common["model"].str.contains("Baseline")]
    best_cf = cf_only.loc[cf_only["rmse"].idxmin()]
    st.markdown(
        ui.kpi_row(
            [(r["model"], f"{r['rmse']:.3f}", f"RMSE · MAE {r['mae']:.3f} · coverage {table[(table.model == r['model']) & (table.subset == 'all predictable')]['coverage'].iloc[0]:.1%}")
             for _, r in common.iterrows()]
        ),
        unsafe_allow_html=True,
    )
    if "Baseline" in best["model"]:
        st.markdown(ui.callout(f"The item-mean baseline has the lowest RMSE here. The best collaborative model is {best_cf['model']} "
                               f"({best_cf['rmse']:.3f}). Baselines are strong on sparse data, so this comparison matters."), unsafe_allow_html=True)
    else:
        st.markdown(ui.callout(f"Lowest RMSE: {best['model']} ({best['rmse']:.3f}), measured on the same {int(best['n_predicted']):,} test ratings for every model."), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**RMSE (lower is better)**")
        st.bar_chart(common.set_index("model")["rmse"])
    with right:
        st.markdown("**MAE (lower is better)**")
        st.bar_chart(common.set_index("model")["mae"])

    with st.expander("Full results table"):
        st.dataframe(table.round(4), hide_index=True)
    with st.expander("How to read these metrics"):
        st.markdown(
            "- **RMSE** = sqrt(mean((actual - predicted)²)). Penalises large mistakes more.\n"
            "- **MAE** = mean(|actual - predicted|). The average miss in stars.\n"
            "- **Coverage** = share of test ratings the model could predict. A model that abstains often can look more accurate.\n"
            "- **Common records** = only test ratings every model predicted, which is the fair head-to-head."
        )


def page_about():
    st.title("About this project")
    st.markdown(
        "**CineMatch** is a collaborative-filtering recommender built on the MovieLens *ml-latest-small* dataset.\n\n"
        "**Pipeline**\n"
        "1. Clean ratings (types, ranges, duplicates, unknown movies).\n"
        "2. Split 80/20 per user, *before* any model sees the data.\n"
        "3. Build the user-item matrix (missing = unrated, never zero).\n"
        "4. Fit user-based and item-based models on the training part only.\n"
        "5. Evaluate on the held-out ratings with RMSE, MAE and coverage.\n\n"
        "**Method**: cosine similarity on mean-centred ratings, shrunk by overlap size; prediction = mean + similarity-weighted deviations from the top-K neighbours.\n\n"
        "**Known limitations**: no genre/content features, no time awareness, popularity bias, and new users need a few ratings. "
        "Offline RMSE does not guarantee viewer satisfaction."
    )
    st.caption("Stack: Python · pandas · NumPy · scikit-learn · Streamlit")


# ------------------------------------------------------------------ app shell
def main():
    st.session_state.setdefault("page", "Home")
    with st.sidebar:
        st.markdown('<div class="sidebrand">🎬 CineMatch</div><div class="sidesub">Collaborative filtering recommender</div>', unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="page", format_func=lambda p: f"{ICONS[p]}  {p}", label_visibility="collapsed")
        st.divider()
        st.caption("Data: MovieLens ml-latest-small (GroupLens)")

    page = st.session_state.page
    if page == "About":
        return page_about()
    try:
        models = get_models()
    except RecommenderError as exc:
        st.error(str(exc))
        st.stop()
    metrics = load_metrics()
    if page == "Home":
        page_home(models, metrics)
    elif page == "Recommend":
        page_recommend(models)
    elif page == "Similar Movies":
        page_similar(models)
    elif page == "New User":
        page_new_user(models)
    elif page == "Model Performance":
        page_performance(metrics)


main()