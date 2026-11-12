# Movie Recommendation System (Collaborative Filtering)

User-based and item-based collaborative filtering on **MovieLens ml-latest-small** (~100k ratings, 610 users, ~9.7k movies), with a Streamlit interface.

## Features
- Top-10 recommendations for any existing user (user-based or item-based) with predicted rating, genres and average rating
- Similar-movie search (item-item similarity)
- Cold-start: a new user rates ~5 movies and gets recommendations without being assigned a userId
- Evaluation: RMSE, MAE and coverage on a held-out 20%, with a popularity-aware item-mean baseline

## Quick start
```bash
pip install -r requirements.txt
python scripts/download_data.py     # fetches ml-latest-small into data/raw/
python scripts/explore_data.py      # optional: EDA charts -> reports/eda/
python scripts/train.py             # split, fit on train only, evaluate, save models + metrics
streamlit run app/streamlit_app.py
pytest                              # tests use synthetic data, no download needed
```
If the download fails (firewall), fetch https://files.grouplens.org/datasets/movielens/ml-latest-small.zip manually and extract it so that `data/raw/ml-latest-small/ratings.csv` exists.

## Design decisions
| Topic | Choice | Why |
|---|---|---|
| Missing ratings | NaN, never 0 | 0 would mean "hated it" |
| Similarity | Cosine on mean-centred ratings | Removes harsh/generous rater bias |
| Small overlaps | Shrinkage `overlap / (overlap + 10)` | Similarity from 2 shared movies is noise |
| Neighbourhood | Top-K positive similarities | Negative neighbours add noise on sparse data |
| Prediction | Mean + weighted deviations | Better than raw weighted ratings |
| Split | User-wise 80/20 on interactions | Every user keeps training data; models never see test rows |
| Unpredictable pairs | NaN, reported as coverage | Honest comparison; also a "common records" head-to-head |
| Thin movies | Need >= 5 training ratings to be recommended | One 5-star vote should not top a list |

## Layout
```
src/            config, data_loader, matrix_builder, user_based_cf, item_based_cf,
                cold_start, evaluation, pipeline, utils, exceptions
app/            streamlit_app.py (UI only, no model logic)
scripts/        download_data.py, explore_data.py, train.py
tests/          pytest suite (synthetic data with planted taste groups)
models/ reports/ data/
```

## Scaling notes
The dense user-user matrix is fine at 610 users. For millions of users, use sparse matrices, approximate nearest neighbours (FAISS/Annoy), precomputed neighbour lists, or matrix factorisation (ALS), and serve from a feature store.

## Limitations
No content or time features, popularity bias, cold-start needs a few ratings, offline RMSE does not equal online satisfaction.
