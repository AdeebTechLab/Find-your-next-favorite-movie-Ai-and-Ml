# Intelligent Movie Recommendation System

User-based and item-based collaborative filtering on **MovieLens ml-latest-small** (~100k ratings, 610 users, ~9.7k movies), served as a fast static website on **Vercel**.

## Features
- Top-10 recommendations for any existing user (user-based, item-based, or side by side) with predicted rating, genres and average rating
- Similar-movie search (item-item similarity)
- Cold-start: a new user rates 3-5 movies and gets recommendations in the browser, with no account and nothing sent anywhere
- Evaluation: RMSE, MAE and coverage on a held-out 20%, with a popularity-aware item-mean baseline

## Deploy on Vercel
The site is plain HTML/CSS/JS in `public/`, with the trained model exported to `public/data/`. There is no server and no build step.

1. Push this folder to GitHub (or run `vercel` from it with the Vercel CLI).
2. In Vercel choose **Add New > Project**, import the repo and click **Deploy**. `vercel.json` already sets the output directory to `public` and needs no framework preset, build command or environment variables.

Preview locally: `cd public && python -m http.server 8000`, then open http://localhost:8000.

## Retrain and refresh the website
```bash
pip install -r requirements.txt
python scripts/download_data.py     # fetches ml-latest-small into data/raw/
python scripts/explore_data.py      # optional: EDA charts -> reports/eda/
python scripts/train.py             # split, fit on train only, evaluate, save models + metrics
python scripts/export_static.py     # writes public/data/* used by the website
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
public/         the Vercel site: index.html, assets/ (app.js, engine.js, style.css, logo), data/ (exported model)
scripts/        download_data.py, explore_data.py, train.py, export_static.py
vercel.json     Vercel settings (serve public/, cache headers)
tests/          pytest suite (synthetic data with planted taste groups)
models/ reports/ data/
```

## Scaling notes
The dense user-user matrix is fine at 610 users. For millions of users, use sparse matrices, approximate nearest neighbours (FAISS/Annoy), precomputed neighbour lists, or matrix factorisation (ALS), and serve from a feature store.

## Limitations
No content or time features, popularity bias, cold-start needs a few ratings, offline RMSE does not equal online satisfaction.
