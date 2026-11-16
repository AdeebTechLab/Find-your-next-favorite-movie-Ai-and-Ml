"""Central configuration. Change experiment settings here, not inside the algorithms."""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_NAME = "ml-latest-small"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATINGS_FILE = DATA_RAW_DIR / DATASET_NAME / "ratings.csv"
MOVIES_FILE = DATA_RAW_DIR / DATASET_NAME / "movies.csv"

# ---------------------------------------------------------------- data rules
MIN_RATING = 0.5
MAX_RATING = 5.0
RATING_STEP = 0.5

# ---------------------------------------------------------------- split
RANDOM_SEED = 42
TEST_SIZE = 0.2                 # 80/20 split
SPLIT_STRATEGY = "user_wise"    # "user_wise" (every user keeps train data) or "random"

# ---------------------------------------------------------------- model hyper-parameters
USER_NEIGHBORS_K = 50           # neighbours kept per user
ITEM_NEIGHBORS_K = 50           # neighbours kept per movie
SIMILARITY_SHRINKAGE = 10.0     # sim *= overlap / (overlap + shrinkage): distrusts tiny overlaps
ITEM_MEAN_DAMPING = 5.0         # pulls item means with few ratings toward the global mean
ITEM_PREDICTION_DAMPING = 0.5   # item-based: shrinks weak-evidence predictions toward the movie's mean (0 = off)
MIN_NEIGHBORS = 2               # a prediction needs at least this many contributing neighbours

# ---------------------------------------------------------------- recommendation rules
TOP_N = 10
MIN_RATINGS_TO_RECOMMEND = 5    # movies with fewer training ratings are never recommended

# ---------------------------------------------------------------- cold start
COLD_START_MIN_RATINGS = 3      # hard minimum below which we refuse to personalise
COLD_START_RECOMMENDED = 5      # what the UI asks for