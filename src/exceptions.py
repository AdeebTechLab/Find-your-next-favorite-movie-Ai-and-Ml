"""Domain-specific exceptions so callers (and the UI) can react precisely."""


class RecommenderError(Exception):
    """Base class for all recommender errors."""


class DataLoadError(RecommenderError):
    """Raw data is missing or could not be downloaded/read."""


class UnknownUserError(RecommenderError):
    """The userId does not exist in the training data."""


class UnknownMovieError(RecommenderError):
    """The movieId does not exist in the training catalog."""


class InvalidRatingsError(RecommenderError):
    """None of the supplied ratings are usable (bad ids or bad values)."""


class InsufficientRatingsError(RecommenderError):
    """Too few valid ratings to build a personalised profile."""


class ModelNotTrainedError(RecommenderError):
    """Persisted model files were not found; run scripts/train.py first."""
