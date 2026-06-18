# pipeline_factory.py — Day 15
# Builds the reusable preprocessing + model Pipeline.
# Imported by 02_pipeline.py and 03_grid_search.py.

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_IMPUTE_STRATEGY,
    CAT_IMPUTE_STRATEGY,
    OHE_HANDLE_UNKNOWN,
)


def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for Ames Housing features."""

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=CAT_IMPUTE_STRATEGY)),
            (
                "ohe",
                OneHotEncoder(handle_unknown=OHE_HANDLE_UNKNOWN, sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drops Order, PID, SalePrice if accidentally passed
    )

    return preprocessor


def build_pipeline(model) -> Pipeline:
    """Wrap preprocessor + any sklearn-compatible estimator into a Pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
