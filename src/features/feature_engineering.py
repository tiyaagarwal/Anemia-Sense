"""Reproducible preprocessing pipeline. All 14 CBC features are numeric with
no missing values in the cleaned dataset, so preprocessing is a single
scaling step — kept as a ColumnTransformer (rather than a bare
StandardScaler) so adding a categorical or engineered feature later doesn't
require restructuring the pipeline, and so it composes directly with any
sklearn estimator via a single Pipeline that gets fit only on the training
split (never on validation/test data)."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, FEATURE_COLUMNS),
    ])
