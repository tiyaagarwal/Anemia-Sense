"""Tests the committed model artifact loads and predicts correctly —
requires `python -m src.models.train` to have been run (CI does this before
tests)."""

import joblib
import pytest

from src.data.load_data import FEATURE_COLUMNS, load_processed
from src.models.train import METADATA_PATH, MODEL_PATH


@pytest.fixture(scope="module")
def pipeline():
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} not found — run `python -m src.models.train` first")
    return joblib.load(MODEL_PATH)


def test_model_metadata_matches_feature_columns():
    import json
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert len(metadata["class_labels"]) == 9


def test_model_predicts_a_valid_class(pipeline):
    df = load_processed()
    row = df[FEATURE_COLUMNS].iloc[[0]]
    prediction = pipeline.predict(row)[0]
    assert prediction in set(df["Diagnosis"].unique())


def test_model_predict_proba_sums_to_one(pipeline):
    df = load_processed()
    row = df[FEATURE_COLUMNS].iloc[[0]]
    proba = pipeline.predict_proba(row)[0]
    assert abs(proba.sum() - 1.0) < 1e-6
    assert (proba >= 0).all()


def test_model_rejects_wrong_column_order_gracefully_via_dataframe():
    """Pipeline should predict correctly regardless of column order in the
    input DataFrame, since it's addressed by name via ColumnTransformer."""
    df = load_processed()
    row = df[FEATURE_COLUMNS].iloc[[0]]
    shuffled = row[list(reversed(FEATURE_COLUMNS))]
    pipeline_obj = joblib.load(MODEL_PATH)
    assert pipeline_obj.predict(row)[0] == pipeline_obj.predict(shuffled)[0]
