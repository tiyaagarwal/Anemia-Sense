"""Loads the trained pipeline once and exposes prediction + explanation as
plain Python functions, shared by both the web routes and the REST API so
there's exactly one code path for "run a prediction"."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.load_data import FEATURE_COLUMNS, load_processed
from src.data.validate_data import PHYSIOLOGICAL_RANGES
from src.explainability.explain import explain_prediction, make_shap_explainer
from src.models.train import MODEL_PATH, METADATA_PATH, RANDOM_STATE

REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelNotTrainedError(RuntimeError):
    pass


class InputValidationError(ValueError):
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__("; ".join(errors))


def _load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise ModelNotTrainedError(
            f"{MODEL_PATH} / {METADATA_PATH} not found. Train the model first with: "
            "python -m src.models.train"
        )
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


_pipeline, _metadata = _load_artifacts()
# Small, fixed background sample for SHAP — computed once at import, not
# per-request, so /predict stays fast.
_processed_df = load_processed()
_background = _processed_df[FEATURE_COLUMNS].sample(min(50, len(_processed_df)), random_state=RANDOM_STATE)
# Building the SHAP explainer is the expensive step — do it once here rather
# than per-request.
_explainer = make_shap_explainer(_pipeline, _background)


def get_feature_columns() -> list:
    return FEATURE_COLUMNS


def get_physiological_ranges() -> dict:
    return PHYSIOLOGICAL_RANGES


def validate_input(raw_values: dict) -> dict:
    """raw_values: {feature_name: raw string or number}. Returns a dict of
    floats on success, raises InputValidationError with all problems found
    (not just the first) otherwise."""
    errors = []
    parsed = {}
    for field in FEATURE_COLUMNS:
        raw = raw_values.get(field)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"{field} is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number, got '{raw}'.")
            continue
        lo, hi = PHYSIOLOGICAL_RANGES[field]
        if not (lo <= value <= hi):
            errors.append(f"{field}={value} is outside the plausible range [{lo}, {hi}].")
            continue
        parsed[field] = value

    if errors:
        raise InputValidationError(errors)
    return parsed


def predict(values: dict, top_n_features: int = 5) -> dict:
    """values: validated {feature_name: float}. Returns prediction, confidence,
    full class probabilities, and the top contributing features."""
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    prediction = _pipeline.predict(input_df)[0]
    class_labels = list(_pipeline.classes_)

    probabilities = None
    confidence = None
    if hasattr(_pipeline, "predict_proba"):
        proba = _pipeline.predict_proba(input_df)[0]
        probabilities = {label: float(p) for label, p in zip(class_labels, proba)}
        confidence = float(max(proba))

    top_features = explain_prediction(_pipeline, input_df, _explainer, top_n=top_n_features)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_features": top_features,
        "model_name": _metadata["model_name"],
    }


def model_metadata() -> dict:
    return _metadata
