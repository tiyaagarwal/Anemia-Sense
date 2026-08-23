"""Model explainability:
- Global feature importance via permutation importance (model-agnostic,
  computed on the held-out test set so it reflects real generalization
  behavior, not training-set memorization).
- Local per-prediction explanation via SHAP, for the "why did the model say
  this" display in the web app and API.

Run with: python -m src.explainability.explain (regenerates the global report)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data.load_data import FEATURE_COLUMNS, TARGET_COLUMN, load_processed
from src.models.train import MODEL_PATH, RANDOM_STATE

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_IMPORTANCE_PATH = REPO_ROOT / "reports" / "figures" / "global_feature_importance.json"


def compute_global_importance(pipeline, X_test, y_test, n_repeats: int = 20) -> dict:
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="f1_macro"
    )
    importances = {
        col: {"mean": float(result.importances_mean[i]), "std": float(result.importances_std[i])}
        for i, col in enumerate(FEATURE_COLUMNS)
    }
    return dict(sorted(importances.items(), key=lambda kv: kv[1]["mean"], reverse=True))


def make_shap_explainer(pipeline, background: pd.DataFrame):
    """Builds a SHAP explainer once (this is the expensive step — callers
    should build it a single time at app startup and reuse it, not rebuild
    it per request/prediction)."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    background_transformed = preprocessor.transform(background)
    # shap.Explainer auto-selects TreeExplainer/LinearExplainer/KernelExplainer
    # based on the model type, so this works regardless of which candidate
    # model train.py ends up selecting.
    return shap.Explainer(model.predict_proba, background_transformed, feature_names=FEATURE_COLUMNS)


def explain_prediction(pipeline, input_row: pd.DataFrame, explainer, top_n: int = 5) -> list:
    """Returns the top_n features driving the prediction for a single input
    row, for the class the model actually predicted. `explainer` should come
    from make_shap_explainer(), built once and reused across calls."""
    predicted_class = pipeline.predict(input_row)[0]
    class_idx = list(pipeline.classes_).index(predicted_class)

    transformed_row = pipeline.named_steps["preprocess"].transform(input_row)
    shap_values = explainer(transformed_row)

    values = shap_values.values[0, :, class_idx] if shap_values.values.ndim == 3 else shap_values.values[0]
    contributions = sorted(
        zip(FEATURE_COLUMNS, values, input_row.iloc[0][FEATURE_COLUMNS].tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:top_n]
    return [
        {
            "feature": name,
            "value": float(raw_value),
            "shap_value": float(shap_val),
            "direction": "increases" if shap_val > 0 else "decreases",
        }
        for name, shap_val, raw_value in contributions
    ]


def main() -> None:
    df = load_processed()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipeline = joblib.load(MODEL_PATH)
    importance = compute_global_importance(pipeline, X_test, y_test)

    GLOBAL_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_IMPORTANCE_PATH, "w") as f:
        json.dump(importance, f, indent=2)

    print("Global permutation importance (drop in macro-F1 when a feature is shuffled):")
    for feature, stats in importance.items():
        print(f"  {feature}: {stats['mean']:+.4f} (+/-{stats['std']:.4f})")
    print(f"\nSaved to {GLOBAL_IMPORTANCE_PATH}")

    # Sanity-check the local explanation path with one real test-set row.
    example = X_test.iloc[[0]]
    background = X.sample(min(50, len(X)), random_state=RANDOM_STATE)
    explainer = make_shap_explainer(pipeline, background)
    local = explain_prediction(pipeline, example, explainer)
    print(f"\nExample local explanation for one held-out row (predicted: {pipeline.predict(example)[0]}):")
    for item in local:
        print(f"  {item['feature']}={item['value']}: {item['direction']} likelihood (shap={item['shap_value']:+.4f})")


if __name__ == "__main__":
    main()
