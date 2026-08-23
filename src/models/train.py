"""End-to-end training pipeline: load cleaned data -> hold out a test set ->
benchmark candidate models via stratified CV (macro-F1, since classes are
imbalanced) -> tune the winner -> evaluate once on the untouched held-out
set -> persist the model + metrics + an experiment-tracking log entry.

Run with: python -m src.models.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.load_data import FEATURE_COLUMNS, TARGET_COLUMN, load_processed
from src.features.feature_engineering import build_preprocessor
from src.models.evaluate import compute_metrics
from src.models.tuning import tune

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
METADATA_PATH = REPO_ROOT / "models" / "metadata.json"
LATEST_METRICS_PATH = REPO_ROOT / "reports" / "metrics" / "latest_metrics.json"
EXPERIMENTS_LOG_PATH = REPO_ROOT / "reports" / "metrics" / "experiments.json"

RANDOM_STATE = 42

# Tie-break preference (lower = preferred) when CV macro-F1 scores are within
# CV_TIE_TOLERANCE of each other: favor more interpretable / less overfit-prone
# models, same policy and rationale as the original single-feature pipeline.
INTERPRETABILITY_RANK = {
    "Logistic Regression": 0,
    "SVM": 1,
    "Random Forest": 2,
    "Gradient Boosting": 3,
    "HistGradientBoosting": 4,
}
CV_TIE_TOLERANCE = 0.01


def build_candidates() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
    }


def select_best(cv_results: dict) -> str:
    best_mean = max(r["macro_f1_mean"] for r in cv_results.values())
    candidates = [
        name for name, r in cv_results.items() if best_mean - r["macro_f1_mean"] <= CV_TIE_TOLERANCE
    ]
    return min(candidates, key=lambda name: INTERPRETABILITY_RANK.get(name, 99))


def main() -> None:
    df = load_processed()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train/val: {len(X_trainval)} rows, held-out test: {len(X_test)} rows "
          "(never used below until final eval)")

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    print("\nCross-validated benchmark (5-fold, macro-F1 is the selection metric — classes are imbalanced):")
    for name, model in build_candidates().items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        f1_scores = cross_val_score(pipe, X_trainval, y_trainval, cv=cv, scoring="f1_macro")
        acc_scores = cross_val_score(pipe, X_trainval, y_trainval, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "macro_f1_mean": float(f1_scores.mean()),
            "macro_f1_std": float(f1_scores.std()),
            "accuracy_mean": float(acc_scores.mean()),
            "accuracy_std": float(acc_scores.std()),
        }
        print(f"  {name}: macro-F1={f1_scores.mean():.1%} (+/-{f1_scores.std():.1%})  "
              f"accuracy={acc_scores.mean():.1%}")

    best_name = select_best(cv_results)
    print(f"\nSelected model: {best_name} (macro-F1={cv_results[best_name]['macro_f1_mean']:.1%}, "
          f"tie-break: interpretability among near-tied top scorers)")

    print(f"\nTuning {best_name} via RandomizedSearchCV (scoring=f1_macro, cv=5)...")
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", build_candidates()[best_name])])
    best_pipeline, best_params = tune(pipe, best_name, X_trainval, y_trainval)
    print(f"Best hyperparameters: {best_params}")

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test) if hasattr(best_pipeline, "predict_proba") else None
    class_labels = list(best_pipeline.classes_)
    test_metrics = compute_metrics(y_test, y_pred, y_proba, labels=class_labels)

    print(f"\nHeld-out test set ({len(y_test)} rows) — never used in model selection or tuning:")
    print(f"  Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.1%}")
    print(f"  Weighted-F1: {test_metrics['weighted_f1']:.1%}")
    if test_metrics.get("roc_auc_ovr_macro") is not None:
        print(f"  ROC-AUC (OVR, macro): {test_metrics['roc_auc_ovr_macro']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_name": best_name,
        "trained_at": timestamp,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,
        "hyperparameters": best_params,
        "n_train_rows": len(X_trainval),
        "n_test_rows": len(X_test),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    LATEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_metrics = {
        "selected_model": best_name,
        "cross_validation_benchmark": cv_results,
        "selected_model_hyperparameters": best_params,
        "held_out_test": test_metrics,
    }
    with open(LATEST_METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)

    experiment_entry = {"timestamp": timestamp, **metadata, "held_out_test": test_metrics,
                         "cross_validation_benchmark": cv_results}
    experiments = []
    if EXPERIMENTS_LOG_PATH.exists():
        with open(EXPERIMENTS_LOG_PATH) as f:
            experiments = json.load(f)
    experiments.append(experiment_entry)
    with open(EXPERIMENTS_LOG_PATH, "w") as f:
        json.dump(experiments, f, indent=2, default=str)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved metrics to {LATEST_METRICS_PATH}")
    print(f"Appended experiment to {EXPERIMENTS_LOG_PATH} ({len(experiments)} total runs)")


if __name__ == "__main__":
    main()
