"""Trains the anemia classifier and saves model.pkl + scaler.pkl + metrics.json.

Fixes two issues in the original inline training code (see analysis/FINDINGS.md):
  - Deduplicates the dataset before splitting (887/1421 rows were exact
    duplicates, contaminating a plain train_test_split).
  - Reports accuracy via stratified 5-fold cross-validation plus a genuine
    held-out test set, instead of a single train_test_split's number.

Run with: python train.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/anemia.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.json"

# Tie-break preference when models are statistically indistinguishable on
# cross-val accuracy (within CV_TIE_TOLERANCE): prefer the more interpretable,
# less overfit-prone model. A perfect 100% CV score with 0 std on ~500 rows
# (Decision Tree, Gradient Boosting below) is as much a small-data overfitting
# risk signal as it is a sign of a genuinely learnable rule — see FINDINGS.md.
INTERPRETABILITY_RANK = {
    "Logistic Regression": 0,
    "Naive Bayes": 1,
    "SVM": 2,
    "Random Forest": 3,
    "Gradient Boosting": 4,
    "Decision Tree": 5,
}
CV_TIE_TOLERANCE = 0.01


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop(columns=["Name", "Address", "Phone"], errors="ignore")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before} rows, {before - len(df)} exact duplicates removed, {len(df)} remain.")
    return df


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def select_best(cv_results: dict) -> str:
    best_mean = max(r["mean"] for r in cv_results.values())
    candidates = [name for name, r in cv_results.items() if best_mean - r["mean"] <= CV_TIE_TOLERANCE]
    return min(candidates, key=lambda name: INTERPRETABILITY_RANK[name])


def main() -> None:
    df = load_data()
    X = df.drop(columns=["Result"])
    y = df["Result"]
    fields = X.columns.tolist()

    # Held out once, up front, and never touched by cross-validation or
    # model selection below — used only for the final reported test metrics.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    cv_results = {}
    print("\nCross-validated accuracy on the training split (5-fold):")
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_val_score(pipe, X_trainval, y_trainval, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {name}: {scores.mean():.1%} (+/- {scores.std():.1%})")

    best_name = select_best(cv_results)
    print(f"\nSelected model: {best_name} "
          f"(cv={cv_results[best_name]['mean']:.1%}, chosen for interpretability among near-tied top scores)")

    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    best_model = models[best_name]
    best_model.fit(X_trainval_scaled, y_trainval)

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_report = classification_report(y_test, y_pred, output_dict=True)
    test_confusion = confusion_matrix(y_test, y_pred).tolist()
    test_roc_auc = float(roc_auc_score(y_test, y_proba))

    print(f"\nHeld-out test set ({len(y_test)} rows) results for {best_name}:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:", test_confusion)
    print(f"ROC-AUC: {test_roc_auc:.4f}")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "selected_model": best_name,
        "fields": fields,
        "n_rows_after_dedup": len(df),
        "cross_validation": cv_results,
        "held_out_test": {
            "n_rows": len(y_test),
            "accuracy": test_report["accuracy"],
            "precision_macro": test_report["macro avg"]["precision"],
            "recall_macro": test_report["macro avg"]["recall"],
            "f1_macro": test_report["macro avg"]["f1-score"],
            "roc_auc": test_roc_auc,
            "confusion_matrix": test_confusion,
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {MODEL_PATH}, {SCALER_PATH}, {METRICS_PATH}")


if __name__ == "__main__":
    main()
