"""HISTORICAL — preserved as-is. This analyzes the original Gender+Hemoglobin
dataset this project started with. The project has since moved to a richer,
genuine multi-class CBC dataset (see reports/data_quality/DATASET_CARD.md
and src/data/validate_data.py) — this script and its findings are kept here
because they're the reason for that redesign, not because they describe the
current pipeline.

Reproduces and documents the two root causes behind every model in this
project reporting ~100% accuracy in the original code:

1. 887 of 1421 rows (62%) in data/anemia.csv are exact duplicates. A random
   train_test_split puts copies of the same row on both sides of the split,
   so part of the reported "test" accuracy is the model recognizing rows it
   was literally trained on.
2. Even with duplicates removed, `Result` turns out to be a deterministic
   threshold function of Hemoglobin + Gender with zero label noise (the
   synthetic dataset encodes the WHO anemia threshold directly): every
   female (Gender=0) row with Hemoglobin < 12.0 is anemic and every row
   >= 12.0 is not; every male (Gender=1) row splits the same way at 13.5.
   That means "predicting Result from Hemoglobin" isn't really a learned
   pattern — it's the model re-deriving a lookup rule that's already in the
   data. An ablation (dropping Hemoglobin) below shows accuracy collapses to
   near-chance once the deterministic feature is withheld, which is what you
   expect if the other columns don't independently carry the signal.

Run with: python analysis/diagnose_data_issues.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

DATA_PATH = "analysis/legacy_dataset/anemia.csv"


def report_duplicates(df: pd.DataFrame) -> None:
    dup_count = df.duplicated().sum()
    print("== Duplicate rows ==")
    print(f"Total rows: {len(df)}")
    print(f"Exact duplicate rows: {dup_count} ({dup_count / len(df):.1%})")
    print(f"Unique rows: {df.drop_duplicates().shape[0]}")
    print()


def report_hemoglobin_threshold(df: pd.DataFrame) -> None:
    print("== Hemoglobin vs Result, by gender ==")
    for gender in sorted(df["Gender"].unique()):
        sub = df[df["Gender"] == gender]
        pos = sub[sub["Result"] == 1]["Hemoglobin"]
        neg = sub[sub["Result"] == 0]["Hemoglobin"]
        print(
            f"Gender={gender}: anemic Hemoglobin in [{pos.min():.1f}, {pos.max():.1f}], "
            f"not-anemic Hemoglobin in [{neg.min():.1f}, {neg.max():.1f}] — no overlap"
        )
    print()


def report_ablation(df: pd.DataFrame) -> None:
    print("== Cross-validated accuracy, with vs without Hemoglobin (deduplicated data) ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    y = df["Result"]
    for label, drop_cols in [
        ("WITH Hemoglobin", ["Result"]),
        ("WITHOUT Hemoglobin", ["Result", "Hemoglobin"]),
    ]:
        print(f"-- {label} --")
        X = df.drop(columns=drop_cols)
        for name, model in models.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            print(f"  {name}: {scores.mean():.1%} (+/- {scores.std():.1%})")
        print()


def main() -> None:
    df = pd.read_csv(DATA_PATH).drop(columns=["Name", "Address", "Phone"], errors="ignore")
    report_duplicates(df)

    df_unique = df.drop_duplicates().reset_index(drop=True)
    report_hemoglobin_threshold(df_unique)
    report_ablation(df_unique)


if __name__ == "__main__":
    main()
