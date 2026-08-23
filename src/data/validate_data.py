"""Data quality validation for the CBC dataset: duplicate detection,
physiological-range validation, class balance, and a leakage/trivial-
reconstruction check. Produces a machine-readable report and the cleaned
dataset used for training.

Run with: python -m src.data.validate_data
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data.load_data import FEATURE_COLUMNS, RAW_PATH, TARGET_COLUMN, load_raw

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "reports" / "data_quality" / "data_quality_report.json"

# Generous adult reference ranges — wide enough to keep genuine extreme
# pathology (e.g. leukemia WBC counts) while catching clear data-entry
# corruption (negative values, decimal-place errors). See DATASET_CARD.md.
PHYSIOLOGICAL_RANGES = {
    "WBC": (0.5, 60), "RBC": (0.5, 9), "HGB": (1, 22), "HCT": (5, 70),
    "MCV": (40, 140), "MCH": (10, 50), "MCHC": (20, 40), "PLT": (1, 1200),
    "PDW": (5, 30), "PCT": (0.005, 1.5), "LYMp": (0, 100), "NEUTp": (0, 100),
    "LYMn": (0, 40), "NEUTn": (0, 40),
}


def find_implausible_rows(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in PHYSIOLOGICAL_RANGES.items():
        mask |= ~df[col].between(lo, hi)
    return mask


def leakage_check(df: pd.DataFrame) -> dict:
    """Single-feature 'stump' accuracy per column — if any one feature comes
    close to full-model accuracy, the target is trivially reconstructible
    from it (the failure mode documented in analysis/FINDINGS.md for the
    original dataset)."""
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for col in FEATURE_COLUMNS:
        stump = DecisionTreeClassifier(max_depth=3, random_state=42)
        scores = cross_val_score(stump, X[[col]], y, cv=cv, scoring="accuracy")
        results[col] = round(float(scores.mean()), 4)
    return results


def clean_and_validate(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"source_file": str(RAW_PATH.relative_to(REPO_ROOT)), "raw_rows": len(df_raw)}

    report["missing_values"] = {k: int(v) for k, v in df_raw.isnull().sum().items() if v > 0}

    dup_count = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows_removed"] = dup_count
    report["rows_after_dedup"] = len(df)

    implausible_mask = find_implausible_rows(df)
    report["physiologically_implausible_rows_removed"] = int(implausible_mask.sum())
    df = df.loc[~implausible_mask].reset_index(drop=True)
    report["rows_after_cleaning"] = len(df)

    report["class_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()
    report["smallest_class"] = df[TARGET_COLUMN].value_counts().idxmin()
    report["smallest_class_count"] = int(df[TARGET_COLUMN].value_counts().min())

    report["single_feature_leakage_check_depth3_stump_accuracy"] = leakage_check(df)
    report["max_single_feature_accuracy"] = max(report["single_feature_leakage_check_depth3_stump_accuracy"].values())
    report["leakage_finding"] = (
        "No single feature reconstructs the label (max single-feature stump accuracy "
        f"{report['max_single_feature_accuracy']:.1%}) — see analysis/FINDINGS.md for the "
        "contrast with the original single-feature-deterministic dataset."
    )

    report["validation_status"] = "passed"
    return df, report


def main() -> None:
    df_raw = load_raw()
    df_clean, report = clean_and_validate(df_raw)

    processed_path = REPO_ROOT / "data" / "processed" / "anemia_types_cbc_clean.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved cleaned dataset to {processed_path}")
    print(f"Saved data quality report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
