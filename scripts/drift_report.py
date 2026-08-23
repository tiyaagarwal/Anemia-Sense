"""Offline data-drift monitoring demonstration.

This does NOT stream real production traffic — there is none. Instead it
compares real feature distributions between two genuine splits of the
dataset (the training split vs. the held-out test split) using a two-sample
Kolmogorov-Smirnov test per feature. Since both splits are stratified
samples from the same underlying data, the expected, correct outcome is "no
significant drift" — this is a sanity check on the drift-detection mechanism
itself, not a claim about live production data.

Run with: python -m scripts.drift_report
Or compare two specific diagnosis classes to see the mechanism correctly
flag a real distributional difference:
  python -m scripts.drift_report --classes "Healthy" "Iron deficiency anemia"
"""

import argparse
import json
from pathlib import Path

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from src.data.load_data import FEATURE_COLUMNS, TARGET_COLUMN, load_processed
from src.models.train import RANDOM_STATE

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports" / "figures" / "drift_report.json"
DRIFT_P_THRESHOLD = 0.05


def compute_drift_report(reference_df, current_df) -> dict:
    results = {}
    for feature in FEATURE_COLUMNS:
        stat, p_value = ks_2samp(reference_df[feature], current_df[feature])
        results[feature] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < DRIFT_P_THRESHOLD),
        }
    n_drifted = sum(1 for r in results.values() if r["drift_detected"])
    return {
        "reference_n": len(reference_df),
        "current_n": len(current_df),
        "test": "two-sample Kolmogorov-Smirnov, per feature",
        "p_value_threshold": DRIFT_P_THRESHOLD,
        "features_with_drift": n_drifted,
        "total_features": len(FEATURE_COLUMNS),
        "per_feature": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes", nargs=2, metavar=("REFERENCE_CLASS", "CURRENT_CLASS"),
        help="Compare two Diagnosis classes instead of the default train/test-split sanity check "
             "(demonstrates the mechanism catching a real distributional difference).",
    )
    args = parser.parse_args()

    df = load_processed()

    if args.classes:
        ref_class, cur_class = args.classes
        reference_df = df[df[TARGET_COLUMN] == ref_class]
        current_df = df[df[TARGET_COLUMN] == cur_class]
        if reference_df.empty or current_df.empty:
            raise SystemExit(f"One of the classes {args.classes} has no rows in the dataset.")
        mode_description = f"class comparison: '{ref_class}' (reference) vs '{cur_class}' (current)"
    else:
        X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
        X_trainval, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        reference_df, current_df = X_trainval, X_test
        mode_description = "sanity check: training split (reference) vs. held-out test split (current)"

    report = compute_drift_report(reference_df, current_df)
    report["mode"] = mode_description

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift report — {mode_description}")
    print(f"Reference n={report['reference_n']}, current n={report['current_n']}\n")
    for feature, r in report["per_feature"].items():
        flag = "DRIFT" if r["drift_detected"] else "     "
        print(f"  [{flag}] {feature}: KS={r['ks_statistic']:.4f} p={r['p_value']:.4f}")
    print(f"\n{report['features_with_drift']}/{report['total_features']} features show drift "
          f"(p < {DRIFT_P_THRESHOLD})")
    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
