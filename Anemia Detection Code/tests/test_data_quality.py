"""Regression guards for the two root causes documented in
analysis/FINDINGS.md. These exist to catch someone accidentally
reintroducing the duplicate-row leakage or losing the ablation evidence,
not to assert a specific accuracy ceiling (the true label rule genuinely
supports near-100% accuracy once the leakage is fixed — see FINDINGS.md).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/anemia.csv"


def load_raw():
    return pd.read_csv(DATA_PATH).drop(columns=["Name", "Address", "Phone"], errors="ignore")


def test_raw_dataset_has_duplicate_rows():
    # Documents the known state of the shipped dataset — if this ever fails
    # because the CSV was cleaned upstream, the dedup step in train.py
    # becomes a no-op rather than a bug fix, which is worth knowing.
    df = load_raw()
    assert df.duplicated().sum() > 0


def test_dedup_actually_removes_rows():
    df = load_raw()
    deduped = df.drop_duplicates()
    assert len(deduped) < len(df)
    assert deduped.duplicated().sum() == 0


def test_hemoglobin_gender_threshold_has_no_overlap():
    df = load_raw().drop_duplicates()
    for gender in df["Gender"].unique():
        sub = df[df["Gender"] == gender]
        pos_max = sub.loc[sub["Result"] == 1, "Hemoglobin"].max()
        neg_min = sub.loc[sub["Result"] == 0, "Hemoglobin"].min()
        assert pos_max < neg_min, (
            f"Gender={gender}: expected anemic/not-anemic Hemoglobin ranges not to "
            f"overlap (got max anemic={pos_max}, min not-anemic={neg_min})"
        )


def test_accuracy_drops_sharply_without_hemoglobin():
    """Guards the ablation claim in FINDINGS.md: MCH/MCHC/MCV/Gender alone
    should NOT come close to matching Hemoglobin-inclusive accuracy. If this
    ever fails, either the ablation logic is broken or a new feature has
    started leaking the label the same way Hemoglobin does — worth a look
    either way."""
    df = load_raw().drop_duplicates()
    y = df["Result"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])

    with_hb = cross_val_score(model, df.drop(columns=["Result"]), y, cv=cv).mean()
    without_hb = cross_val_score(model, df.drop(columns=["Result", "Hemoglobin"]), y, cv=cv).mean()

    assert with_hb > 0.95
    assert with_hb - without_hb > 0.25
