"""Tests for src/data/validate_data.py — duplicate detection, physiological
range validation, and the leakage/trivial-reconstruction check."""

import pandas as pd

from src.data.load_data import load_raw
from src.data.validate_data import PHYSIOLOGICAL_RANGES, clean_and_validate, find_implausible_rows


def test_raw_dataset_loads():
    df = load_raw()
    assert len(df) > 0
    assert "Diagnosis" in df.columns


def test_find_implausible_rows_flags_known_bad_values():
    df = pd.DataFrame({
        "WBC": [7.0, 7.0], "LYMp": [30, 30], "NEUTp": [50, 50], "LYMn": [2, 2], "NEUTn": [4, 4],
        "RBC": [5.0, 5.0], "HGB": [14.0, -10.0],  # second row: impossible negative HGB
        "HCT": [40, 40], "MCV": [90, 90], "MCH": [28, 28], "MCHC": [33, 33],
        "PLT": [250, 250], "PDW": [15, 15], "PCT": [0.2, 0.2],
    })
    mask = find_implausible_rows(df)
    assert mask.tolist() == [False, True]


def test_clean_and_validate_removes_duplicates_and_implausible_rows():
    df_raw = load_raw()
    df_clean, report = clean_and_validate(df_raw)

    assert report["raw_rows"] == len(df_raw)
    assert report["duplicate_rows_removed"] > 0
    assert report["physiologically_implausible_rows_removed"] > 0
    assert len(df_clean) == report["rows_after_cleaning"]
    assert df_clean.duplicated().sum() == 0
    assert find_implausible_rows(df_clean).sum() == 0
    assert report["validation_status"] == "passed"


def test_no_single_feature_reconstructs_the_label():
    """Regression guard: the whole point of the new dataset is that the
    label isn't trivially derivable from one column the way it was in the
    original Gender+Hemoglobin dataset (see analysis/FINDINGS.md)."""
    df_clean, report = clean_and_validate(load_raw())
    assert report["max_single_feature_accuracy"] < 0.75


def test_physiological_ranges_cover_all_feature_columns():
    from src.data.load_data import FEATURE_COLUMNS
    assert set(PHYSIOLOGICAL_RANGES.keys()) == set(FEATURE_COLUMNS)
