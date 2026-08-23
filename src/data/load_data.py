"""Loads the raw CBC dataset. See reports/data_quality/DATASET_CARD.md for
source, license, and schema."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "anemia_types_cbc.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "anemia_types_cbc_clean.csv"

FEATURE_COLUMNS = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT",
    "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT",
]
TARGET_COLUMN = "Diagnosis"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def load_processed() -> pd.DataFrame:
    """Loads the cleaned dataset, regenerating it if it doesn't exist yet."""
    if not PROCESSED_PATH.exists():
        from src.data.validate_data import clean_and_validate

        df, _ = clean_and_validate(load_raw())
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        return df
    return pd.read_csv(PROCESSED_PATH)
