"""Load and validate the Indian food delivery dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_data(path=None) -> pd.DataFrame:
    """Load the CSV, validate schema, coerce types and drop broken rows."""
    path = Path(path) if path else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from {config.DATASET_URL} "
            f"and place it at data/IndianFoodDelivery.csv"
        )
    df = pd.read_csv(path)

    required = config.FEATURES + [config.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.drop_duplicates(subset=config.ID_COLS, keep="first") if all(
        c in df.columns for c in config.ID_COLS
    ) else df.drop_duplicates()

    for c in config.NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in config.CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).str.strip()
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"[data_loader] dropped {before - len(df)} rows with missing required values")

    return df.reset_index(drop=True)
