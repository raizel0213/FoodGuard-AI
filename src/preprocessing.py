"""Leakage-safe preprocessing: imputation, scaling, one-hot encoding."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES),
         ("cat", categorical, config.CATEGORICAL_FEATURES)],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights per sample (computed on training data only)."""
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    w = np.ones(n, dtype=float)
    for c in classes:
        w[y == c] = n / (len(classes) * int((y == c).sum()))
    return w
