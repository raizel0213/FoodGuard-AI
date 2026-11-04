"""Prediction service used by the Streamlit app (and any external caller)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.recommendations import generate_recommendations
from src.root_cause import explain_order

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


class FoodGuardPredictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipe = bundle["pipeline"]

    @classmethod
    def load(cls, path=None) -> "FoodGuardPredictor":
        import joblib
        path = path or config.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run:  python -m src.train  (or just start the app - "
                "it trains automatically on first launch)."
            )
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ input
    def _build_row(self, order: dict):
        row, warns = {}, []
        ranges = self.bundle["train_ranges"]
        levels = self.bundle["categorical_levels"]
        for f in config.NUMERIC_FEATURES:
            v = order.get(f)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
                warns.append(f"{f}: invalid number, imputed with training median")
            lo, hi = ranges[f]["min"], ranges[f]["max"]
            if np.isfinite(v):
                if v < lo or v > hi:
                    warns.append(f"{f}: {v:g} is outside the observed training range "
                                 f"[{lo:g}, {hi:g}] - clipped")
                    v = float(np.clip(v, lo, hi))
            row[f] = v
        for f in config.CATEGORICAL_FEATURES:
            v = str(order.get(f, "")).strip()
            if v not in levels[f]:
                warns.append(f"{f}: '{v}' was not seen in training "
                             f"(known: {', '.join(levels[f])}) - prediction may be less reliable")
            row[f] = v
        return pd.DataFrame([row], columns=config.FEATURES), warns

    # ----------------------------------------------------------------- output
    def risk_band(self, p: float) -> str:
        """Threshold-relative bands so labels stay meaningful whatever the tuned
        threshold turns out to be: Critical = flagged, High >= 0.7x thr,
        Medium >= 0.4x thr, Low below."""
        thr = float(self.bundle["threshold"])
        if p >= thr:
            return "Critical"
        if p >= 0.7 * thr:
            return "High"
        if p >= 0.4 * thr:
            return "Medium"
        return "Low"

    def predict(self, order: dict, explain: bool = True, recommend: bool = True) -> dict:
        row, warns = self._build_row(order)
        p = float(self.pipe.predict_proba(row)[0, 1])
        out = {
            "probability": p,
            "issue_predicted": bool(p >= self.bundle["threshold"]),
            "threshold": float(self.bundle["threshold"]),
            "risk": self.risk_band(p),
            "model": self.bundle["model_name"],
            "warnings": warns,
        }
        if explain:
            out["contributions"] = explain_order(self.pipe, self.bundle, row)
        if recommend:
            out["recommendations"] = generate_recommendations(
                row.iloc[0].to_dict(), out.get("contributions", []), self.bundle
            )
        return out

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.pipe.predict_proba(X[config.FEATURES])[:, 1]
        return pd.DataFrame({
            "issue_probability": probs,
            "risk": [self.risk_band(p) for p in probs],
            "issue_predicted": probs >= self.bundle["threshold"],
        })

    def score_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[["issue_probability", "risk", "issue_predicted"]] = self.predict_batch(df)
        return out
