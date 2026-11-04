"""Root-cause explanation layer.

Prefers SHAP for tree-based models; always has a transparent, model-agnostic
fallback (leave-one-out-to-reference probability delta). All outputs are
phrased as *probable contributing factors*, never confirmed causes.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

TREE_MODELS = {"Random Forest", "Extra Trees", "Gradient Boosting",
               "Hist Gradient Boosting", "XGBoost"}


def _pyval(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v


def _orig_feature(tname: str, features: list) -> str:
    """Map a transformed column name like 'cat__city_Delhi' back to 'city'."""
    t = str(tname).split("__", 1)[-1]
    for f in features:
        if t == f or t.startswith(f + "_"):
            return f
    return t


def explain_order(pipe, bundle: dict, row_df: pd.DataFrame) -> list:
    """Explain one order. Returns [{feature, value, contribution, method}] sorted
    by absolute contribution (descending)."""
    model_name = bundle["model_name"]
    features = bundle["features"]["all"]
    if model_name in TREE_MODELS:
        try:
            return _shap_contributions(pipe, row_df, features)
        except Exception:
            pass  # any SHAP incompatibility -> transparent fallback
    return _delta_contributions(pipe, row_df, bundle, features)


def _shap_contributions(pipe, row_df, features) -> list:
    import shap
    Xt = pipe[:-1].transform(row_df)
    clf = pipe.named_steps["model"]
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    while sv.ndim > 1:
        sv = sv[0]
    names = pipe[:-1].get_feature_names_out()
    agg = defaultdict(float)
    for nm, v in zip(names, sv):
        agg[_orig_feature(nm, features)] += float(v)
    vals = row_df.iloc[0]
    out = [{"feature": f, "value": _pyval(vals[f]),
            "contribution": float(agg.get(f, 0.0)), "method": "SHAP"}
           for f in features if f in agg]
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


def _delta_contributions(pipe, row_df, bundle, features) -> list:
    """Probability drop when the feature is set back to the training baseline:
    contribution_i = P(full order) - P(order with feature i replaced)."""
    p_full = float(pipe.predict_proba(row_df)[0, 1])
    base = bundle["baseline"]
    out = []
    for f in features:
        mod = row_df.copy()
        mod[f] = base[f]
        p_wo = float(pipe.predict_proba(mod)[0, 1])
        out.append({"feature": f, "value": _pyval(row_df.iloc[0][f]),
                    "contribution": p_full - p_wo, "method": "reference-delta"})
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


def aggregate_causes(pipe, X_flagged: pd.DataFrame, bundle: dict) -> list:
    """Mean per-feature probability delta across all flagged orders (batch,
    vectorised: one predict call per feature). Returns risk-increasing factors
    sorted descending."""
    if X_flagged is None or len(X_flagged) == 0:
        return []
    p_full = pipe.predict_proba(X_flagged)[:, 1]
    base = bundle["baseline"]
    rows = []
    for f in bundle["features"]["all"]:
        X_mod = X_flagged.copy()
        X_mod[f] = base[f]
        p_wo = pipe.predict_proba(X_mod)[:, 1]
        rows.append({"feature": f,
                     "mean_contribution": float(np.mean(p_full - p_wo)),
                     "n": int(len(X_flagged))})
    rows.sort(key=lambda d: d["mean_contribution"], reverse=True)
    return rows


def describe_contribution(c: dict) -> str:
    direction = "increases risk" if c["contribution"] > 0 else "reduces risk"
    return f"{c['feature']} = {c['value']} ({direction})"
