"""Train, compare and select the delivery-issue model.

Run from the project root:  python -m src.train
Produces: models/foodguard_model.joblib, reports/model_comparison.csv,
reports/evaluation_report.json, reports/feature_importance.csv
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import config
from src.data_loader import load_data
from src.preprocessing import balanced_sample_weight, build_preprocessor

warnings.filterwarnings("ignore")

SW_MODELS = {"Gradient Boosting", "Hist Gradient Boosting"}  # need sample_weight


def _make_model(name: str, y_fit=None) -> Pipeline:
    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                 random_state=config.RANDOM_STATE)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight="balanced", random_state=config.RANDOM_STATE,
                                     n_jobs=-1)
    elif name == "Extra Trees":
        clf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight="balanced", random_state=config.RANDOM_STATE,
                                   n_jobs=-1)
    elif name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.06, max_depth=3,
                                         random_state=config.RANDOM_STATE)
    elif name == "Hist Gradient Boosting":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             min_samples_leaf=5, random_state=config.RANDOM_STATE)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        spw = 1.0
        if y_fit is not None:
            y_arr = np.asarray(y_fit)
            spw = float((y_arr == 0).sum() / max((y_arr == 1).sum(), 1))
        clf = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9,
                            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                            random_state=config.RANDOM_STATE, verbosity=0, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("pre", build_preprocessor()), ("model", clf)])


def _fit(pipe: Pipeline, name: str, X, y):
    if name in SW_MODELS:
        pipe.fit(X, y, model__sample_weight=balanced_sample_weight(np.asarray(y)))
    else:
        pipe.fit(X, y)
    return pipe


def _clf_metrics(y_true, probs, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def _best_f1_threshold(y_true, probs) -> float:
    prec, rec, thrs = precision_recall_curve(np.asarray(y_true), probs)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    ix = int(np.nanargmax(f1))
    return float(thrs[ix])


def _best_accuracy_threshold(y_true, probs) -> float:
    """Threshold that maximizes accuracy on out-of-fold training predictions."""
    y_true = np.asarray(y_true)
    grid = np.linspace(0.05, 0.95, 181)
    accs = [float(np.mean((probs >= t).astype(int) == y_true)) for t in grid]
    return float(grid[int(np.argmax(accs))])


def _build_evidence(X_tr: pd.DataFrame, y_tr: pd.Series) -> dict:
    """Empirical issue rates per feature bin (training data only)."""
    ev = {"base_rate": float(y_tr.mean()), "numeric": {}, "categorical": {}}
    for f in config.NUMERIC_FEATURES:
        s = X_tr[f]
        try:
            edges = np.unique(np.quantile(s, [0.0, 0.25, 0.5, 0.75, 1.0]))
        except Exception:
            edges = np.array([s.min(), s.max()])
        if len(edges) < 2:
            edges = np.array([s.min() - 0.5, s.max() + 0.5])
        binned = pd.cut(s, bins=edges, include_lowest=True)
        tmp = pd.DataFrame({"b": binned, "y": y_tr}).groupby("b", observed=True)["y"].agg(["count", "mean"])
        ev["numeric"][f] = {
            "edges": [float(e) for e in edges],
            "labels": [str(i) for i in tmp.index],
            "n": [int(x) for x in tmp["count"]],
            "issue_rate": [float(x) for x in tmp["mean"]],
        }
    for f in config.CATEGORICAL_FEATURES:
        tmp = pd.DataFrame({"c": X_tr[f], "y": y_tr}).groupby("c")["y"].agg(["count", "mean"])
        ev["categorical"][f] = {
            "level": [str(i) for i in tmp.index],
            "n": [int(x) for x in tmp["count"]],
            "issue_rate": [float(x) for x in tmp["mean"]],
        }
    return ev


def _py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _downsample(arr, n=120):
    a = np.asarray(arr, dtype=float)
    if len(a) <= n:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return [float(x) for x in a[idx]]


def main(verbose: bool = True) -> dict:
    t0 = time.time()
    df = load_data()
    X, y = df[config.FEATURES], df[config.TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    model_names = ["Logistic Regression", "Random Forest", "Extra Trees",
                   "Gradient Boosting", "Hist Gradient Boosting"]
    try:
        import xgboost  # noqa: F401
        model_names.append("XGBoost")
    except ImportError:
        if verbose:
            print("[train] xgboost not installed - skipping XGBoost")

    cv = StratifiedKFold(n_splits=config.N_CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    rows, oof = [], {}
    for name in model_names:
        oof_probs = np.zeros(len(y_tr))
        fold_metrics = []
        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            m = _make_model(name, y_tr.iloc[tr_idx])
            _fit(m, name, X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
            oof_probs[va_idx] = p
            fold_metrics.append(_clf_metrics(y_tr.iloc[va_idx], p, 0.5))
        oof[name] = oof_probs
        rows.append({
            "model": name,
            "cv_roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "cv_roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
            "cv_pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
            "cv_f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
            "cv_precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
            "cv_recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
            "cv_accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        })

    comp = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    best_name = str(comp.iloc[0]["model"])
    threshold = _best_f1_threshold(y_tr, oof[best_name])
    accuracy_threshold = _best_accuracy_threshold(y_tr, oof[best_name])

    final = _make_model(best_name, y_tr)
    _fit(final, best_name, X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]

    test_at_thr = _clf_metrics(y_te, p_te, threshold)
    test_at_050 = _clf_metrics(y_te, p_te, 0.5)
    test_at_acc = _clf_metrics(y_te, p_te, accuracy_threshold)
    cm = confusion_matrix(y_te, (p_te >= threshold).astype(int))
    report_dict = classification_report(y_te, (p_te >= threshold).astype(int),
                                        target_names=["no issue", "issue"], zero_division=0,
                                        output_dict=True)

    fpr, tpr, _ = roc_curve(y_te, p_te)
    pr_r, pr_p, _ = precision_recall_curve(y_te, p_te)
    cal_p, cal_y = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")

    pi = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=15,
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = sorted(
        [{"feature": f, "importance_mean": float(m), "importance_std": float(s)}
         for f, m, s in zip(config.FEATURES, pi.importances_mean, pi.importances_std)],
        key=lambda d: d["importance_mean"], reverse=True,
    )

    baseline = {}
    for f in config.NUMERIC_FEATURES:
        baseline[f] = float(X_tr[f].median())
    for f in config.CATEGORICAL_FEATURES:
        baseline[f] = str(X_tr[f].mode().iloc[0])

    train_ranges = {
        f: {"min": float(X_tr[f].min()), "max": float(X_tr[f].max()),
            "median": float(X_tr[f].median())}
        for f in config.NUMERIC_FEATURES
    }
    categorical_levels = {f: sorted(X_tr[f].astype(str).unique().tolist())
                          for f in config.CATEGORICAL_FEATURES}

    bundle = {
        "pipeline": final,
        "model_name": best_name,
        "threshold": threshold,
        "features": {"numeric": config.NUMERIC_FEATURES,
                     "categorical": config.CATEGORICAL_FEATURES,
                     "all": config.FEATURES},
        "baseline": baseline,
        "train_ranges": train_ranges,
        "categorical_levels": categorical_levels,
        "evidence": _build_evidence(X_tr, y_tr),
        "feature_importance": importance,
        "test_metrics_at_threshold": test_at_thr,
        "test_metrics_at_050": test_at_050,
        "test_metrics_at_accuracy_threshold": test_at_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "curves": {
            "roc": {"fpr": _downsample(fpr), "tpr": _downsample(tpr),
                    "auc": test_at_thr["roc_auc"]},
            "pr": {"recall": _downsample(pr_r), "precision": _downsample(pr_p),
                   "auc": test_at_thr["pr_auc"]},
            "calibration": {"prob_pred": [float(x) for x in cal_p],
                            "prob_true": [float(x) for x in cal_y]},
        },
        "cv_comparison": comp.to_dict(orient="records"),
        "data_stats": {
            "n_rows": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "issue_rate": float(df[config.TARGET].mean()),
            "base_rate_train": float(y_tr.mean()),
        },
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "threshold_strategy": "max F1 on out-of-fold training predictions "
                                  "(accuracy-optimal operating point also reported)",
            "accuracy_threshold": float(accuracy_threshold),
            "selection_metric": "mean CV ROC-AUC",
            "leakage_policy": "order_id and delivery_time_minutes excluded; "
                              "only features available at dispatch time are used",
        },
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(importance).to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    eval_report = {
        "best_model": best_name,
        "threshold": threshold,
        "accuracy_threshold": float(accuracy_threshold),
        "test_metrics_at_threshold": test_at_thr,
        "test_metrics_at_050": test_at_050,
        "test_metrics_at_accuracy_threshold": test_at_acc,
        "confusion_matrix": {"labels": [["TN", "FP"], ["FN", "TP"]], "values": cm.tolist()},
        "classification_report": report_dict,
        "cv_comparison": comp.to_dict(orient="records"),
        "feature_importance": importance,
        "data_stats": bundle["data_stats"],
        "meta": bundle["meta"],
    }
    with open(config.REPORTS_DIR / "evaluation_report.json", "w") as fh:
        json.dump(eval_report, fh, indent=2, default=_py)

    if verbose:
        print("\n=== MODEL COMPARISON (5-fold CV on training split) ===")
        print(comp.round(4).to_string(index=False))
        print(f"\nBest model: {best_name}  |  F1-optimal threshold: {threshold:.3f} "
              f"(accuracy-optimal: {accuracy_threshold:.3f})")
        print(f"Test metrics @F1-threshold:  {json.dumps(test_at_thr, indent=2)}")
        print(f"Test metrics @acc-threshold: {json.dumps(test_at_acc, indent=2)}")
        print(f"Test metrics @0.50:      {json.dumps(test_at_050, indent=2)}")
        print(f"Confusion matrix @threshold:\n{cm}")
        print(f"\nDone in {time.time() - t0:.1f}s -> {config.MODEL_PATH}")

    return {"best_model": best_name, "threshold": threshold,
            "test_metrics": test_at_thr, "cv_comparison": comp}


if __name__ == "__main__":
    main()
