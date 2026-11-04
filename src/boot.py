"""App-layer bootstrap shared by app.py and every page:
path setup, auto-training on first run, cached artefacts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402


def _load_bundle_on_disk():
    """Try to unpickle the model. Returns (bundle, error). Never raises."""
    try:
        import joblib

        return joblib.load(config.MODEL_PATH), None
    except Exception as exc:  # any unpickle failure must trigger a retrain
        return None, exc


def ensure_model():
    """Guarantee a usable model bundle built with the *running* scikit-learn.

    Pickled scikit-learn estimators only unpickle reliably with the sklearn
    version that created them (private classes such as
    ``sklearn.compose._column_transformer._RemainderColsList`` appear and
    disappear between releases). If the shipped model file cannot be loaded -
    e.g. after a scikit-learn/Python upgrade - or was trained with a
    different version, it is retrained automatically here with the local
    scikit-learn. Training takes well under a minute on the 1,000-row dataset.
    """
    bundle, err = _load_bundle_on_disk()
    if bundle is not None:
        import sklearn

        saved = None
        if isinstance(bundle, dict) and isinstance(bundle.get("meta"), dict):
            saved = bundle["meta"].get("sklearn_version")
        if saved == sklearn.__version__:
            return bundle
        reason = (f"model was trained with scikit-learn {saved} but this "
                  f"environment has {sklearn.__version__}")
    else:
        reason = f"model file could not be unpickled ({type(err).__name__}: {err})"

    print(f"[boot] {reason} -> retraining with the local scikit-learn ...")
    with st.spinner("Rebuilding the model for your scikit-learn version "
                    "(one-time, usually under a minute)..."):
        import src.train as train_mod
        train_mod.main(verbose=False)

    bundle, err2 = _load_bundle_on_disk()
    if bundle is None:
        raise RuntimeError(
            "FoodGuard AI could not load or rebuild its model. "
            f"Original problem: {reason}. Retraining also failed: {err2}"
        )
    return bundle


@st.cache_resource(show_spinner=False)
def get_bundle():
    return ensure_model()


@st.cache_resource(show_spinner=False)
def get_scored_df(_bundle):
    from src.data_loader import load_data
    from src.predict import FoodGuardPredictor
    df = load_data()
    return FoodGuardPredictor(_bundle).score_dataset(df)


@st.cache_resource(show_spinner=False)
def get_aggregate_causes(_bundle, flagged: str, city: str, vehicle: str):
    from src.root_cause import aggregate_causes
    df = get_scored_df(_bundle)
    sub = df
    if flagged == "flagged":
        sub = sub[sub["issue_predicted"]]
    if city:
        sub = sub[sub["city"] == city]
    if vehicle:
        sub = sub[sub["vehicle_type"] == vehicle]
    return aggregate_causes(_bundle["pipeline"], sub[_bundle["features"]["all"]], _bundle)
