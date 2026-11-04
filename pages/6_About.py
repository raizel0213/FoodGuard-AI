"""Page 6: About - methodology, attribution, deployment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.boot import get_bundle
from src.ui import inject_css, page_header

st.set_page_config(page_title="About - FoodGuard AI", layout="wide")
inject_css()

bundle = get_bundle()
page_header("About FoodGuard AI", "What this system does, how it works and where it stops.")

st.markdown(
    """
    <div class='fg-card'>
    <b>FoodGuard AI</b> predicts whether a food-delivery order will report a delivery issue
    (late, damaged, wrong, missing - any reported problem) <i>before dispatch</i>, and turns the
    prediction into an investigation layer: probability, risk level, probable contributing
    factors and recommended corrective actions.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.subheader("Methodology")
    st.markdown(
        f"""
        - **Data**: {bundle['data_stats']['n_rows']:,} Indian food-delivery orders
          (YBI Foundation public dataset), stratified {int((1 - 0.2) * 100)}/{int(0.2 * 100)} split.
        - **Models compared** (5-fold stratified CV): Logistic Regression, Random Forest,
          Extra Trees, Gradient Boosting, Hist Gradient Boosting, XGBoost.
        - **Selection**: highest mean CV ROC-AUC -> **{bundle['model_name']}**.
        - **Class imbalance**: balanced class weights / sample weights (training folds only).
        - **Threshold**: {bundle['threshold']:.2f}, tuned for max F1 on out-of-fold training
          predictions - never on the test set. An accuracy-optimal operating point (~88%
          accuracy, minimal recall) is reported alongside it on the Model Performance page.
        - **Risk levels** (relative to the tuned threshold): Critical = flagged (>= threshold),
          High >= 0.7x threshold, Medium >= 0.4x threshold, Low below.
        - **Explanations**: SHAP for tree models with a transparent reference-delta fallback;
          aggregates use the reference-delta method in probability points.
        - **Recommendations**: rule engine grounded in empirical issue rates per condition
          from the training split.
        """
    )

with right:
    st.subheader("Data-leakage policy")
    st.markdown(
        """
        Only features available **at dispatch time** are used:
        city, distance, order value, traffic, rainfall, prep estimate, queue length,
        rider experience, peak-hour and weekend flags, vehicle type.

        Deliberately excluded:
        - `order_id` - pure identifier;
        - `delivery_time_minutes` - only known **after** delivery; using it would leak the outcome.

        The correlation heatmap in the Data Explorer shows why this matters:
        delivery time correlates with issues, but it is a consequence, not an input we can act on
        beforehand. Early-warning (this app) and post-mortem investigation are kept separate.
        """
    )

st.subheader("Honest performance statement (and the 95% question)")
m = bundle["test_metrics_at_threshold"]
st.markdown(
    f"""
    On the untouched holdout set ({bundle['data_stats']['n_test']} orders) the selected model
    reaches **{m['accuracy']:.1%} accuracy**, **{m['roc_auc']:.3f} ROC-AUC** and
    **{m['recall']:.0%} issue recall** at the tuned threshold. The naive baseline (always
    predict "no issue") scores ~87.6% accuracy with **0% recall** - it never catches a single
    problem order.

    A 95%+ accuracy demand was tested seriously before building this app: 8 model families
    (including tuned XGBoost and stacking/soft-voting ensembles) x 2 feature sets (with and
    without the post-event delivery_time_minutes) all land at **~88% CV accuracy / ~0.68
    ROC-AUC - the honest ceiling of this dataset**. See the Model Performance page for the
    full ceiling analysis. Numbers shown in this app are always the actual computed results;
    claiming more than the data supports would be fabrication.
    """
)

st.subheader("Dataset attribution")
st.markdown(
    "Dataset: **IndianFoodDelivery.csv** from the YBI Foundation Machine Learning collection "
    "(public educational dataset). Files and code were produced for a hackathon project."
)

st.subheader("Run it yourself")
st.code(
    "cd foodguard-ai\n"
    "python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate\n"
    "pip install -r requirements.txt\n"
    "python -m src.train            # optional - the app trains on first launch\n"
    "streamlit run app.py           # then open http://localhost:8501",
    language="bash",
)

st.subheader("Limitations")
st.markdown(
    """
    - 1,000 rows -> wide uncertainty on all metrics; treat them as estimates.
    - The dataset carries no labels for *why* an issue occurred, so the root-cause layer
      explains model behaviour, not verified business causes (no invented "wrong address" /
      "payment failure" categories).
    - Recommendations are operational suggestions grounded in observed rates, not guarantees.
    - Retrain periodically as operations change; the pipeline is deterministic (seeded).
    """
)
