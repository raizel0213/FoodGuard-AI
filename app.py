"""FoodGuard AI - entry point.

Run:  streamlit run app.py
The app trains the models automatically on first launch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

import config
from src.boot import get_bundle, get_scored_df
from src.ui import apply_plotly_theme, inject_css, page_header

st.set_page_config(page_title="FoodGuard AI", page_icon=None, layout="wide")

inject_css()

try:
    bundle = get_bundle()
    scored = get_scored_df(bundle)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ hero
st.markdown(
    """
    <div class='fg-card'>
      <h1 class='fg-hero-title'>FoodGuard AI</h1>
      <p class='fg-sub'>Food-delivery problem detection and root-cause analysis for the
      Indian food delivery dataset. Predicts whether an order will report a delivery
      issue <i>before</i> dispatch, explains the probable contributing factors and
      recommends corrective actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m = bundle["test_metrics_at_threshold"]
stats = bundle["data_stats"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Orders analysed", f"{stats['n_rows']:,}")
k2.metric("Historical issue rate", f"{stats['issue_rate']:.1%}")
k3.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
k4.metric("Issue recall (test)", f"{m['recall']:.0%}")
k5.metric("Best model", bundle["model_name"])

st.caption(
    f"Decision threshold {bundle['threshold']:.2f} (F1-optimal on out-of-fold training "
    f"predictions; an accuracy-optimal operating point is also reported on the Model "
    f"Performance page). Metrics are honest holdout results - no leakage: "
    f"order_id and delivery_time_minutes are excluded from all features."
)

# ------------------------------------------------- snapshot: where issues happen
left, right = st.columns(2)

with left:
    by_city = (
        scored.groupby("city")["delivery_issue_reported"]
        .agg(["count", "mean"]).reset_index()
        .sort_values("mean", ascending=False)
    )
    fig = px.bar(
        by_city, x="city", y="mean", color="mean",
        color_continuous_scale=["#0ea5e9", "#22d3ee"],
        labels={"mean": "issue rate", "count": "orders", "city": ""},
        title="Issue rate by city",
    )
    fig.add_hline(y=stats["issue_rate"], line_dash="dot",
                  annotation_text="overall rate")
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

with right:
    by_traffic = (
        scored.groupby("traffic_index")["delivery_issue_reported"]
        .agg(["count", "mean"]).reset_index()
    )
    fig = px.bar(
        by_traffic, x="traffic_index", y="mean", color="mean",
        color_continuous_scale=["#38bdf8", "#fb923c"],
        labels={"mean": "issue rate", "traffic_index": "traffic index (1-10)"},
        title="Issue rate vs traffic conditions",
    )
    fig.add_hline(y=stats["issue_rate"], line_dash="dot",
                  annotation_text="overall rate")
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

st.markdown("### Explore the app")
c1, c2, c3 = st.columns(3)
c1.markdown("**Order Analysis** - score a single order, see risk, factors, actions")
c2.markdown("**Model Performance** - CV comparison, ROC/PR, confusion matrix")
c3.markdown("**Root Cause Analysis** - aggregate drivers behind flagged orders")
