"""Page 1: Overview Dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.boot import get_aggregate_causes, get_bundle, get_scored_df
from src.ui import apply_plotly_theme, inject_css, page_header

st.set_page_config(page_title="Overview Dashboard - FoodGuard AI", layout="wide")
inject_css()

bundle = get_bundle()
scored = get_scored_df(bundle)
page_header(
    "Overview Dashboard",
    "Live snapshot of the scored dataset: where delivery issues happen and which "
    "factors the model blames most.",
)

stats = bundle["data_stats"]
flagged = scored[scored["issue_predicted"]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders analysed", f"{len(scored):,}")
k2.metric("Flagged as at-risk", f"{len(flagged):,} ({len(flagged)/len(scored):.0%})")
k3.metric("True issue rate", f"{stats['issue_rate']:.1%}")
k4.metric("Model recall on test", f"{bundle['test_metrics_at_threshold']['recall']:.0%}")

left, right = st.columns(2)

with left:
    fig = px.pie(
        flagged["risk"].value_counts().reset_index(),
        names="risk", values="count", hole=0.55,
        title="Risk levels among flagged orders",
        color="risk",
        color_discrete_map={"Low": "#34d399", "Medium": "#fbbf24",
                            "High": "#fb923c", "Critical": "#f87171"},
    )
    st.plotly_chart(apply_plotly_theme(fig, height=340), width="stretch")

with right:
    bins = [0, 5, 10, 20, 31]
    labels = ["0-5 mm", "5-10 mm", "10-20 mm", "20+ mm"]
    tmp = scored.copy()
    tmp["rain_bin"] = pd.cut(tmp["rainfall_mm"], bins=bins, labels=labels, right=False)
    by_rain = tmp.groupby("rain_bin", observed=True)["delivery_issue_reported"].mean().reset_index()
    fig = px.bar(
        by_rain, x="rain_bin", y="delivery_issue_reported",
        color="delivery_issue_reported", color_continuous_scale=["#0ea5e9", "#22d3ee"],
        labels={"rain_bin": "rainfall", "delivery_issue_reported": "issue rate"},
        title="Issue rate vs rainfall",
    )
    fig.add_hline(y=stats["issue_rate"], line_dash="dot", annotation_text="overall rate")
    st.plotly_chart(apply_plotly_theme(fig, height=340), width="stretch")

st.subheader("Top probable contributing factors (flagged orders)")
causes = get_aggregate_causes(bundle, "flagged", "", "")
if causes:
    cdf = pd.DataFrame(causes).sort_values("mean_contribution")
    colors = ["#fb923c" if v > 0 else "#34d399" for v in cdf["mean_contribution"]]
    fig = px.bar(
        cdf, x="mean_contribution", y="feature", orientation="h",
        labels={"mean_contribution": "mean probability delta", "feature": ""},
        title="How much each factor raises the predicted issue probability",
    )
    fig.update_traces(marker_color=colors)
    st.plotly_chart(apply_plotly_theme(fig, height=380), width="stretch")
    st.caption(
        "Method: for every flagged order each feature is reset to its training baseline and "
        "the change in predicted probability is averaged. Positive bars are probable "
        "contributing factors - an interpretable association, not a confirmed root cause."
    )
else:
    st.info("No orders were flagged at the current threshold.")

