"""Page 4: Root Cause Analysis - aggregate drivers and evidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.boot import get_aggregate_causes, get_bundle, get_scored_df
from src.ui import apply_plotly_theme, inject_css, page_header, risk_badge_html

st.set_page_config(page_title="Root Cause Analysis - FoodGuard AI", layout="wide")
inject_css()

bundle = get_bundle()
scored = get_scored_df(bundle)
page_header(
    "Root Cause Analysis",
    "Aggregate investigation of flagged orders: which factors the model holds "
    "responsible and what the training data says about each of them.",
)

ev = bundle["evidence"]
base = ev["base_rate"]

c1, c2, c3 = st.columns(3)
cities = [""] + sorted(scored["city"].unique().tolist())
vehicles = [""] + sorted(scored["vehicle_type"].unique().tolist())
sel_city = c1.selectbox("Filter by city", cities)
sel_vehicle = c2.selectbox("Filter by vehicle", vehicles)
scope = c3.selectbox("Orders to analyse", ["flagged (predicted issues)", "all orders"])

flag = "flagged" if scope.startswith("flagged") else "all"
causes = get_aggregate_causes(bundle, flag, sel_city, sel_vehicle)

st.subheader(f"Aggregate probable contributing factors - {scope.split(' ')[0]} orders")
if causes:
    cdf = pd.DataFrame(causes)
    cdf["label"] = cdf["feature"].apply(
        lambda f: f"{f}  (+{cdf.loc[cdf['feature'] == f, 'mean_contribution'].iloc[0]:+.3f})"
        if cdf.loc[cdf["feature"] == f, "mean_contribution"].iloc[0] > 0 else f
    )
    cdf = cdf.sort_values("mean_contribution")
    colors = ["#fb923c" if v > 0 else "#34d399" for v in cdf["mean_contribution"]]
    fig = px.bar(
        cdf, x="mean_contribution", y="feature", orientation="h",
        labels={"mean_contribution": "mean probability delta when feature is reset to baseline",
                "feature": ""},
        title="Drivers of predicted risk (reference-delta method)",
    )
    fig.update_traces(marker_color=colors)
    st.plotly_chart(apply_plotly_theme(fig, height=400), width="stretch")
    st.caption(
        "Positive bars: resetting the factor to its baseline lowers predicted risk, so it is a "
        "probable contributing factor. This is an association learned from data, not a confirmed "
        "operational root cause."
    )
else:
    st.info("No orders match this filter.")

st.subheader("Evidence tables - observed issue rate per condition (training data)")
top_factors = [c["feature"] for c in (causes or [])[:3]]
cols = st.columns(max(len(top_factors), 1))
for i, f in enumerate(top_factors):
    with cols[i]:
        st.markdown(f"**{f}**")
        if f in ev["numeric"]:
            t = pd.DataFrame({
                "condition": ev["numeric"][f]["labels"],
                "orders": ev["numeric"][f]["n"],
                "issue rate": ev["numeric"][f]["issue_rate"],
            })
        else:
            t = pd.DataFrame({
                "condition": ev["categorical"][f]["level"],
                "orders": ev["categorical"][f]["n"],
                "issue rate": ev["categorical"][f]["issue_rate"],
            })
        t["vs overall"] = t["issue rate"] - base
        st.dataframe(
            t.style.format({"issue rate": "{:.1%}", "vs overall": "{:+.1%}"}),
            hide_index=True, width="stretch",
        )

st.subheader("Highest-risk flagged orders")
sub = scored[scored["issue_predicted"]].sort_values("issue_probability", ascending=False)
if sel_city:
    sub = sub[sub["city"] == sel_city]
if sel_vehicle:
    sub = sub[sub["vehicle_type"] == sel_vehicle]
show_cols = ["order_id", "city", "distance_km", "traffic_index", "rainfall_mm",
             "vehicle_type", "issue_probability", "risk"]
st.dataframe(sub[show_cols].head(15), hide_index=True, width="stretch")
if len(sub):
    st.markdown(f"Worst case: {risk_badge_html(sub.iloc[0]['risk'])} at "
                f"{sub.iloc[0]['issue_probability']:.0%} probability.", unsafe_allow_html=True)
