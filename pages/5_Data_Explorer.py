"""Page 5: Data Explorer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.boot import get_bundle
from src.data_loader import load_data
from src.ui import apply_plotly_theme, inject_css, page_header

st.set_page_config(page_title="Data Explorer - FoodGuard AI", layout="wide")
inject_css()

bundle = get_bundle()
df = load_data()
page_header(
    "Data Explorer",
    "Raw dataset with live filters and distributions. Target: delivery_issue_reported.",
)

c1, c2, c3, c4 = st.columns(4)
sel_city = c1.selectbox("City", ["All"] + sorted(df["city"].unique().tolist()))
sel_vehicle = c2.selectbox("Vehicle", ["All"] + sorted(df["vehicle_type"].unique().tolist()))
sel_peak = c3.selectbox("Peak hour", ["All", "Yes", "No"])
sel_target = c4.selectbox("Delivery issue", ["All", "Yes (1)", "No (0)"])

view = df.copy()
if sel_city != "All":
    view = view[view["city"] == sel_city]
if sel_vehicle != "All":
    view = view[view["vehicle_type"] == sel_vehicle]
if sel_peak == "Yes":
    view = view[view["is_peak_hour"] == 1]
elif sel_peak == "No":
    view = view[view["is_peak_hour"] == 0]
if sel_target == "Yes (1)":
    view = view[view[config.TARGET] == 1]
elif sel_target == "No (0)":
    view = view[view[config.TARGET] == 0]

k1, k2, k3 = st.columns(3)
k1.metric("Rows shown", f"{len(view):,}")
k2.metric("Issue rate in view", f"{view[config.TARGET].mean():.1%}")
k3.metric("Avg delivery time (min)", f"{view['delivery_time_minutes'].mean():.1f}")

st.dataframe(view, width="stretch", height=360)

left, right = st.columns(2)

with left:
    num = st.selectbox("Distribution of", config.NUMERIC_FEATURES)
    fig = px.histogram(
        view, x=num, color=str(config.TARGET), nbins=30, barmode="overlay",
        color_discrete_map={"0": "#0ea5e9", "1": "#f87171"},
        labels={num: num, config.TARGET: "delivery_issue_reported"},
        title=f"{num} by outcome",
    )
    st.plotly_chart(apply_plotly_theme(fig, height=360), width="stretch")

with right:
    fig = px.box(
        view, x=view[config.TARGET].astype(str), y=num, color=view[config.TARGET].astype(str),
        color_discrete_map={"0": "#0ea5e9", "1": "#f87171"},
        labels={"x": "delivery_issue_reported", "y": num},
        title=f"{num} vs outcome",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_plotly_theme(fig, height=360), width="stretch")

st.subheader("Correlation matrix (numeric features + target)")
corr_cols = config.NUMERIC_FEATURES + ["delivery_time_minutes", config.TARGET]
corr = view[corr_cols].corr(numeric_only=True)
fig = px.imshow(
    corr, text_auto=".2f", aspect="auto",
    color_continuous_scale=["#0b1e3a", "#0ea5e9", "#22d3ee"], zmin=-1, zmax=1,
    title="Pearson correlation (delivery_time_minutes shown for context - excluded from the model)",
)
st.plotly_chart(apply_plotly_theme(fig, height=480), width="stretch")
st.caption(
    "delivery_time_minutes correlates with the target but is only known after delivery - "
    "it is excluded from the model to prevent data leakage."
)
