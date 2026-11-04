"""Page 2: Order Analysis - score a single order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

import config
from src.boot import get_bundle
from src.ui import apply_plotly_theme, inject_css, page_header, risk_badge_html

st.set_page_config(page_title="Order Analysis - FoodGuard AI", layout="wide")
inject_css()

bundle = get_bundle()
page_header(
    "Order Analysis",
    "Enter the dispatch-time details of one order. FoodGuard predicts the probability of "
    "a reported delivery issue, the probable contributing factors and recommended actions.",
)

r = bundle["train_ranges"]
levels = bundle["categorical_levels"]

with st.form("order_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", levels["city"])
        distance = st.slider("Distance (km)", 0.5, 15.0, 4.2, 0.25)
        value = st.slider("Order value (INR)", 200, 2400, 700, 10)
        traffic = st.slider("Traffic index (1=clear, 10=gridlock)", 1, 10, 5)
    with col2:
        rainfall = st.slider("Rainfall (mm)", 0.0, 30.0, 0.0, 0.5)
        prep = st.slider("Restaurant prep estimate (min)", 8, 35, 18)
        ahead = st.slider("Orders already ahead", 0, 10, 3)
        experience = st.slider("Rider experience (months)", 0, 60, 18)
    with col3:
        vehicle = st.selectbox("Vehicle type", levels["vehicle_type"])
        peak = st.checkbox("Peak hour")
        weekend = st.checkbox("Weekend")
        submitted = st.form_submit_button("Run FoodGuard analysis", width="stretch")

order = {
    "city": city,
    "distance_km": distance,
    "order_value_inr": value,
    "traffic_index": traffic,
    "rainfall_mm": rainfall,
    "restaurant_prep_est_min": prep,
    "orders_ahead": ahead,
    "rider_experience_months": experience,
    "is_peak_hour": int(peak),
    "is_weekend": int(weekend),
    "vehicle_type": vehicle,
}

if submitted:
    from src.predict import FoodGuardPredictor

    pred = FoodGuardPredictor(bundle).predict(order)
    thr = pred["threshold"]

    left, right = st.columns([1, 2])

    with left:
        color = config.RISK_COLORS[pred["risk"]]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred["probability"] * 100,
                number={"suffix": "%", "font": {"size": 40, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40 * thr * 100], "color": "rgba(52,211,153,.18)"},
                        {"range": [40 * thr * 100, 70 * thr * 100], "color": "rgba(251,191,36,.18)"},
                        {"range": [70 * thr * 100, thr * 100], "color": "rgba(251,146,60,.18)"},
                        {"range": [thr * 100, 100], "color": "rgba(248,113,113,.20)"},
                    ],
                },
            )
        )
        fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font={"color": config.THEME["text"]})
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"<div class='fg-card' style='text-align:center'>{risk_badge_html(pred['risk'])}"
            f"<p style='margin:10px 0 0'>Issue {'predicted' if pred['issue_predicted'] else 'not predicted'}"
            f" &nbsp;|&nbsp; threshold {pred['threshold']:.2f}<br>"
            f"<span style='color:#9db4d6'>model: {pred['model']}</span></p></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Probable contributing factors")
        st.caption(
            "Interpretable contributions from the model - treat as probable factors, "
            "not confirmed root causes."
        )
        contribs = pred["contributions"][:8]
        max_abs = max(abs(c["contribution"]) for c in contribs) or 1e-9
        for c in contribs:
            width = max(int(abs(c["contribution"]) / max_abs * 60), 1)
            bar = ("#" * width).ljust(60)
            sign = "+" if c["contribution"] >= 0 else "-"
            st.markdown(
                f"`{bar[:width].ljust(3)}` **{c['feature']}** = {c['value']} &nbsp; "
                f"{sign}{abs(c['contribution']):.3f} {'raises risk' if c['contribution'] > 0 else 'lowers risk'}"
            )

    st.markdown("#### Recommended actions")
    rec_cols = st.columns(len(pred["recommendations"][:3]) or 1)
    for i, rec in enumerate(pred["recommendations"][:3]):
        with rec_cols[i]:
            st.markdown(
                f"<div class='fg-card'><b>{rec['title']}</b>"
                f"<br><span class='fg-badge' style='background:#22d3ee22;color:#22d3ee;"
                f"border:1px solid #22d3ee55'>{rec['priority']}</span>"
                f"<p>{rec['action']}</p>"
                f"<span style='color:#9db4d6;font-size:.85rem'>{rec['evidence']}</span></div>",
                unsafe_allow_html=True,
            )
    with st.expander("All recommendations and validation notes"):
        for rec in pred["recommendations"][3:]:
            st.markdown(f"- **{rec['title']}** ({rec['priority']}): {rec['action']}")
        if pred["warnings"]:
            st.warning("Input validation: " + " | ".join(pred["warnings"]))
        else:
            st.success("All inputs are within the ranges seen during training.")
else:
    st.info("Set the order details and press **Run FoodGuard analysis**.")
