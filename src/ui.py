"""Shared Streamlit theming helpers (navy + cyan, subtle glass cards)."""
from __future__ import annotations

import config

CSS = """
<style>
.stApp {
  background:
    radial-gradient(1100px 520px at 85% -10%, rgba(34,211,238,0.14), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(14,165,233,0.10), transparent 55%),
    linear-gradient(160deg, #0a1628 0%, #0b1e3a 55%, #0a1628 100%);
  color: #e6f1ff;
}
h1, h2, h3, h4 { color: #e6f1ff; letter-spacing: .2px; }
p, li, span, div[data-testid="stMarkdownContainer"] { color: #dbe7ff; }
[data-testid="stMetric"] {
  background: rgba(17,34,64,.55);
  border: 1px solid rgba(56,189,248,.25);
  border-radius: 14px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] p { color: #9fd8e8 !important; }
.fg-card {
  background: rgba(17,34,64,.55);
  border: 1px solid rgba(56,189,248,.25);
  border-radius: 16px; padding: 18px 22px; margin-bottom: 14px;
}
.fg-badge {
  display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: .75rem; font-weight: 700; letter-spacing: .6px;
}
.fg-hero-title {
  font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: 0;
  background: linear-gradient(90deg, #7dd3fc, #22d3ee 55%, #0ea5e9);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.fg-sub { color: #9db4d6; font-size: 1.02rem; margin-top: 6px; }
.stButton > button {
  background: linear-gradient(90deg, #0ea5e9, #22d3ee);
  color: #06121f; font-weight: 700; border: none; border-radius: 10px;
  padding: .45rem 1.2rem;
}
.stButton > button:hover { filter: brightness(1.1); }
section[data-testid="stSidebar"] {
  background: #081120; border-right: 1px solid rgba(56,189,248,.18);
}
section[data-testid="stSidebar"] * { color: #cfe3ff !important; }
hr { border-color: rgba(56,189,248,.18); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(17,34,64,.5); border-radius: 10px 10px 0 0;
  color: #9db4d6 !important; border: 1px solid rgba(56,189,248,.15);
}
.stTabs [aria-selected="true"] { color: #22d3ee !important; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def apply_plotly_theme(fig, height: int = 380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=config.THEME["text"], size=13),
        height=height,
        margin=dict(l=10, r=10, t=52, b=10),
        title_font=dict(color=config.THEME["text"], size=16),
        legend=dict(font=dict(color=config.THEME["muted"])),
    )
    fig.update_xaxes(gridcolor=config.THEME["grid"], zerolinecolor=config.THEME["grid"])
    fig.update_yaxes(gridcolor=config.THEME["grid"], zerolinecolor=config.THEME["grid"])
    return fig


def risk_badge_html(risk: str) -> str:
    color = config.RISK_COLORS.get(risk, "#94a3b8")
    return (f"<span class='fg-badge' style='background:{color}22;color:{color};"
            f"border:1px solid {color}66'>{risk.upper()} RISK</span>")


def page_header(title: str, subtitle: str = ""):
    import streamlit as st
    st.markdown(
        f"<h2 style='margin-bottom:0'>{title}</h2>"
        f"<p class='fg-sub'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
