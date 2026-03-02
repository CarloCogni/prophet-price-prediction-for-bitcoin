import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Halving Cycles", page_icon="₿", layout="wide")

@st.cache_data
def load_weekly():
    return pd.read_csv("data/weekly_close.csv", parse_dates=["Datetime"], index_col="Datetime")

weekly = load_weekly()

st.title("₿ Bitcoin Halving Cycle Analysis")
st.markdown("""
Every ~4 years the Bitcoin block reward is halved, reducing new supply issuance by 50%.
Historically, each halving has preceded a major bull run. This page lets you overlay
and compare cycles interactively.
""")

HALVINGS = {
    "H1 — Nov 2012": pd.Timestamp("2012-11-28"),
    "H2 — Jul 2016": pd.Timestamp("2016-07-09"),
    "H3 — May 2020": pd.Timestamp("2020-05-11"),
    "H4 — Apr 2024": pd.Timestamp("2024-04-19"),
}
COLORS = {"H1 — Nov 2012": "#3498db", "H2 — Jul 2016": "#e67e22",
          "H3 — May 2020": "#2ecc71", "H4 — Apr 2024": "#e74c3c"}

# ── Controls ──
col1, col2 = st.columns([2, 1])
with col1:
    selected = st.multiselect("Select cycles to compare",
                              list(HALVINGS.keys()),
                              default=list(HALVINGS.keys()))
with col2:
    log_scale = st.toggle("Log scale", value=True)

tab1, tab2 = st.tabs(["🔀 Cycle Overlay (Indexed)", "📈 Full Timeline"])

# ── Tab 1: Overlay ──
with tab1:
    fig = go.Figure()
    for label in selected:
        h_date = HALVINGS[label]
        idx = list(HALVINGS.values()).index(h_date)
        next_date = list(HALVINGS.values())[idx + 1] if idx < len(HALVINGS) - 1 else weekly.index[-1]

        cycle = weekly.loc[h_date:next_date, "Close"].dropna()
        if len(cycle) == 0:
            continue

        normalized = cycle / cycle.iloc[0] * 100
        weeks = list(range(len(normalized)))
        dash = "dash" if "H4" in label else "solid"

        fig.add_trace(go.Scatter(
            x=weeks, y=normalized.values,
            mode="lines", name=label,
            line=dict(color=COLORS[label], width=2, dash=dash),
            hovertemplate="Week %{x}<br>Indexed: %{y:.1f}<extra>" + label + "</extra>"
        ))

    yaxis_type = "log" if log_scale else "linear"
    fig.update_layout(
        title="BTC Halving Cycles — Price Indexed to 100 at Halving Date",
        xaxis_title="Weeks Since Halving",
        yaxis_title="Indexed Price (100 = halving date)",
        yaxis_type=yaxis_type,
        template="plotly_dark", height=550,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key patterns:** Each cycle shows an initial slow accumulation phase (6-12 months),
    followed by a parabolic blow-off top, and then a ~80% crash.
    H4 (current, dashed) is still unfolding — compare its trajectory to prior cycles
    to form a view on where we might be in the current epoch.
    """)

# ── Tab 2: Full timeline ──
with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weekly.index, y=weekly["Close"],
                              mode="lines", name="BTC Close",
                              line=dict(color="#F7931A", width=1)))

    for label in selected:
        h_date = HALVINGS[label]
        fig2.add_vline(x=h_date.timestamp() * 1000, line_dash="dash", line_color=COLORS[label],
                       annotation_text=label, annotation_position="top left",
                       annotation_font_color=COLORS[label])

    ytype = "log" if log_scale else "linear"
    fig2.update_layout(
        title="BTC Weekly Close with Halving Events",
        xaxis_title="Date", yaxis_title="Price (USD)",
        yaxis_type=ytype,
        template="plotly_dark", height=500,
    )
    st.plotly_chart(fig2, use_container_width=True)