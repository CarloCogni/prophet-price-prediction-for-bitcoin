import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Seasonality", page_icon="🔄", layout="wide")

@st.cache_data
def load_hourly():
    return pd.read_csv("data/hourly_close.csv", parse_dates=["Datetime"], index_col="Datetime")

@st.cache_data
def load_weekly():
    return pd.read_csv("data/weekly_close.csv", parse_dates=["Datetime"], index_col="Datetime")

df = load_hourly()

st.title("🔄 Seasonality Analysis")

tab1, tab2, tab3 = st.tabs(["📈 Price-Based", "📊 Returns-Based (De-trended)", "🔬 STL Decomposition"])

# ── Tab 1: raw price seasonality ──
with tab1:
    st.subheader("Average BTC Price by Hour / Day of Week")
    st.caption("⚠️ Raw price averages are skewed toward recent high-price years. See the Returns tab for a de-trended view.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Avg Price by Hour (UTC)", "Avg Price by Day of Week"])

    hourly_avg = df.groupby(df.index.hour)["Close"].mean()
    fig.add_trace(go.Scatter(x=list(range(24)), y=hourly_avg.values,
                             mode="lines+markers", marker=dict(color="#F7931A"),
                             line=dict(color="#F7931A")), row=1, col=1)

    daily_means = df.resample("D").mean()
    weekly_avg = daily_means.groupby(daily_means.index.dayofweek)["Close"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.add_trace(go.Bar(x=day_names, y=weekly_avg.values,
                         marker_color="#F7931A"), row=1, col=2)

    fig.update_layout(template="plotly_dark", height=400, showlegend=False)
    fig.update_xaxes(title_text="Hour (UTC)", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_yaxes(title_text="Avg Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Price (USD)", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: returns-based ──
with tab2:
    st.subheader("Average Hourly Return (%) — Scale-Invariant")
    st.markdown("Percentage returns remove the price-level bias, revealing genuine intraday/intraweek patterns.")

    returns = df["Close"].pct_change()

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Avg Hourly Return by Hour (UTC)",
                                                          "Avg Daily Return by Day of Week"])

    hourly_ret = returns.groupby(df.index.hour).mean() * 100
    colors_h = ["#2ecc71" if v >= 0 else "#e74c3c" for v in hourly_ret.values]
    fig2.add_trace(go.Bar(x=list(range(24)), y=hourly_ret.values,
                          marker_color=colors_h), row=1, col=1)
    fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)

    weekly_ret = returns.groupby(df.index.dayofweek).mean() * 100
    colors_w = ["#2ecc71" if v >= 0 else "#e74c3c" for v in weekly_ret.values]
    fig2.add_trace(go.Bar(x=day_names, y=weekly_ret.values,
                          marker_color=colors_w), row=1, col=2)
    fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=2)

    fig2.update_layout(template="plotly_dark", height=400, showlegend=False)
    fig2.update_xaxes(title_text="Hour (UTC)", row=1, col=1)
    fig2.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig2.update_yaxes(title_text="Mean Return (%)", row=1, col=1)
    fig2.update_yaxes(title_text="Mean Return (%)", row=1, col=2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Observations:** Returns tend to be slightly negative around hours 0 and 8-9 UTC
    (US late night / Asian morning), and positive during US/EU active trading hours.
    Early weekdays (Mon-Wed) show stronger positive mean returns than weekends.
    """)

# ── Tab 3: STL ──
with tab3:
    st.subheader("STL Decomposition (Weekly, period=52)")
    st.markdown("Seasonal-Trend decomposition using Loess separates the weekly series into Trend, Seasonal (yearly cycle), and Residual.")

    from statsmodels.tsa.seasonal import STL

    weekly = load_weekly()
    stl = STL(weekly["Close"], period=52, robust=True)
    result = stl.fit()

    components = {
        "Observed": weekly["Close"],
        "Trend": result.trend,
        "Seasonal": result.seasonal,
        "Residual": result.resid,
    }

    fig3 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         subplot_titles=list(components.keys()),
                         vertical_spacing=0.06)

    for i, (name, series) in enumerate(components.items(), 1):
        fig3.add_trace(go.Scatter(x=series.index, y=series.values,
                                  mode="lines", line=dict(width=1, color="#F7931A"),
                                  name=name), row=i, col=1)

    fig3.update_layout(template="plotly_dark", height=700, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **Interpretation:** The trend component captures BTC's long-run exponential growth.
    The seasonal component (period=52) shows a weak but present yearly cycle.
    The large residuals during 2021 and 2024-25 reflect the extreme bull-run volatility
    that cannot be explained by trend or yearly seasonality alone — this is where the
    4-year halving cycle (see next page) becomes the dominant explanatory factor.
    """)