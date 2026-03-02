import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

st.set_page_config(page_title="ACF / PACF", page_icon="📈", layout="wide")

@st.cache_data
def load_hourly():
    return pd.read_csv("data/hourly_close.csv", parse_dates=["Datetime"], index_col="Datetime")

@st.cache_data
def load_daily():
    return pd.read_csv("data/daily_close.csv", parse_dates=["Datetime"], index_col="Datetime")

hourly = load_hourly()
daily = load_daily()

st.title("📈 Statistical Analysis — ACF / PACF")

n_lags_hourly = st.slider("Hourly lags", 12, 168, 48)
n_lags_daily = st.slider("Daily lags", 7, 90, 30)

def plot_acf_pacf(series, nlags, title_prefix):
    """Return a Plotly figure with ACF and PACF side by side."""
    acf_vals = acf(series.dropna(), nlags=nlags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=nlags, method="ywm")

    n = len(series.dropna())
    ci = 1.96 / np.sqrt(n)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"ACF — {title_prefix}", f"PACF — {title_prefix}"])

    for col_idx, (vals, name) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], 1):
        lags = list(range(len(vals)))
        fig.add_trace(go.Bar(x=lags, y=vals, marker_color="#F7931A",
                             opacity=0.8, name=name), row=1, col=col_idx)
        fig.add_hline(y=ci, line_dash="dash", line_color="cyan", opacity=0.5, row=1, col=col_idx)
        fig.add_hline(y=-ci, line_dash="dash", line_color="cyan", opacity=0.5, row=1, col=col_idx)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=1, col=col_idx)

    fig.update_layout(template="plotly_dark", height=350, showlegend=False)
    return fig

st.subheader("Hourly BTC Price")
st.plotly_chart(plot_acf_pacf(hourly["Close"], n_lags_hourly, f"Hourly ({n_lags_hourly} lags)"),
                use_container_width=True)

st.subheader("Daily BTC Price")
st.plotly_chart(plot_acf_pacf(daily["Close"], n_lags_daily, f"Daily ({n_lags_daily} lags)"),
                use_container_width=True)

st.markdown("""
---
**Interpretation:**

The **Hourly ACF** shows extremely high, slowly decaying autocorrelation across all lags — each bar
is near 1.0. This confirms BTC price is strongly non-stationary (a random walk with drift).
The PACF drops sharply after lag 1, suggesting that once you condition on the immediately
preceding hour, further lags add minimal new information — classic AR(1)-like behavior.

The **Daily ACF** shows the same persistent pattern. This high autocorrelation is why
differencing (d=1) is required before fitting ARIMA-family models, and why Prophet
(which estimates a flexible trend component) is well-suited to this type of data.
""")