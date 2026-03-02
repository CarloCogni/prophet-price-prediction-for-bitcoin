import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="BTC Time Series Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared data loader (cached across all pages) ──
@st.cache_data
def load_hourly():
    df = pd.read_csv("data/hourly_close.csv", parse_dates=["Datetime"], index_col="Datetime")
    return df

@st.cache_data
def load_daily():
    df = pd.read_csv("data/daily_close.csv", parse_dates=["Datetime"], index_col="Datetime")
    return df

@st.cache_data
def load_weekly():
    df = pd.read_csv("data/weekly_close.csv", parse_dates=["Datetime"], index_col="Datetime")
    return df

# ── Landing page ──
st.title("₿ Bitcoin Time Series Analysis")
st.markdown("""
**MSc in AI for Architecture & Construction — M5U1 Group Assignment**

This app presents an interactive exploration of Bitcoin historical price data,
adapted from the PJMW energy-consumption assignment to BTC/USD.
The analysis covers **data cleaning, multi-scale visualization, seasonality,
autocorrelation, and Prophet-based forecasting** — with a focus on the unique
4-year halving cycle that drives Bitcoin's price dynamics.

---

### 📑 Pages

| Page | Description |
|------|-------------|
| **📊 Data Explorer** | Navigate BTC price at any date/week/range interactively |
| **🔄 Seasonality** | Hourly, weekly, and returns-based seasonal patterns + STL decomposition |
| **₿ Halving Cycles** | Overlay and compare the four halving epochs side-by-side |
| **📈 Statistical Analysis** | ACF / PACF correlograms at hourly and daily resolution |
| **🔮 Prophet Forecast** | Tune Prophet hyperparameters live and see MAE/RMSE update in real time |

---
""")

# Quick stats
try:
    weekly = load_weekly()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Start", str(weekly.index.min().date()))
    col2.metric("Dataset End", str(weekly.index.max().date()))
    col3.metric("Latest Close", f"${weekly['Close'].iloc[-1]:,.0f}")
    col4.metric("All-Time High", f"${weekly['Close'].max():,.0f}")
except Exception:
    st.info("Run `python prepare_data.py` first to generate the data files.")

st.caption("Data: Kaggle `mczielinski/bitcoin-historical-data` · Built with Streamlit & Prophet")