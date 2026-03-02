import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

@st.cache_data
def load_hourly():
    return pd.read_csv("data/hourly_close.csv", parse_dates=["Datetime"], index_col="Datetime")

df = load_hourly()

st.title("📊 Interactive Data Explorer")
st.markdown("Explore BTC price at **any** date, week, or custom range.")

# ── Controls ──
view_mode = st.radio("View mode", ["Single Day", "Single Week", "Custom Range", "Full Dataset"],
                     horizontal=True)

if view_mode == "Single Day":
    date = st.date_input("Pick a date",
                         value=pd.Timestamp("2023-06-15"),
                         min_value=df.index.min().date(),
                         max_value=df.index.max().date())
    mask = df.index.date == date
    data = df[mask]
    title = f"BTC Price — {date}"

elif view_mode == "Single Week":
    date = st.date_input("Pick any day in the week",
                         value=pd.Timestamp("2023-06-15"),
                         min_value=df.index.min().date(),
                         max_value=df.index.max().date())
    week_start = pd.Timestamp(date) - timedelta(days=pd.Timestamp(date).dayofweek)
    week_end = week_start + timedelta(days=6, hours=23)
    data = df.loc[str(week_start):str(week_end)]
    title = f"BTC Price — Week of {week_start.date()}"

elif view_mode == "Custom Range":
    col1, col2 = st.columns(2)
    start = col1.date_input("Start", value=pd.Timestamp("2024-01-01"),
                            min_value=df.index.min().date(),
                            max_value=df.index.max().date())
    end = col2.date_input("End", value=pd.Timestamp("2024-12-31"),
                          min_value=df.index.min().date(),
                          max_value=df.index.max().date())
    data = df.loc[str(start):str(end)]
    title = f"BTC Price — {start} to {end}"

else:  # Full Dataset
    data = df
    title = "BTC Price — Full Historical Dataset"

# ── Plot ──
if len(data) == 0:
    st.warning("No data for the selected range.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"],
                             mode="lines", name="Close",
                             line=dict(color="#F7931A", width=1.2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price (USD)",
                      template="plotly_dark", height=500,
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open", f"${data['Close'].iloc[0]:,.2f}")
    c2.metric("Close", f"${data['Close'].iloc[-1]:,.2f}")
    c3.metric("High", f"${data['Close'].max():,.2f}")
    c4.metric("Low", f"${data['Close'].min():,.2f}")

st.markdown("""
---
**Commentary:** At the single-day level, intraday micro-structure and volatility clusters
are visible. The weekly view reveals multi-day trends and recoveries. The full dataset
shows the dominant macro pattern: exponential growth punctuated by ~80% drawdowns,
driven by the 4-year halving cycle.
""")