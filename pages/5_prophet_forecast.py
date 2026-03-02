import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prophet Forecast", page_icon="🔮", layout="wide")

@st.cache_data
def load_weekly():
    df = pd.read_csv("data/weekly_close.csv", parse_dates=["Datetime"])
    df.columns = ["ds", "y"]
    return df

weekly_df = load_weekly()

st.title("🔮 Prophet Forecasting Playground")
st.markdown("Tune hyperparameters and see forecast accuracy update in real time.")

# ── Sidebar controls ──
st.sidebar.header("Prophet Parameters")
changepoint_prior = st.sidebar.slider("changepoint_prior_scale",
                                       0.001, 0.5, 0.05, step=0.01,
                                       help="Higher = more flexible trend. Lower = smoother.")
n_changepoints = st.sidebar.slider("n_changepoints", 5, 50, 25,
                                    help="Number of potential changepoint locations.")
seasonality_mode = st.sidebar.radio("seasonality_mode",
                                     ["multiplicative", "additive"],
                                     help="Multiplicative is better when seasonal swings scale with the level.")
yearly_seasonality = st.sidebar.toggle("Yearly seasonality", value=True)
add_halving = st.sidebar.toggle("Add 4-year halving cycle", value=False,
                                 help="Custom Fourier seasonality with period ≈ 208.7 weeks.")
holdout_weeks = st.sidebar.slider("Test holdout (weeks)", 12, 104, 52)

# ── Train / Predict ──
@st.cache_data(show_spinner="Training Prophet...")
def train_prophet(cp_prior, n_cp, s_mode, yearly, halving, holdout):
    cutoff = weekly_df["ds"].max() - pd.Timedelta(weeks=holdout)
    train = weekly_df[weekly_df["ds"] <= cutoff]
    test = weekly_df[weekly_df["ds"] > cutoff]

    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cp_prior,
        n_changepoints=n_cp,
        seasonality_mode=s_mode,
    )
    if halving:
        m.add_seasonality(name="halving_cycle", period=365.25 * 4 / 7, fourier_order=3)

    m.fit(train)
    future = m.make_future_dataframe(periods=holdout, freq="W")
    forecast = m.predict(future)

    merged = test.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
    mae = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))

    return forecast, merged, mae, rmse, cutoff, m

if st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True):
    forecast, merged, mae, rmse, cutoff, model = train_prophet(
        changepoint_prior, n_changepoints, seasonality_mode,
        yearly_seasonality, add_halving, holdout_weeks
    )
    st.session_state["result"] = (forecast, merged, mae, rmse, cutoff, model)

# ── Display ──
if "result" in st.session_state:
    forecast, merged, mae, rmse, cutoff, model = st.session_state["result"]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:,.0f}")
    col2.metric("RMSE", f"${rmse:,.0f}")
    pct_error = mae / merged["y"].mean() * 100
    col3.metric("MAE / Mean Price", f"{pct_error:.1f}%")

    # Forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["ds"], y=weekly_df["y"],
                             mode="lines", name="Actual",
                             line=dict(color="white", width=1)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             mode="lines", name="Forecast",
                             line=dict(color="#F7931A", width=1.5)))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(247,147,26,0.15)",
        line=dict(width=0), name="Confidence Interval",
    ))
    fig.add_vline(x=cutoff.timestamp() * 1000, line_dash="dash", line_color="red",
                  annotation_text="Train/Test Cutoff")

    fig.update_layout(
        title="BTC Weekly Price — Prophet Forecast",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Components
    st.subheader("Prophet Components")
    comp_fig = model.plot_components(forecast)
    st.pyplot(comp_fig)

    # Test period zoom
    with st.expander("🔍 Test Period Detail"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=merged["ds"], y=merged["y"],
                                  mode="lines+markers", name="Actual",
                                  line=dict(color="white")))
        fig2.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"],
                                  mode="lines+markers", name="Predicted",
                                  line=dict(color="#F7931A")))
        fig2.update_layout(template="plotly_dark", height=400,
                           title="Actual vs Predicted (Test Set)")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Adjust parameters in the sidebar and click **Train Model** to see results.")
    st.markdown("""
    **Suggested experiments:**
    - Start with defaults (changepoint_prior=0.05, 25 changepoints, multiplicative)
    - Increase changepoint_prior to 0.3 — watch how the forecast becomes more reactive
    - Toggle the 4-year halving cycle and see if it improves MAE
    - Try additive vs multiplicative seasonality
    """)