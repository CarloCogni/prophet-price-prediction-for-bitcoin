# ₿ Bitcoin Time Series Analysis

An end-to-end time series analysis of Bitcoin historical price data, from data cleaning and seasonality exploration to 
Prophet-based forecasting. The project delivers both a **Jupyter notebook** and **interactive Streamlit web app**.

---

## Live App

**[prophet-price-prediction-for-bitcoin.streamlit.app](https://prophet-price-prediction-for-bitcoin.streamlit.app)**

---

## App Pages

| Page | Description |
|------|-------------|
| **Home** | Dataset summary stats and navigation overview |
| **📊 Data Explorer** | Browse BTC price at any date, week, or custom range interactively |
| **🔄 Seasonality** | Price-based and returns-based (de-trended) hourly/weekly patterns + STL decomposition |
| **₿ Halving Cycles** | Overlay and compare the four 4-year halving epochs, indexed and on full timeline |
| **📈 Statistical Analysis** | ACF / PACF correlograms at configurable hourly and daily lag counts |
| **🔮 Prophet Forecast** | Tune Prophet hyperparameters live (changepoints, seasonality mode, halving cycle) and inspect MAE/RMSE in real time |

---

## Analysis Overview

The analysis follows six sequential exercises:

1. **Data Cleaning** — Downloads 1-minute BTC/USD OHLCV data from Kaggle (`mczielinski/bitcoin-historical-data`), deduplicates timestamps (averaging), resamples to hourly via linear interpolation (~115 K observations, 2018–2025).
2. **Multi-Scale Visualization** — Plots price at day, week, and full-history scales.
3. **Seasonality Analysis** — Hourly-of-day and day-of-week patterns using normalised returns to remove trend bias; STL decomposition (period = 52 weeks).
4. **ACF / PACF Diagnostics** — Autocorrelation analysis at hourly (48 lags) and daily (30 lags) resolution confirming non-stationarity and AR(1)-like behaviour.
5. **Prophet Forecasting** — Weekly resampled data, 80/20 train/test split (last 52 weeks as test), multiplicative seasonality, optional 4-year halving-cycle Fourier term; compares baseline vs. flexible changepoint model using MAE/RMSE.
6. **SARIMA (Bonus)** — ADF stationarity test, SARIMA(1,1,1)(1,0,1,52) with yearly seasonality, fallback to ARIMA(2,1,2); comparison with Prophet (notebook only).

### Data Flow

```
Kaggle (btcusd_1-min_data.csv)
  └─ prepare_data.py ──► data/hourly_close.csv   ──► Pages 1, 2, 4
                     ──► data/daily_close.csv    ──► Page 4
                     ──► data/weekly_close.csv   ──► Pages 3, 5
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data | `pandas`, `numpy`, `kagglehub` |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |
| Statistics | `statsmodels` (ACF/PACF, STL, SARIMA, ADF) |
| Forecasting | `prophet`, `scikit-learn` (metrics) |
| App | `streamlit` |
| Packaging | `uv` / `pyproject.toml` |

---

## Local Setup

### Prerequisites

- Python ≥ 3.13
- [Kaggle API credentials](https://www.kaggle.com/docs/api#authentication) — `~/.kaggle/kaggle.json` **or** environment variables `KAGGLE_USERNAME` / `KAGGLE_KEY`

### Install & Run

```bash
# 1. Clone the repository
git clone https://github.com/CarloCogni/prophet-price-prediction-for-bitcoin.git
cd prophet-price-prediction-for-bitcoin

# 2. Create and activate virtual environment (uv recommended)
uv venv
source .venv/Scripts/activate   # Windows / Git Bash
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Generate the pre-processed CSV files (run once)
python prepare_data.py

# 5. Launch the Streamlit app
streamlit run app.py
```

> **Note:** `prepare_data.py` downloads the Kaggle dataset (~500 MB) and writes three lightweight CSV files to `data/`. The CSVs are committed to the repository so this step is only needed if you want to refresh the data.

### Jupyter Notebook (Google Colab)

The original assignment notebook runs standalone in Google Colab without any local setup:

1. Open `notebooks/MAICEN1125_M5U1_Time_Series_bonus_track_bitcoin.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Execute cells sequentially — the notebook self-installs all dependencies via `pip install`.
3. Kaggle credentials are required for the data download cell (see above).

---

## Project Structure

```
.
├── app.py                      # Streamlit entry point (Home page)
├── pages/
│   ├── 1_data_explorer.py      # Interactive price browser
│   ├── 2_seasonality.py        # Seasonal patterns + STL
│   ├── 3_halving_cycles.py     # Halving epoch overlay
│   ├── 4_statistical_analysis.py  # ACF / PACF
│   └── 5_prophet_forecast.py   # Prophet playground
├── data/
│   ├── hourly_close.csv        # ~115 K hourly rows
│   ├── daily_close.csv         # ~2 600 daily rows
│   └── weekly_close.csv        # ~370 weekly rows
├── notebooks/
│   └── MAICEN1125_M5U1_Time_Series_bonus_track_bitcoin.ipynb
├── prepare_data.py             # One-time data pipeline (Kaggle → CSVs)
├── pyproject.toml
└── requirements.txt
```

---

## Data Source

Kaggle dataset: [`mczielinski/bitcoin-historical-data`](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
1-minute BTC/USD OHLCV candles from 2011 to 2025.

---

## License

[GPL-3.0](LICENSE)
