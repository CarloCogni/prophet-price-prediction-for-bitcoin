# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational time series analysis project (MAICEN1125 M5U1 course assignment) that analyzes Bitcoin historical price data. The primary artifact is a single Jupyter notebook designed to run in Google Colab.

## Running the Notebook

**Google Colab (primary target environment):**
- Open `notebooks/MAICEN1125_M5U1_Time_Series_bonus_track_bitcoin.ipynb` in Colab
- Execute cells sequentially — the notebook self-installs dependencies via `pip install`

**Locally:**
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows/Git Bash

# Install dependencies (not declared in pyproject.toml — managed in-notebook)
pip install prophet kagglehub pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter

# Launch notebook
jupyter notebook notebooks/MAICEN1125_M5U1_Time_Series_bonus_track_bitcoin.ipynb
```

**Kaggle credentials** are required for `kagglehub` to download the dataset (`mczielinski/bitcoin-historical-data`). Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables or place `~/.kaggle/kaggle.json`.

## Architecture

The project consists of a single end-to-end analysis notebook with six sequential exercises:

1. **Data Cleaning** — Downloads 1-minute BTC/USD OHLCV data from Kaggle, deduplicates timestamps (averaging), resamples to hourly via linear interpolation (~115K observations, 2018–2025)
2. **Multi-Scale Visualization** — Plots at day, week, and full-history scales
3. **Seasonality Analysis** — Hourly-of-day and day-of-week patterns using normalized returns (to remove trend bias)
4. **ACF/PACF Diagnostics** — Autocorrelation analysis at hourly (48 lags) and daily (30 lags) resolutions using `statsmodels`
5. **Prophet Forecasting** — Weekly resampled data, 80/20 train/test split (last 52 weeks as test), multiplicative seasonality; compares baseline vs. flexible changepoint model using MAE/RMSE
6. **SARIMA (Bonus)** — ADF stationarity test, SARIMA(1,1,1)(1,0,1,52) with yearly seasonality, fallback to ARIMA(2,1,2) if computation fails; comparison with Prophet

## Key Data Flow

```
Kaggle (btcusd_1-min_data.csv)
  → deduplicate & resample → hourly DataFrame (Close price)
    → Exercise 2-4: visualization & diagnostics
    → Exercise 5: weekly resample → Prophet model
    → Exercise 6: weekly resample → SARIMA model
```

## Python Version

Requires Python ≥ 3.13 (per `pyproject.toml`). The `.venv/` virtual environment is included but dependencies are not tracked in `pyproject.toml` — they are installed inline in the notebook.
