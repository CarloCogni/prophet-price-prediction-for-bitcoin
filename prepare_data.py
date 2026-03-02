"""
prepare_data.py — Run ONCE locally before deploying.
Generates three CSVs in data/ from the Kaggle BTC dataset.
"""
import os, sys
import pandas as pd

try:
    import kagglehub
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
except Exception as e:
    print(f"kagglehub failed: {e}")
    sys.exit(1)

raw_df = pd.read_csv(os.path.join(path, "btcusd_1-min_data.csv"))
print(f"Raw shape: {raw_df.shape}")

df = raw_df.copy()
df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")
df = df.set_index("Datetime").sort_index()
df = df.groupby(df.index).mean()
df = df.asfreq("h")
df["Close"] = df["Close"].interpolate(method="linear")
df = df.dropna(subset=["Close"])
print(f"Cleaned: {df.shape[0]} hourly rows, {df.index.min()} -> {df.index.max()}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(out, exist_ok=True)

hourly = df[["Close"]].copy()
hourly.index.name = "Datetime"
hourly.to_csv(os.path.join(out, "hourly_close.csv"))

daily = df["Close"].resample("D").mean().dropna()
daily.name = "Close"
daily.index.name = "Datetime"
daily.to_csv(os.path.join(out, "daily_close.csv"))

weekly = df["Close"].resample("W").mean().dropna()
weekly.name = "Close"
weekly.index.name = "Datetime"
weekly.to_csv(os.path.join(out, "weekly_close.csv"))

print(f"Wrote: hourly ({len(hourly)}), daily ({len(daily)}), weekly ({len(weekly)}) rows")
print("Done!")
