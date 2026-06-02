import yfinance as yf
import pandas as pd
import os

TICKERS = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "XOM", "JPM",
           "AMZN", "NVDA", "BRK-B", "V", "UNH", "JNJ", "WMT",
           "PG", "MA", "HD", "DIS", "BAC"]

def fetch_equity_data(start="2014-01-01", end=None):
    print("Fetching equity data from Yahoo Finance...")
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True)
    close = raw["Close"]
    volume = raw["Volume"]
    close.to_parquet("data/close_prices.parquet")
    volume.to_parquet("data/volume.parquet")
    print(f"Saved {len(close)} rows for {len(TICKERS)} tickers.")
    return close, volume

def fetch_macro_data():
    try:
        import time
        from fredapi import Fred
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            print("Warning: FRED_API_KEY not set, skipping macro data.")
            return None
        fred = Fred(api_key=api_key)
        series = {}
        calls = [
            ("FEDFUNDS", "Federal Funds Rate"),
            ("VIXCLS",   "VIX"),
            ("CPIAUCSL", "CPI"),
        ]
        for series_id, name in calls:
            print(f"Fetching {name}...")
            try:
                series[series_id] = fred.get_series(
                    series_id, observation_start="2014-01-01")
                time.sleep(2)
            except Exception as e:
                print(f"Failed to fetch {name}: {e}")
                time.sleep(10)
                series[series_id] = fred.get_series(
                    series_id, observation_start="2014-01-01")
        macro = pd.DataFrame(series)
        macro = macro.ffill()
        macro.to_parquet("data/macro.parquet")
        print("Saved macro data.")
        return macro
    except Exception as e:
        print(f"Macro fetch failed: {e}")
        return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    fetch_equity_data()
    fetch_macro_data()