import pandas as pd
import numpy as np

def compute_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    records = []

    for ticker in close.columns:
        df = pd.DataFrame({
            "close":  close[ticker],
            "volume": volume[ticker],
        }).dropna()

        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        for w in [5, 10, 20]:
            df[f"roll_mean_{w}"] = df["log_return"].rolling(w).mean().shift(1)
            df[f"roll_std_{w}"]  = df["log_return"].rolling(w).std().shift(1)

        for lag in range(1, 6):
            df[f"lag_{lag}"] = df["log_return"].shift(lag)

        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["norm_volume"] = (
            (df["volume"] - df["volume"].rolling(20).mean()) /
            (df["volume"].rolling(20).std() + 1e-9)
        ).shift(1)

        df["target"] = df["log_return"].shift(-1)
        df["ticker"] = ticker
        df = df.dropna()
        records.append(df)

    features = pd.concat(records)
    features.to_parquet("data/features.parquet")
    print(f"Feature engineering complete: {features.shape}")
    return features

if __name__ == "__main__":
    close  = pd.read_parquet("data/close_prices.parquet")
    volume = pd.read_parquet("data/volume.parquet")
    compute_features(close, volume)