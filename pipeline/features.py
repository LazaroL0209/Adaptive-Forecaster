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

        df["realized_vol_10"] = df["log_return"].rolling(10).std().shift(1)
        df["volume_trend"]    = (
            df["volume"].rolling(5).mean() /
            (df["volume"].rolling(20).mean() + 1e-9)
        ).shift(1)

        df["target"] = df["log_return"].shift(-1)
        df["ticker"] = ticker
        df = df.dropna()
        records.append(df)

    features = pd.concat(records)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    feature_cols = [c for c in features.columns
                    if c not in ["target", "ticker", "close", "log_return"]]

    pca_records = []
    for ticker in features["ticker"].unique():
        t_df  = features[features["ticker"] == ticker].copy()
        valid = t_df[feature_cols].dropna()
        if len(valid) < 10:
            pca_records.append(t_df)
            continue

        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(valid)
        n_comp    = min(10, scaled.shape[1])
        pca       = PCA(n_components=n_comp)
        pcs       = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_.cumsum()
        print(f"{ticker} PCA: {n_comp} components explain {explained[-1]:.1%} of variance")

        for i in range(n_comp):
            t_df.loc[valid.index, f"pc_{i+1}"] = pcs[:, i]

        regime_features = ["realized_vol_10", "roll_mean_20", "roll_std_20"]
        regime_valid    = t_df[regime_features].dropna()
        if len(regime_valid) >= 3:
            reg_scaled = StandardScaler().fit_transform(regime_valid)
            kmeans     = KMeans(n_clusters=3, random_state=42, n_init=10)
            regimes    = kmeans.fit_predict(reg_scaled)
            t_df.loc[regime_valid.index, "market_regime"] = regimes.astype(float)

        pca_records.append(t_df)

    features = pd.concat(pca_records)
    features = features.dropna()
    features.to_parquet("data/features.parquet")
    print(f"Feature engineering complete: {features.shape}")
    return features

if __name__ == "__main__":
    close  = pd.read_parquet("data/close_prices.parquet")
    volume = pd.read_parquet("data/volume.parquet")
    compute_features(close, volume)