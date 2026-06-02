import pandas as pd
import numpy as np
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.train_loop import make_sequences, TimeSeriesDataset
import torch
from torch.utils.data import DataLoader

df = pd.read_parquet("data/features.parquet")
ticker_df = df[df["ticker"] == "SPY"].sort_index()

feature_cols = [c for c in ticker_df.columns
                if c not in ["target", "ticker", "close"]]

data = ticker_df[feature_cols + ["target"]].values
X, y = make_sequences(data, seq_len=20)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
print(f"Feature count: {X_train.shape[2]}")

input_size = X_train.shape[2]
lstm = LSTMModel(input_size=input_size)
transformer = TransformerModel(input_size=input_size)

x_sample = torch.tensor(X_train[:4], dtype=torch.float32)

lstm.eval()
transformer.eval()

with torch.no_grad():
    lstm_out = lstm(x_sample)
    trans_out = transformer(x_sample)

print(f"LSTM output shape: {lstm_out.shape}")
print(f"Transformer output shape: {trans_out.shape}")
print("Model smoke test passed.")