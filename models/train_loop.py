from pyexpat import model

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_sequences(data: np.ndarray, seq_len: int = 20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])  # target is first column (log_return)
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, X_val, y_val,
                model_name="model", epochs=30, lr=1e-3, batch_size=64):

    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    mlflow.set_experiment("adaptive-forecaster")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds   = TimeSeriesDataset(X_val,   y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                output = model(xb)
                pred = output[1] if isinstance(output, tuple) else output
                if pred.dim() > 1:
                    pred = pred[:, -1]                
                pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                loss = criterion(pred, yb)                
                loss.backward()
                optimizer.step()

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                output = model(xb)
                pred = output[1] if isinstance(output, tuple) else output
                if pred.dim() > 1:
                    pred = pred[:, -1]                
                pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                preds.extend(pred.numpy())                
                actuals.extend(yb.numpy())

        preds   = np.array(preds)
        actuals = np.array(actuals)
        mae     = np.mean(np.abs(preds - actuals))
        dir_acc = np.mean(np.sign(preds) == np.sign(actuals))

        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("directional_accuracy", dir_acc)
        mlflow.pytorch.log_model(model, "model")

        print(f"{model_name} | MAE: {mae:.6f} | Dir Acc: {dir_acc:.3f}")
        return mae, dir_acc
    
if __name__ == "__main__":
    import pandas as pd
    import os
    from lstm_model import LSTMModel
    from transformer_model import TransformerModel
    from nbeats_pytorch.model import NBeatsNet

    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")

    print("Loading features...")
    df = pd.read_parquet("data/features.parquet")

    feature_cols = [c for c in df.columns if c not in ["target", "ticker", "close"]]
    results = {}

    for ticker in ["SPY", "QQQ", "AAPL"]:
        print(f"\nTraining on {ticker}...")
        ticker_df = df[df["ticker"] == ticker].sort_index()
        data = ticker_df[["log_return"] + [c for c in feature_cols if c != "log_return"]].values

        split = int(len(data) * 0.8)
        train_data = data[:split]
        val_data   = data[split:]

        X_train, y_train = make_sequences(train_data)
        X_val,   y_val   = make_sequences(val_data)

        input_size = X_train.shape[2]
        print(f"  Train sequences: {len(X_train)} | Val sequences: {len(X_val)} | Features: {input_size}")

        lstm = LSTMModel(input_size=input_size)
        mae, acc = train_model(lstm, X_train, y_train, X_val, y_val,
                               model_name=f"lstm_{ticker}")
        results[f"lstm_{ticker}"] = (mae, acc, lstm)

        transformer = TransformerModel(input_size=input_size)
        mae, acc = train_model(transformer, X_train, y_train, X_val, y_val,
                               model_name=f"transformer_{ticker}")
        results[f"transformer_{ticker}"] = (mae, acc, transformer)

        nbeats = NBeatsNet(
            stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
            forecast_length=1,
            backcast_length=20,
            hidden_layer_units=64,
        )
        mae, acc = train_model(nbeats, X_train[:, :, 0:1], y_train,
                               X_val[:, :, 0:1],   y_val,
                               model_name=f"nbeats_{ticker}")
        results[f"nbeats_{ticker}"] = (mae, acc, nbeats)

    print("\nAll results:")
    best_mae   = float("inf")
    best_model = None
    best_name  = ""
    for name, (mae, acc, model) in results.items():
        print(f"  {name} | MAE: {mae:.6f} | Dir Acc: {acc:.3f}")
        if mae < best_mae:
            best_mae   = mae
            best_model = model
            best_name  = name

    print(f"\nBest model: {best_name} with MAE {best_mae:.6f}")
    os.makedirs("experiments/best_model", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    mlflow.pytorch.save_model(best_model, "experiments/best_model")
    print("Saved to experiments/best_model")