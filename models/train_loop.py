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
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(model(xb).numpy())
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