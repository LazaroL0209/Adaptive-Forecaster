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
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)

def kfold_evaluate(model_class, data, seq_len=20, n_splits=5,
                   epochs=50, lr=1e-3, input_size=None, log_callback=None):
    from sklearn.model_selection import TimeSeriesSplit
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        X_train, y_train = make_sequences(data[train_idx], seq_len)
        X_val,   y_val   = make_sequences(data[val_idx],   seq_len)
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        in_size = input_size or X_train.shape[2]
        model   = model_class(input_size=in_size)
        mae, acc, stopped, precision, recall, f1 = train_model(
            model, X_train, y_train, X_val, y_val,
            model_name=f"kfold_fold_{fold}",
            epochs=epochs, lr=lr,
            log_callback=log_callback
        )
        scores.append({"fold": fold+1, "mae": mae, "acc": acc,
                       "precision": precision, "recall": recall, "f1": f1})
        if log_callback:
            log_callback(f"  Fold {fold+1}: MAE={mae:.6f} Acc={acc:.3f} F1={f1:.3f}")

    avg_mae = float(np.mean([s["mae"] for s in scores]))
    avg_acc = float(np.mean([s["acc"] for s in scores]))
    avg_f1  = float(np.mean([s["f1"] for s in scores]))
    if log_callback:
        log_callback(f"  K-Fold avg: MAE={avg_mae:.6f} Acc={avg_acc:.3f} F1={avg_f1:.3f}")
    return avg_mae, avg_acc, avg_f1, scores

def train_model(model, X_train, y_train, X_val, y_val,
                model_name="model", epochs=50, lr=1e-3, batch_size=64,
                early_stopping_patience=10, log_callback=None):

    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    mlflow.set_experiment("adaptive-forecaster")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model",      model_name)
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("lr",         lr)
        mlflow.log_param("batch_size", batch_size)

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds   = TimeSeriesDataset(X_val,   y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion  = nn.MSELoss()
        best_val   = float("inf")
        patience   = 0
        best_state = None
        stopped_at = epochs

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                output = model(xb)
                pred   = output[1] if isinstance(output, tuple) else output
                if pred.dim() > 1:
                    pred = pred[:, -1]
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_preds, val_acts = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    output = model(xb)
                    pred   = output[1] if isinstance(output, tuple) else output
                    if pred.dim() > 1:
                        pred = pred[:, -1]
                    val_preds.extend(pred.numpy())
                    val_acts.extend(yb.numpy())

            val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_acts))))
            mlflow.log_metric("val_mae", val_mae, step=epoch)

            if log_callback and (epoch + 1) % 5 == 0:
                log_callback(f"  epoch {epoch+1}/{epochs} | val_mae: {val_mae:.6f}")

            if val_mae < best_val - 1e-6:
                best_val   = val_mae
                patience   = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    stopped_at = epoch + 1
                    if log_callback:
                        log_callback(f"  early stop at epoch {stopped_at} (best val_mae: {best_val:.6f})")
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                output = model(xb)
                pred   = output[1] if isinstance(output, tuple) else output
                if pred.dim() > 1:
                    pred = pred[:, -1]
                preds.extend(pred.numpy())
                actuals.extend(yb.numpy())

        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        preds   = np.array(preds)
        actuals = np.array(actuals)
        mae     = float(np.mean(np.abs(preds - actuals)))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))

        preds_binary   = (preds   > 0).astype(int)
        actuals_binary = (actuals > 0).astype(int)
        precision = float(precision_score(actuals_binary, preds_binary, zero_division=0))
        recall    = float(recall_score(actuals_binary,    preds_binary, zero_division=0))
        f1        = float(f1_score(actuals_binary,        preds_binary, zero_division=0))
        cm        = confusion_matrix(actuals_binary, preds_binary)
        tp = int(cm[1][1]) if cm.shape == (2, 2) else 0
        fp = int(cm[0][1]) if cm.shape == (2, 2) else 0
        tn = int(cm[0][0]) if cm.shape == (2, 2) else 0
        fn = int(cm[1][0]) if cm.shape == (2, 2) else 0

        mlflow.log_metric("final_val_mae",        mae)
        mlflow.log_metric("directional_accuracy", dir_acc)
        mlflow.log_metric("precision",            precision)
        mlflow.log_metric("recall",               recall)
        mlflow.log_metric("f1_score",             f1)
        mlflow.log_metric("true_positives",       tp)
        mlflow.log_metric("false_positives",      fp)
        mlflow.log_metric("true_negatives",       tn)
        mlflow.log_metric("false_negatives",      fn)
        mlflow.log_metric("stopped_at_epoch",     stopped_at)
        mlflow.pytorch.log_model(model, "model")

        if log_callback:
            log_callback(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            log_callback(f"  TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

        return mae, dir_acc, stopped_at, precision, recall, f1


if __name__ == "__main__":
    import pandas as pd
    import os
    import shutil
    from lstm_model import LSTMModel
    from transformer_model import TransformerModel
    from nbeats_pytorch.model import NBeatsNet

    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")

    print("Loading features...")
    df = pd.read_parquet("data/features.parquet")

    feature_cols = [c for c in df.columns if c not in ["target", "ticker", "close"]]
    results = {}

    for ticker in ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT"]:
        print(f"\nTraining on {ticker}...")
        ticker_df = df[df["ticker"] == ticker].sort_index()
        data = ticker_df[["log_return"] + [c for c in feature_cols if c != "log_return"]].values

        split      = int(len(data) * 0.8)
        train_data = data[:split]
        val_data   = data[split:]

        X_train, y_train = make_sequences(train_data)
        X_val,   y_val   = make_sequences(val_data)

        input_size = X_train.shape[2]
        print(f"  Train sequences: {len(X_train)} | Val sequences: {len(X_val)} | Features: {input_size}")

        lstm = LSTMModel(input_size=input_size)
        mae, acc, stopped, precision, recall, f1 = train_model(
            lstm, X_train, y_train, X_val, y_val,
            model_name=f"lstm_{ticker}", epochs=100)
        results[f"lstm_{ticker}"] = (mae, acc, lstm)

        transformer = TransformerModel(input_size=input_size)
        mae, acc, stopped, precision, recall, f1 = train_model(
            transformer, X_train, y_train, X_val, y_val,
            model_name=f"transformer_{ticker}", epochs=100)
        results[f"transformer_{ticker}"] = (mae, acc, transformer)

        nbeats = NBeatsNet(
            stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
            forecast_length=1,
            backcast_length=20,
            hidden_layer_units=64,
        )
        mae, acc, stopped, precision, recall, f1 = train_model(
            nbeats, X_train[:, :, 0:1], y_train,
            X_val[:, :, 0:1], y_val,
            model_name=f"nbeats_{ticker}", epochs=100)
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
    if os.path.exists("experiments/best_model"):
        shutil.rmtree("experiments/best_model")
    mlflow.pytorch.save_model(best_model, "experiments/best_model")
    print("Saved to experiments/best_model")