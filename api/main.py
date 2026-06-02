import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import numpy as np
import mlflow.pytorch
import mlflow

app = FastAPI(title="Adaptive Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "models":            {},
    "model_version":     "none",
    "last_retrain":      "never",
    "training":          False,
    "training_log":      [],
    "training_progress": {"current": "", "step": 0, "total": 0, "epoch": 0, "max_epochs": 0},
    "results_table":     [],
    "conf_threshold":    0.0,
}

@app.on_event("startup")
def load_models():
    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    try:
        df = pd.read_parquet("data/features.parquet")
    except Exception:
        print("No features.parquet found. Run pipeline first.")
        return

    for ticker in ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT"]:
        for arch in ["lstm", "transformer", "nbeats"]:
            model_path = f"experiments/model_{arch}_{ticker}"
            if os.path.exists(model_path):
                try:
                    m = mlflow.pytorch.load_model(model_path)
                    m.eval()
                    state["models"][ticker] = m
                    print(f"Loaded {arch}_{ticker}")
                    break
                except Exception as e:
                    print(f"Could not load {arch}_{ticker}: {e}")

    if not state["models"]:
        for path in ["experiments/best_model"]:
            if os.path.exists(path):
                try:
                    m = mlflow.pytorch.load_model(path)
                    m.eval()
                    for ticker in ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT"]:
                        state["models"][ticker] = m
                    print("Loaded shared best model.")
                    break
                except Exception as e:
                    print(f"Could not load best model: {e}")

    print(f"Models ready: {list(state['models'].keys())}")

@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_version":     state["model_version"],
        "last_retrain":      state["last_retrain"],
        "tickers_loaded":    list(state["models"].keys()),
        "training":          state["training"],
        "training_progress": state["training_progress"],
    }

@app.get("/forecast")
def forecast(ticker: str = "SPY", response: Response = None):
    if not state["models"]:
        return {"error": "No models loaded. Run training first."}

    try:
        df        = pd.read_parquet("data/features.parquet")
        ticker_df = df[df["ticker"] == ticker].sort_index()
        if ticker_df.empty:
            return {"error": f"No data for {ticker}"}

        feature_cols = [c for c in ticker_df.columns if c not in ["target", "ticker", "close"]]
        seq_len      = state.get("last_seq_len", 20)
        seq_raw      = ticker_df[feature_cols].values[-seq_len:]
        mean         = seq_raw.mean(axis=0)
        std          = seq_raw.std(axis=0) + 1e-9
        seq          = (seq_raw - mean) / std
        x            = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        model = state["models"].get(ticker) or list(state["models"].values())[0]

        with torch.no_grad():
            if isinstance(model, dict) and model.get("mode") == "ensemble":
                preds_list = []
                for arch, m in model.items():
                    if arch == "mode":
                        continue
                    inp  = x if arch != "nbeats" else x[:, :, 0:1]
                    out  = m(inp)
                    p    = out[1] if isinstance(out, tuple) else out
                    if p.dim() > 1:
                        p = p[:, -1]
                    preds_list.append(p.item())
                pred = float(np.mean(preds_list))
            else:
                out  = model(x)
                pred = out[1] if isinstance(out, tuple) else out
                if isinstance(pred, torch.Tensor):
                    if pred.dim() > 1:
                        pred = pred[:, -1]
                    pred = pred.item()

        ticker_std       = float(ticker_df["log_return"].std())
        confidence_ratio = abs(pred) / (ticker_std + 1e-9)
        confidence       = "High" if confidence_ratio > 0.5 else "Medium" if confidence_ratio > 0.2 else "Low"
        threshold        = state.get("conf_threshold", 0.0)
        direction        = ("flat" if threshold > 0 and abs(pred) < threshold
                           else "long" if pred > 0 else "flat")

        if response:
            response.headers["X-Model-Version"] = state["model_version"]

        return {
            "ticker":            ticker,
            "predicted_return":  round(pred, 6),
            "direction":         direction,
            "confidence":        confidence,
            "confidence_ratio":  round(confidence_ratio, 4),
            "ticker_volatility": round(ticker_std, 6),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/training_status")
def training_status():
    return {
        "training":  state["training"],
        "progress":  state["training_progress"],
        "log":       state["training_log"][-50:],
        "results":   state["results_table"],
    }

@app.post("/train")
def trigger_training(
    background_tasks:  BackgroundTasks,
    epochs:            int   = 50,
    seq_len:           int   = 20,
    lr:                float = 0.001,
    tickers:           str   = "SPY,QQQ,AAPL,MSFT,GLD,TLT",
    use_ensemble:      bool  = False,
    use_conf_filter:   bool  = False,
    conf_percentile:   float = 40.0,
    early_stopping:    bool  = True,
    patience:          int   = 10,
):
    if state["training"]:
        return {"error": "Training already in progress"}
    ticker_list = [t.strip() for t in tickers.split(",")]
    state["last_seq_len"] = seq_len
    background_tasks.add_task(
        run_training, epochs, seq_len, lr, ticker_list,
        use_ensemble, use_conf_filter, conf_percentile,
        early_stopping, patience
    )
    return {"status": "Training started"}

def run_training(epochs, seq_len, lr, ticker_list,
                 use_ensemble, use_conf_filter, conf_percentile,
                 early_stopping, patience):
    import shutil
    from train_loop import make_sequences, train_model
    from lstm_model import LSTMModel
    from transformer_model import TransformerModel
    from nbeats_pytorch.model import NBeatsNet

    def log(msg):
        state["training_log"].append(msg)

    state["training"]        = True
    state["training_log"]    = []
    state["results_table"]   = []
    total_models             = len(ticker_list) * (3 if use_ensemble else 1)
    state["training_progress"] = {
        "current":    "Initializing...",
        "step":       0,
        "total":      total_models,
        "epoch":      0,
        "max_epochs": epochs,
    }

    mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")
    df           = pd.read_parquet("data/features.parquet")
    feature_cols = [c for c in df.columns if c not in ["target", "ticker", "close"]]

    best_mae    = float("inf")
    best_model  = None
    best_name   = ""
    all_preds   = []
    all_actuals = []
    step        = 0

    for ticker in ticker_list:
        log(f"━━━ {ticker} ━━━")
        ticker_df  = df[df["ticker"] == ticker].sort_index()
        data       = ticker_df[["log_return"] + [c for c in feature_cols if c != "log_return"]].values
        split      = int(len(data) * 0.8)
        X_train, y_train = make_sequences(data[:split], seq_len)
        X_val,   y_val   = make_sequences(data[split:], seq_len)
        input_size = X_train.shape[2]

        log(f"  Train: {len(X_train)} sequences | Val: {len(X_val)} sequences | Features: {input_size}")

        archs_to_run  = ["lstm", "transformer", "nbeats"] if use_ensemble else ["lstm"]
        ticker_models = {}

        for arch in archs_to_run:
            step += 1
            state["training_progress"]["current"]    = f"{arch.upper()} on {ticker}"
            state["training_progress"]["step"]       = step
            state["training_progress"]["epoch"]      = 0
            state["training_progress"]["max_epochs"] = epochs
            log(f"  Training {arch.upper()}...")

            def epoch_cb(msg, arch=arch):
                ep = int(msg.split("epoch")[1].split("/")[0].strip()) if "epoch" in msg else 0
                state["training_progress"]["epoch"] = ep
                log(f"  [{arch.upper()}] {msg.strip()}")

            try:
                if arch == "lstm":
                    model      = LSTMModel(input_size=input_size)
                    X_in, y_in = X_train, y_train
                    X_v,  y_v  = X_val,   y_val
                elif arch == "transformer":
                    model      = TransformerModel(input_size=input_size)
                    X_in, y_in = X_train, y_train
                    X_v,  y_v  = X_val,   y_val
                else:
                    model = NBeatsNet(
                        stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                        forecast_length=1, backcast_length=seq_len, hidden_layer_units=64,
                    )
                    X_in, y_in = X_train[:, :, 0:1], y_train
                    X_v,  y_v  = X_val[:, :, 0:1],   y_val

                mae, acc, stopped, precision, recall, f1 = train_model(
                    model, X_in, y_in, X_v, y_v,
                    model_name=f"{arch}_{ticker}",
                    epochs=epochs, lr=lr,
                    early_stopping_patience=patience if early_stopping else epochs + 1,
                    log_callback=epoch_cb,
                )

                ticker_models[arch] = (model, mae, acc)
                log(f"  {arch.upper()} done | MAE: {mae:.6f} | Dir Acc: {acc:.1%} | F1: {f1:.3f} | Stopped: ep {stopped}")

                state["results_table"].append({
                    "model":     f"{arch}_{ticker}",
                    "ticker":    ticker,
                    "arch":      arch.upper(),
                    "mae":       round(mae, 6),
                    "dir_acc":   round(acc * 100, 1),
                    "precision": round(precision, 3),
                    "recall":    round(recall, 3),
                    "f1":        round(f1, 3),
                    "stopped":   stopped,
                    "status":    "done",
                    "best":      False,
                })

                path = f"experiments/model_{arch}_{ticker}"
                if os.path.exists(path):
                    shutil.rmtree(path)
                mlflow.pytorch.save_model(model, path)

                if mae < best_mae:
                    best_mae   = mae
                    best_model = model
                    best_name  = f"{arch}_{ticker}"

                model.eval()
                x_t = torch.tensor(X_v, dtype=torch.float32)
                with torch.no_grad():
                    out  = model(x_t)
                    p    = out[1] if isinstance(out, tuple) else out
                    if p.dim() > 1:
                        p = p[:, -1]
                    all_preds.extend(p.numpy().tolist())
                    all_actuals.extend(y_v.tolist())

            except Exception as e:
                log(f"  {arch.upper()} failed: {str(e)}")
                state["results_table"].append({
                    "model":     f"{arch}_{ticker}",
                    "ticker":    ticker,
                    "arch":      arch.upper(),
                    "mae":       None,
                    "dir_acc":   None,
                    "precision": None,
                    "recall":    None,
                    "f1":        None,
                    "stopped":   None,
                    "status":    "error",
                    "best":      False,
                })

        if use_ensemble and len(ticker_models) > 1:
            ensemble = {arch: m for arch, (m, _, _) in ticker_models.items()}
            ensemble["mode"] = "ensemble"
            state["models"][ticker] = ensemble
            log(f"  Ensemble active for {ticker}")
        elif ticker_models:
            best_arch = min(ticker_models, key=lambda k: ticker_models[k][1])
            state["models"][ticker] = ticker_models[best_arch][0]

    if use_conf_filter and all_preds:
        preds_arr   = np.array(all_preds)
        threshold   = float(np.percentile(np.abs(preds_arr), conf_percentile))
        state["conf_threshold"] = threshold
        confident   = np.abs(preds_arr) > threshold
        actuals_arr = np.array(all_actuals)
        if confident.sum() > 0:
            f_acc = float(np.mean(np.sign(preds_arr[confident]) == np.sign(actuals_arr[confident])))
            log(f"Confidence filter | threshold: {threshold:.6f} | filtered acc: {f_acc:.1%} | trades: {confident.sum()}/{len(preds_arr)}")
    else:
        state["conf_threshold"] = 0.0

    if best_model:
        path = "experiments/best_model"
        if os.path.exists(path):
            shutil.rmtree(path)
        mlflow.pytorch.save_model(best_model, path)
        state["model_version"] = best_name
        for row in state["results_table"]:
            row["best"] = row["model"] == best_name

    state["last_retrain"]      = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    state["training"]          = False
    state["training_progress"] = {
        "current": "complete", "step": total_models,
        "total": total_models, "epoch": epochs, "max_epochs": epochs
    }
    log(f"Training complete. Best: {best_name} | MAE: {best_mae:.6f}")