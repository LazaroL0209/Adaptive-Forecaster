from fastapi import FastAPI, Response
import torch
import pandas as pd
import numpy as np
import mlflow.pytorch
import os

app   = FastAPI(title="Adaptive Forecaster API")
state = {"model": None, "model_version": "none", "last_retrain": "never"}

@app.on_event("startup")
def load_model():
    model_path = "experiments/best_model"
    if os.path.exists(model_path):
        state["model"] = mlflow.pytorch.load_model(model_path)
        state["model"].eval()
        print("Model loaded.")
    else:
        print("No saved model found. Train a model first.")

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_version": state["model_version"],
        "last_retrain":  state["last_retrain"],
    }

@app.get("/forecast")
def forecast(ticker: str = "SPY", horizon: int = 1, response: Response = None):
    if state["model"] is None:
        return {"error": "No model loaded. Run training first."}
    features = pd.read_parquet("data/features.parquet")
    ticker_df = features[features["ticker"] == ticker].sort_index()
    feature_cols = [c for c in ticker_df.columns
                    if c not in ["target", "ticker", "close"]]
    seq = ticker_df[feature_cols].values[-20:]
    x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = state["model"](x).item()
    if response:
        response.headers["X-Model-Version"] = state["model_version"]
    return {
        "ticker":           ticker,
        "predicted_return": round(pred, 6),
        "direction":        "long" if pred > 0 else "flat",
    }