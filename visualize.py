import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pytorch
from models.train_loop import make_sequences

mlflow.set_tracking_uri("sqlite:///experiments/mlflow.db")

print("Loading model and data...")
model = mlflow.pytorch.load_model("experiments/best_model")
model.eval()

df = pd.read_parquet("data/features.parquet")
ticker = "SPY"
ticker_df = df[df["ticker"] == ticker].sort_index()

feature_cols = [c for c in ticker_df.columns if c not in ["target", "ticker", "close"]]
data = ticker_df[["log_return"] + [c for c in feature_cols if c != "log_return"]].values
dates = ticker_df.index

split = int(len(data) * 0.8)
val_data = data[split:]
val_dates = dates[split + 20:]

X_val, y_val = make_sequences(val_data)

preds = []
with torch.no_grad():
    for i in range(len(X_val)):
        x = torch.tensor(X_val[i], dtype=torch.float32).unsqueeze(0)
        out = model(x)
        pred = out[1] if isinstance(out, tuple) else out
        if pred.dim() > 1:
            pred = pred[:, -1]
        preds.append(pred.item())

preds   = np.array(preds)
actuals = y_val

dir_acc = np.mean(np.sign(preds) == np.sign(actuals))
mae     = np.mean(np.abs(preds - actuals))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(f"SPY Forecast — LSTM | MAE: {mae:.6f} | Dir Acc: {dir_acc:.1%}", fontsize=14)

# Panel 1: predicted vs actual returns
ax1 = axes[0]
ax1.plot(val_dates[:len(actuals)], actuals, label="Actual", alpha=0.7, linewidth=0.8)
ax1.plot(val_dates[:len(preds)],   preds,   label="Predicted", alpha=0.7, linewidth=0.8)
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_title("Predicted vs Actual Log Returns")
ax1.set_ylabel("Log Return")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: directional accuracy rolling 30 days
correct = (np.sign(preds) == np.sign(actuals)).astype(float)
rolling_acc = pd.Series(correct).rolling(30).mean()
ax2 = axes[1]
ax2.plot(val_dates[:len(rolling_acc)], rolling_acc, color="green", linewidth=1)
ax2.axhline(0.5, color="red", linewidth=0.8, linestyle="--", label="Random baseline (50%)")
ax2.axhline(dir_acc, color="blue", linewidth=0.8, linestyle="--", label=f"Overall {dir_acc:.1%}")
ax2.set_title("Rolling 30-Day Directional Accuracy")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: cumulative paper P&L vs buy and hold
strategy_returns  = actuals * np.sign(preds)
buyhold_returns   = actuals
cumulative_strat  = np.cumprod(1 + strategy_returns) - 1
cumulative_bh     = np.cumprod(1 + buyhold_returns)  - 1
ax3 = axes[2]
ax3.plot(val_dates[:len(cumulative_strat)], cumulative_strat * 100, label="Model strategy", color="green")
ax3.plot(val_dates[:len(cumulative_bh)],    cumulative_bh    * 100, label="Buy and hold SPY", color="gray", alpha=0.7)
ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("Cumulative Paper P&L vs Buy and Hold (%)")
ax3.set_ylabel("Return (%)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("forecast_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to forecast_visualization.png")