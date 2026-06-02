import pytest
import pandas as pd
import numpy as np
import os

def test_data_files_exist():
    assert os.path.exists("data/close_prices.parquet"), "close_prices.parquet missing"
    assert os.path.exists("data/volume.parquet"), "volume.parquet missing"
    assert os.path.exists("data/macro.parquet"), "macro.parquet missing"

def test_close_prices_shape():
    df = pd.read_parquet("data/close_prices.parquet")
    assert len(df) > 0, "close_prices is empty"
    assert len(df.columns) == 20, "expected 20 tickers"

def test_features_exist():
    assert os.path.exists("data/features.parquet"), "features.parquet missing"

def test_features_no_nulls():
    df = pd.read_parquet("data/features.parquet")
    assert df.isnull().sum().sum() == 0, "features contain null values"

def test_features_has_target():
    df = pd.read_parquet("data/features.parquet")
    assert "target" in df.columns, "target column missing"

def test_log_returns_reasonable():
    df = pd.read_parquet("data/features.parquet")
    assert df["log_return"].abs().max() < 1.0, "log returns contain unreasonable values"