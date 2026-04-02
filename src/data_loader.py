"""
Energy-Demand-Forecasting-DevMLOps
L6 Production Data Loader and Feature Engineering Pipeline

Author: Corey Leath (Trojan3877)

This module performs:
✔ Data ingestion
✔ Datetime parsing and sorting
✔ Missing value handling
✔ Time feature generation
✔ Lag and moving average features
✔ Scaling (MinMax)
✔ Train/val/test split
✔ Sliding-window creation for LSTM/GRU/Transformer
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import (
    ensure_dir,
    load_config,
    save_scaler,
    print_gpu_info,
)

# -------------------------------------------------------------
# Load Raw Dataset
# -------------------------------------------------------------
def load_raw_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Raw dataset not found: {path}")

    print(f"[INFO] Loading dataset: {path}")
    df = pd.read_csv(path)

    return df


# -------------------------------------------------------------
# Preprocess Raw Data
# -------------------------------------------------------------
def preprocess_data(df, config):
    dt_col = config["data"]["datetime_column"]
    target_col = config["data"]["target_column"]

    # Convert to datetime
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    # Sort by date
    df = df.sort_values(dt_col).reset_index(drop=True)

    # Handle missing target values
    df[target_col] = df[target_col].interpolate().bfill()

    return df


# -------------------------------------------------------------
# Time-Based Feature Engineering
# -------------------------------------------------------------
def add_time_features(df, config):
    if not config["features"]["create_time_features"]:
        return df

    dt_col = config["data"]["datetime_column"]

    df["hour"] = df[dt_col].dt.hour
    df["day"] = df[dt_col].dt.day
    df["day_of_week"] = df[dt_col].dt.dayofweek
    df["month"] = df[dt_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


# -------------------------------------------------------------
# Add Lag Features
# -------------------------------------------------------------
def add_lag_features(df, config):
    lags = config["features"]["include_lags"]
    target_col = config["data"]["target_column"]

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df


# -------------------------------------------------------------
# Add Moving Average Features
# -------------------------------------------------------------
def add_moving_average_features(df, config):
    ma_settings = config["features"]["include_moving_averages"]
    target_col = config["data"]["target_column"]

    for setting in ma_settings:
        window = setting["window"]
        df[f"ma_{window}"] = df[target_col].rolling(window).mean()

    return df


# -------------------------------------------------------------
# Scale Features
# -------------------------------------------------------------
def scale_features(df, config):
    if not config["features"]["scale_features"]:
        return df, None

    feature_cols = [col for col in df.columns if col not in [
        config["data"]["datetime_column"],
        config["data"]["target_column"]
    ]]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    save_scaler(scaler, "models/scaler.pkl")

    return df, scaler


# -------------------------------------------------------------
# Create Sliding Window Sequences for LSTM/GRU/Transformer
# -------------------------------------------------------------
def create_sequences(df, config):
    target = config["data"]["target_column"]
    seq_len = config["model"]["sequence_length"]
    horizon = config["model"]["forecast_horizon"]

    feature_cols = df.columns.drop([config["data"]["datetime_column"]]).tolist()
    values = df[feature_cols].values

    X, y = [], []

    for i in range(len(values) - seq_len - horizon):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len : i + seq_len + horizon, feature_cols.index(target)])

    return np.array(X), np.array(y)


# -------------------------------------------------------------
# Train / Validation / Test Split
# -------------------------------------------------------------
def train_val_test_split(X, y, config):
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    test_len = int(len(X) * test_size)
    val_len = int(len(X) * val_size)

    X_train, y_train = X[: -(test_len + val_len)], y[: -(test_len + val_len)]
    X_val, y_val = X[-(test_len + val_len) : -test_len], y[-(test_len + val_len) : -test_len]
    X_test, y_test = X[-test_len:], y[-test_len:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------------------------------------------
# Main loader function (Called by train.py)
# -------------------------------------------------------------
def load_dataset(config_path="config/config.yaml"):
    config = load_config(config_path)

    print_gpu_info()

    df = load_raw_data(config["data"]["raw_path"])
    df = preprocess_data(df, config)
    df = add_time_features(df, config)
    df = add_lag_features(df, config)
    df = add_moving_average_features(df, config)

    df = df.dropna().reset_index(drop=True)

    df, scaler = scale_features(df, config)

    X, y = create_sequences(df, config)
    splits = train_val_test_split(X, y, config)

    print("[INFO] Dataset prepared successfully.")
    print(f" - X shape: {X.shape}")
    print(f" - y shape: {y.shape}")

    return splits
