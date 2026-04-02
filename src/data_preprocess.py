"""
Energy-Demand-Forecasting-DevMLOps
Advanced Data Preprocessing & Feature Engineering

Author: Corey Leath (Trojan3877)

Provides:
✔ Timestamp parsing & extraction
✔ Holiday encoding
✔ Lag features
✔ Rolling window statistics
✔ Missing value handling
✔ Normalization (shared scaler)
✔ Saves cleaned dataset → data/processed.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.utils import ensure_dir, save_json, save_scaler


# -----------------------------------------------------------
# Load Raw CSV
# -----------------------------------------------------------
def load_raw_data(path):
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    return df



# -----------------------------------------------------------
# Add Timestamp Features
# -----------------------------------------------------------
def add_time_features(df):
    if "timestamp" not in df.columns:
        raise KeyError("Dataset must include a 'timestamp' column.")

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Seasonal feature (optional)
    df["season"] = (
        (df["month"] % 12 + 3) // 3
    )  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

    return df



# -----------------------------------------------------------
# Add Holiday Features
# -----------------------------------------------------------
def add_holiday_features(df):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["timestamp"].min(), end=df["timestamp"].max())

    df["is_holiday"] = df["timestamp"].isin(holidays).astype(int)
    return df



# -----------------------------------------------------------
# Add Lag Features
# -----------------------------------------------------------
def add_lag_features(df, target_col="value", lags=[1, 24, 48, 72]):
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df



# -----------------------------------------------------------
# Add Rolling Window Features
# -----------------------------------------------------------
def add_rolling_features(df, target_col="value", windows=[24, 48, 72]):
    for win in windows:
        df[f"roll_mean_{win}"] = df[target_col].rolling(win).mean()
        df[f"roll_std_{win}"] = df[target_col].rolling(win).std()
        df[f"roll_min_{win}"] = df[target_col].rolling(win).min()
        df[f"roll_max_{win}"] = df[target_col].rolling(win).max()
    return df



# -----------------------------------------------------------
# Handle Missing Values
# -----------------------------------------------------------
def handle_missing(df):
    df = df.ffill()
    df = df.bfill()
    return df



# -----------------------------------------------------------
# Normalize Features
# -----------------------------------------------------------
def normalize_features(df, target_col="value"):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)

    save_scaler(scaler, "models/scaler.pkl")
    return pd.DataFrame(scaled_values, columns=df.columns), scaler



# -----------------------------------------------------------
# Run Full Preprocessing Pipeline
# -----------------------------------------------------------
def run_preprocessing(
    raw_path="data/raw.csv",
    output_path="data/processed.csv",
    target_col="value"
):
    print("[INFO] Starting preprocessing pipeline...")

    # Load raw
    df = load_raw_data(raw_path)
    print(f"[INFO] Loaded raw dataset → {raw_path}, shape={df.shape}")

    # Feature Engineering
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)

    # Clean NA created by lags/rolls
    df = handle_missing(df)

    # Save pre-normalized metadata
    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "target_column": target_col,
        "lag_features": [1, 24, 48, 72],
        "rolling_windows": [24, 48, 72],
    }
    ensure_dir("artifacts")
    save_json(metadata, "artifacts/preprocess_metadata.json")

    # Normalize entire dataset
    numeric_df = df.select_dtypes(include=[np.number])
    scaled_df, _ = normalize_features(numeric_df)

    # Save processed dataset
    ensure_dir("data")
    scaled_df.to_csv(output_path, index=False)

    print(f"[INFO] Preprocessing complete → saved to {output_path}")
    return scaled_df
    


# -----------------------------------------------------------
# Script Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
