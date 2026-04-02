"""
build_features.py

Module: Feature Engineering for Energy Demand Forecasting Pipeline
Author: Corey Leath

Description:
- Loads processed combined data
- Creates rolling window features
- Outputs engineered features CSV

Input:
- data/processed/combined.csv

Output:
- data/processed/features.csv
"""

import pandas as pd
import os

def main():
    # Define input path
    input_path = 'data/processed/combined.csv'

    # Define output path
    output_path = 'data/processed/features.csv'

    # Load combined data
    print(f"Loading combined data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Create rolling window features
    print("Creating rolling window features (3-hour moving average)...")
    if 'load' in df.columns:
        df['load_ma_3h'] = df['load'].rolling(window=3).mean()
    if 'temperature' in df.columns:
        df['temperature_ma_3h'] = df['temperature'].rolling(window=3).mean()

    # Drop rows with NaN created by rolling
    df.dropna(inplace=True)

    # Save engineered features
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved engineered features to {output_path}.")

if __name__ == "__main__":
    main()
