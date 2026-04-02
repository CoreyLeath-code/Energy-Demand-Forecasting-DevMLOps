"""
preprocess.py

Module: Data Preprocessing for Energy Demand Forecasting Pipeline
Author: Corey Leath

Description:
- Loads raw energy demand and weather data
- Merges on timestamp
- Cleans missing values
- Outputs processed combined CSV

Input:
- data/raw/energy.csv
- data/raw/weather.csv

Output:
- data/processed/combined.csv
"""

import pandas as pd
import os

def main():
    # Define input paths
    energy_path = 'data/raw/energy.csv'
    weather_path = 'data/raw/weather.csv'

    # Define output path
    output_path = 'data/processed/combined.csv'

    # Load raw energy data
    print(f"Loading energy data from {energy_path}...")
    energy_df = pd.read_csv(energy_path, parse_dates=['timestamp'])

    # Load raw weather data
    print(f"Loading weather data from {weather_path}...")
    weather_df = pd.read_csv(weather_path, parse_dates=['timestamp'])

    # Merge on timestamp
    print("Merging energy and weather data...")
    df = pd.merge(energy_df, weather_df, on='timestamp', how='inner')

    # Clean missing values
    print("Cleaning missing values...")
    df.dropna(inplace=True)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}.")

if __name__ == "__main__":
    main()
