"""
evaluate.py

Module: Model Evaluation for Energy Demand Forecasting Pipeline
Author: Corey Leath

Description:
- Loads trained model
- Loads test data
- Runs predictions
- Computes MAPE and latency
- Logs metrics

Input:
- models/energy_forecast_model.pkl
- data/processed/features.csv

Metrics:
- MAPE (Mean Absolute Percentage Error)
- Inference Latency (ms/sample)

"""

import pandas as pd
import joblib
import time
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def main():
    # Load config
    config_path = 'configs/train.yaml'
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load trained model
    model_path = 'models/energy_forecast_model.pkl'
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # Load test data
    input_path = 'data/processed/features.csv'
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    feature_cols = config['features']
    target_col = config['target']
    test_size = config['test_size']
    random_state = config['random_state']

    X = df[feature_cols]
    y = df[target_col]

    # Split data (same as in train)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Run predictions + measure latency
    print("Running predictions...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    # Compute MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE on test set: {mape:.4f}")

    # Compute latency per sample (ms)
    total_latency = (end_time - start_time) * 1000  # ms
    latency_per_sample = total_latency / len(y_test)
    print(f"Inference latency: {latency_per_sample:.2f} ms/sample")

    # Save evaluation metrics
    metrics_path = 'models/evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"MAPE: {mape:.4f}\n")
        f.write(f"Inference latency: {latency_per_sample:.2f} ms/sample\n")

    print(f"Saved evaluation metrics to {metrics_path}.")

if __name__ == "__main__":
    main()
