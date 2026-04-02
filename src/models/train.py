"""
train.py

Module: Model Training for Energy Demand Forecasting Pipeline
Author: Corey Leath

Description:
- Loads engineered features
- Trains forecasting model
- Saves trained model to artifacts directory

Input:
- data/processed/features.csv
- configs/train.yaml

Output:
- models/energy_forecast_model.pkl
"""

import pandas as pd
import joblib
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

def main(config_path='configs/train.yaml'):
    # Load config
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load engineered features
    input_path = 'data/processed/features.csv'
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Prepare training data
    print("Preparing training data...")
    feature_cols = config['features']
    target_col = config['target']
    test_size = config['test_size']
    random_state = config['random_state']

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Train model
    print("Training model...")
    model = XGBRegressor(**config['model_params'])
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE on test set: {mape:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/energy_forecast_model.pkl'
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}.")

if __name__ == "__main__":
    main()
