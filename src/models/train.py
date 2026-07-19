"""Train a reproducible XGBoost energy-demand model and emit provenance evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(config_path: str = "configs/train.yaml") -> None:
    """Train with a chronological holdout and save a model-evidence sidecar."""

    config_file = Path(config_path)
    with config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    input_path = Path("data/processed/features.csv")
    df = pd.read_csv(input_path, parse_dates=["timestamp"]).sort_values("timestamp")
    feature_cols = config["features"]
    target_col = config["target"]
    random_state = int(config["random_state"])

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df[target_col],
        test_size=float(config["test_size"]),
        shuffle=False,
    )

    model_params = dict(config["model_params"])
    model_params.setdefault("random_state", random_state)
    model_params.setdefault("n_jobs", 1)
    model_params.setdefault("objective", "reg:squarederror")

    model = XGBRegressor(**model_params)
    started = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - started
    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mape_percent": float(mean_absolute_percentage_error(y_test, y_pred) * 100.0),
        "r2": float(r2_score(y_test, y_pred)),
    }

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "energy_forecast_model.pkl"
    metadata_path = output_dir / "energy_forecast_model.metadata.json"
    joblib.dump(model, model_path)

    train_rows = len(X_train)
    metadata = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "xgboost.XGBRegressor",
        "estimator_parameters": model_params,
        "seed": random_state,
        "features": feature_cols,
        "target": target_col,
        "split": {
            "strategy": "chronological_holdout",
            "shuffle": False,
            "train_rows": train_rows,
            "test_rows": len(X_test),
            "train_end": pd.Timestamp(df["timestamp"].iloc[train_rows - 1]).isoformat(),
            "test_start": pd.Timestamp(df["timestamp"].iloc[train_rows]).isoformat(),
        },
        "metrics": metrics,
        "training_seconds": training_seconds,
        "artifacts": {
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "config_path": str(config_file),
            "config_sha256": _sha256(config_file),
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"Saved model to {model_path} and evidence to {metadata_path}.")


if __name__ == "__main__":
    main()
