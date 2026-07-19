"""Time-aware, reproducible evaluation for energy-demand forecasts."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting import (
    DEFAULT_DEMAND_COLUMN,
    DEFAULT_TIMESTAMP_COLUMN,
    ForecastSettings,
    normalize_energy_frame,
    seasonal_naive_forecast,
)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of canonical timestamps and demand."""
    normalized = normalize_energy_frame(frame)
    canonical = normalized[[DEFAULT_TIMESTAMP_COLUMN, DEFAULT_DEMAND_COLUMN]].copy()
    canonical[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(
        canonical[DEFAULT_TIMESTAMP_COLUMN]
    ).map(lambda value: value.isoformat())
    payload = canonical.to_csv(index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rolling_origin_backtest(
    frame: pd.DataFrame,
    *,
    horizon: int = 24,
    seasonal_period: int = 24,
    initial_train_size: int = 24 * 7,
    step: int = 24,
    max_folds: int | None = 30,
) -> pd.DataFrame:
    """Evaluate forecasts using expanding-window origins without future leakage."""
    normalized = normalize_energy_frame(frame)
    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive.")
    minimum = max(initial_train_size, seasonal_period * 2)
    final_origin = len(normalized) - horizon
    if final_origin < minimum:
        raise ValueError("insufficient observations for the requested backtest.")

    origins = list(range(minimum, final_origin + 1, step))
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be positive when provided.")
        origins = origins[-max_folds:]

    rows: list[pd.DataFrame] = []
    settings = ForecastSettings(horizon=horizon, seasonal_period=seasonal_period)
    for fold, origin in enumerate(origins):
        training = normalized.iloc[:origin]
        actual = normalized.iloc[origin : origin + horizon]
        forecast = seasonal_naive_forecast(training, settings)
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "origin_timestamp": training[DEFAULT_TIMESTAMP_COLUMN].iloc[-1],
                    "timestamp": actual[DEFAULT_TIMESTAMP_COLUMN].to_numpy(),
                    "actual": actual[DEFAULT_DEMAND_COLUMN].to_numpy(dtype=float),
                    "prediction": forecast["forecast"].to_numpy(dtype=float),
                }
            )
        )

    result = pd.concat(rows, ignore_index=True)
    result["error"] = result["prediction"] - result["actual"]
    return result


def regression_metrics(
    backtest: pd.DataFrame,
    *,
    history: pd.DataFrame,
    seasonal_period: int = 24,
) -> dict[str, Any]:
    """Compute scale-aware forecast metrics from out-of-sample predictions."""
    actual = backtest["actual"].to_numpy(dtype=float)
    predicted = backtest["prediction"].to_numpy(dtype=float)
    errors = predicted - actual
    absolute = np.abs(errors)
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(absolute[nonzero] / np.abs(actual[nonzero])) * 100)
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_mask = smape_denominator > 1e-12
    smape = float(
        np.mean(2.0 * absolute[smape_mask] / smape_denominator[smape_mask]) * 100
    )

    normalized = normalize_energy_frame(history)
    demand = normalized[DEFAULT_DEMAND_COLUMN].to_numpy(dtype=float)
    seasonal_naive_errors = np.abs(demand[seasonal_period:] - demand[:-seasonal_period])
    scale = float(np.mean(seasonal_naive_errors))
    mase = float(np.mean(absolute) / scale) if scale > 0 else float("nan")

    by_fold = (
        backtest.assign(abs_error=np.abs(backtest["error"]))
        .groupby("fold", sort=True)["abs_error"]
        .mean()
    )
    return {
        "samples": int(len(actual)),
        "folds": int(backtest["fold"].nunique()),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape_percent": mape,
        "smape_percent": smape,
        "mase": mase,
        "bias": float(np.mean(errors)),
        "r2": float(1.0 - np.sum(errors**2) / np.sum((actual - np.mean(actual)) ** 2)),
        "fold_mae_mean": float(by_fold.mean()),
        "fold_mae_std": float(by_fold.std(ddof=1)) if len(by_fold) > 1 else 0.0,
    }
