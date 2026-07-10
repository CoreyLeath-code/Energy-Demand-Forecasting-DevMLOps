"""Validated, deterministic forecasting utilities for Energy Demand Forecasting.

The functions in this module intentionally avoid heavyweight model dependencies so
that API tests, Streamlit Community Cloud, and constrained CI runners can execute
reliably. Trained-model serving remains available through the dedicated model
pipeline, while this module provides a transparent seasonal-naive baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_TIMESTAMP_COLUMN: Final[str] = "timestamp"
DEFAULT_DEMAND_COLUMN: Final[str] = "demand"
MINIMUM_OBSERVATIONS: Final[int] = 48
MAX_FORECAST_HORIZON: Final[int] = 168


@dataclass(frozen=True)
class ForecastSettings:
    """Configuration for deterministic baseline forecasting."""

    horizon: int = 24
    seasonal_period: int = 24
    confidence_z_score: float = 1.96

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= MAX_FORECAST_HORIZON:
            raise ValueError(
                f"horizon must be between 1 and {MAX_FORECAST_HORIZON}; "
                f"received {self.horizon}."
            )
        if self.seasonal_period < 2:
            raise ValueError("seasonal_period must be at least 2.")
        if self.confidence_z_score <= 0:
            raise ValueError("confidence_z_score must be positive.")


def generate_demo_data(
    periods: int = 24 * 30,
    *,
    seed: int = 42,
    start: str | pd.Timestamp = "2026-01-01",
) -> pd.DataFrame:
    """Generate a reproducible hourly energy-demand dataset for the public demo.

    The signal combines a modest growth trend, daily and weekly seasonality,
    temperature effects, and seeded Gaussian noise. The generated data is
    synthetic and is clearly labeled as such in the Streamlit application.
    """

    if periods < MINIMUM_OBSERVATIONS:
        raise ValueError(f"periods must be at least {MINIMUM_OBSERVATIONS}.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=pd.Timestamp(start), periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    elapsed = np.arange(periods, dtype=float)

    temperature = (
        18.0
        + 7.5 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
        + rng.normal(0.0, 1.2, periods)
    )
    daily_cycle = 145.0 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
    evening_peak = 110.0 * np.exp(-((hour - 19.0) ** 2) / 9.0)
    weekday_effect = np.where(day_of_week < 5, 85.0, -45.0)
    temperature_effect = 5.5 * np.abs(temperature - 20.0)
    trend = elapsed * 0.08
    noise = rng.normal(0.0, 22.0, periods)

    demand = 1_050.0 + daily_cycle + evening_peak + weekday_effect + temperature_effect + trend + noise
    demand = np.maximum(demand, 100.0)

    return pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: timestamps,
            DEFAULT_DEMAND_COLUMN: demand.round(2),
            "temperature": temperature.round(2),
            "is_weekend": (day_of_week >= 5).astype(int),
        }
    )


def normalize_energy_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    demand_column: str = DEFAULT_DEMAND_COLUMN,
) -> pd.DataFrame:
    """Validate and normalize an energy-demand frame.

    Invalid timestamps and non-numeric demand rows are removed. Duplicate
    timestamps are aggregated using the mean, and the result is sorted in
    ascending time order. The returned frame always uses canonical column names.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Energy dataset is empty.")

    missing = [column for column in (timestamp_column, demand_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Energy dataset is missing required columns: {missing}")

    normalized = frame.copy()
    normalized[timestamp_column] = pd.to_datetime(
        normalized[timestamp_column], errors="coerce", utc=False
    )
    normalized[demand_column] = pd.to_numeric(normalized[demand_column], errors="coerce")
    normalized = normalized.dropna(subset=[timestamp_column, demand_column])

    if normalized.empty:
        raise ValueError("Energy dataset contains no valid timestamp and demand rows.")
    if (normalized[demand_column] < 0).any():
        raise ValueError("Energy demand values must be non-negative.")

    numeric_columns = normalized.select_dtypes(include=[np.number]).columns.tolist()
    aggregation = {column: "mean" for column in numeric_columns}
    if demand_column not in aggregation:
        aggregation[demand_column] = "mean"

    normalized = (
        normalized.groupby(timestamp_column, as_index=False)
        .agg(aggregation)
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )
    normalized = normalized.rename(
        columns={timestamp_column: DEFAULT_TIMESTAMP_COLUMN, demand_column: DEFAULT_DEMAND_COLUMN}
    )

    if len(normalized) < MINIMUM_OBSERVATIONS:
        raise ValueError(
            f"At least {MINIMUM_OBSERVATIONS} valid hourly observations are required; "
            f"received {len(normalized)}."
        )

    return normalized


def infer_frequency(frame: pd.DataFrame) -> pd.Timedelta:
    """Infer a stable sampling interval, defaulting to one hour."""

    timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN])
    differences = timestamps.diff().dropna()
    if differences.empty:
        return pd.Timedelta(hours=1)

    positive_differences = differences[differences > pd.Timedelta(0)]
    if positive_differences.empty:
        return pd.Timedelta(hours=1)

    interval = positive_differences.median()
    return interval if interval > pd.Timedelta(0) else pd.Timedelta(hours=1)


def seasonal_naive_forecast(
    frame: pd.DataFrame,
    settings: ForecastSettings | None = None,
) -> pd.DataFrame:
    """Produce a deterministic seasonal-naive forecast with uncertainty bands.

    The forecast repeats the most recent seasonal pattern and applies a bounded
    trend adjustment estimated from the previous seasonal window. Prediction
    intervals are derived from historical seasonal residuals.
    """

    settings = settings or ForecastSettings()
    normalized = normalize_energy_frame(frame)

    if len(normalized) < settings.seasonal_period * 2:
        raise ValueError(
            "At least two full seasonal periods are required for forecasting; "
            f"need {settings.seasonal_period * 2}, received {len(normalized)}."
        )

    demand = normalized[DEFAULT_DEMAND_COLUMN].to_numpy(dtype=float)
    recent = demand[-settings.seasonal_period :]
    previous = demand[-2 * settings.seasonal_period : -settings.seasonal_period]

    raw_trend = float(np.mean(recent) - np.mean(previous))
    scale = max(float(np.mean(recent)), 1.0)
    bounded_trend = float(np.clip(raw_trend, -0.10 * scale, 0.10 * scale))

    seasonal_residuals = demand[settings.seasonal_period :] - demand[: -settings.seasonal_period]
    residual_std = float(np.std(seasonal_residuals, ddof=1)) if len(seasonal_residuals) > 1 else 0.0
    if not np.isfinite(residual_std) or residual_std <= 0.0:
        residual_std = max(float(np.std(recent, ddof=1)), 1.0)

    forecasts: list[float] = []
    for step in range(settings.horizon):
        seasonal_value = recent[step % settings.seasonal_period]
        season_number = (step // settings.seasonal_period) + 1
        forecast_value = seasonal_value + bounded_trend * season_number
        forecasts.append(max(float(forecast_value), 0.0))

    interval = infer_frequency(normalized)
    first_timestamp = pd.Timestamp(normalized[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]) + interval
    forecast_timestamps = pd.date_range(
        start=first_timestamp,
        periods=settings.horizon,
        freq=interval,
    )

    forecast_values = np.asarray(forecasts, dtype=float)
    margin = settings.confidence_z_score * residual_std

    return pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: forecast_timestamps,
            "forecast": forecast_values.round(2),
            "lower_bound": np.maximum(forecast_values - margin, 0.0).round(2),
            "upper_bound": (forecast_values + margin).round(2),
        }
    )


def summarize_dataset(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Return concise, serializable dataset statistics for UIs and APIs."""

    normalized = normalize_energy_frame(frame)
    demand = normalized[DEFAULT_DEMAND_COLUMN]

    return {
        "observations": int(len(normalized)),
        "start": pd.Timestamp(normalized[DEFAULT_TIMESTAMP_COLUMN].iloc[0]).isoformat(),
        "end": pd.Timestamp(normalized[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]).isoformat(),
        "mean_demand": round(float(demand.mean()), 2),
        "peak_demand": round(float(demand.max()), 2),
        "minimum_demand": round(float(demand.min()), 2),
    }
