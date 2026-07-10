"""Tests for the lightweight, deterministic forecasting service."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting import (
    ForecastSettings,
    generate_demo_data,
    normalize_energy_frame,
    seasonal_naive_forecast,
    summarize_dataset,
)


def test_demo_data_is_reproducible() -> None:
    first = generate_demo_data(periods=96, seed=7)
    second = generate_demo_data(periods=96, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["timestamp", "demand", "temperature", "is_weekend"]


def test_normalize_energy_frame_sorts_and_aggregates_duplicates() -> None:
    frame = generate_demo_data(periods=48)
    duplicated = pd.concat([frame.iloc[::-1], frame.iloc[[0]]], ignore_index=True)

    normalized = normalize_energy_frame(duplicated)

    assert len(normalized) == 48
    assert normalized["timestamp"].is_monotonic_increasing
    assert normalized["timestamp"].is_unique


def test_normalize_energy_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_energy_frame(pd.DataFrame({"timestamp": ["2026-01-01"]}))


def test_normalize_energy_frame_rejects_negative_demand() -> None:
    frame = generate_demo_data(periods=48)
    frame.loc[0, "demand"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        normalize_energy_frame(frame)


def test_forecast_contract_is_complete_and_deterministic() -> None:
    frame = generate_demo_data(periods=24 * 14, seed=11)
    settings = ForecastSettings(horizon=48, seasonal_period=24)

    first = seasonal_naive_forecast(frame, settings)
    second = seasonal_naive_forecast(frame, settings)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 48
    assert list(first.columns) == ["timestamp", "forecast", "lower_bound", "upper_bound"]
    assert (first["lower_bound"] <= first["forecast"]).all()
    assert (first["forecast"] <= first["upper_bound"]).all()
    assert (first[["forecast", "lower_bound", "upper_bound"]] >= 0).all().all()


def test_forecast_rejects_insufficient_history() -> None:
    frame = generate_demo_data(periods=48)

    with pytest.raises(ValueError, match="two full seasonal periods"):
        seasonal_naive_forecast(frame, ForecastSettings(horizon=24, seasonal_period=48))


@pytest.mark.parametrize("horizon", [0, 169])
def test_invalid_horizons_are_rejected(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        ForecastSettings(horizon=horizon)


def test_dataset_summary_is_serializable() -> None:
    frame = generate_demo_data(periods=72)

    summary = summarize_dataset(frame)

    assert summary["observations"] == 72
    assert summary["peak_demand"] >= summary["mean_demand"]
    assert isinstance(summary["start"], str)
    assert isinstance(summary["end"], str)
