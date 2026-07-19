"""Tests for leakage-safe rolling-origin forecast evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import data_fingerprint, regression_metrics, rolling_origin_backtest
from src.forecasting import generate_demo_data


def test_backtest_is_reproducible_and_time_ordered() -> None:
    frame = generate_demo_data(periods=24 * 30, seed=17)
    first = rolling_origin_backtest(frame, max_folds=8)
    second = rolling_origin_backtest(frame, max_folds=8)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8 * 24
    assert (first["origin_timestamp"] < first["timestamp"]).all()


def test_backtest_metrics_are_finite() -> None:
    frame = generate_demo_data(periods=24 * 30, seed=17)
    result = rolling_origin_backtest(frame, max_folds=8)
    metrics = regression_metrics(result, history=frame)

    assert metrics["samples"] == 8 * 24
    assert metrics["folds"] == 8
    for name in ("mae", "rmse", "mape_percent", "smape_percent", "mase", "bias", "r2"):
        assert np.isfinite(metrics[name])
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]


def test_fingerprint_is_deterministic_and_content_sensitive() -> None:
    frame = generate_demo_data(periods=96, seed=5)
    original = data_fingerprint(frame)
    assert original == data_fingerprint(frame.copy())

    modified = frame.copy()
    modified.loc[0, "demand"] += 1
    assert data_fingerprint(modified) != original


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "positive"),
        ({"step": 0}, "positive"),
        ({"max_folds": 0}, "max_folds"),
    ],
)
def test_invalid_backtest_configuration(kwargs, message) -> None:
    frame = generate_demo_data(periods=24 * 30)
    with pytest.raises(ValueError, match=message):
        rolling_origin_backtest(frame, **kwargs)
