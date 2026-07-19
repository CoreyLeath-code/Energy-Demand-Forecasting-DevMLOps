#!/usr/bin/env python3
"""Reproducible benchmark for the public seasonal-naive forecast."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation import data_fingerprint, regression_metrics, rolling_origin_backtest
from src.forecasting import ForecastSettings, generate_demo_data, seasonal_naive_forecast


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def run(iterations: int, warmup: int) -> dict[str, object]:
    seed = 20260719
    frame = generate_demo_data(periods=24 * 90, seed=seed)
    settings = ForecastSettings(horizon=24, seasonal_period=24)
    evidence = rolling_origin_backtest(
        frame, horizon=24, seasonal_period=24, initial_train_size=24 * 14,
        step=24, max_folds=30,
    )
    quality = regression_metrics(evidence, seasonal_period=24)

    for _ in range(warmup):
        seasonal_naive_forecast(frame, settings)

    durations: list[float] = []
    tracemalloc.start()
    for _ in range(iterations):
        started = perf_counter_ns()
        seasonal_naive_forecast(frame, settings)
        durations.append((perf_counter_ns() - started) / 1_000_000)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    scaling = []
    for rows in (168, 720, 2160, 8760):
        sample = generate_demo_data(periods=rows, seed=seed)
        started = perf_counter_ns()
        seasonal_naive_forecast(sample, settings)
        scaling.append({"history_rows": rows, "latency_ms": (perf_counter_ns() - started) / 1_000_000})

    mean_ms = statistics.fmean(durations)
    return {
        "schema_version": "1.0",
        "benchmark": "seasonal-naive-public-baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "protocol": {
            "dataset": "seeded synthetic hourly demand; 2,160 observations",
            "evaluation": "expanding-window rolling-origin; 30 non-overlapping 24-hour folds",
            "initial_train_rows": 336,
            "forecast_horizon": 24,
            "seasonal_period": 24,
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
        },
        "data_sha256": data_fingerprint(frame),
        "quality": quality,
        "latency_ms": {
            "mean": mean_ms,
            "median": statistics.median(durations),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
            "minimum": min(durations),
            "maximum": max(durations),
        },
        "throughput": {
            "forecasts_per_second": 1000.0 / mean_ms,
            "forecasted_hours_per_second": 24_000.0 / mean_ms,
        },
        "peak_python_memory_mib": peak_bytes / (1024 * 1024),
        "scaling": scaling,
        "limitations": [
            "Synthetic data is not evidence of utility-grid performance.",
            "Latency is runner-dependent and excludes network and UI overhead.",
            "Quality describes the transparent seasonal-naive baseline, not trained deep-learning models.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.iterations < 10 or args.warmup < 0:
        parser.error("iterations must be >= 10 and warmup must be >= 0")
    result = run(args.iterations, args.warmup)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
