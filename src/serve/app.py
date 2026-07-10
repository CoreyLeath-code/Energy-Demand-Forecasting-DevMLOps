"""Production-oriented FastAPI service for energy-demand forecasting.

The service lazily loads an optional trained model and provides a transparent,
deterministic fallback when model artifacts are not present. This keeps health
checks, local development, CI, and portfolio demonstrations operational without
misrepresenting which inference backend produced a prediction.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import joblib
import numpy as np
import prometheus_client as prom
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

APP_VERSION: Final[str] = "1.1.0"
DEFAULT_MODEL_PATH: Final[str] = "models/energy_forecast_model.pkl"

REQUEST_COUNTER = Counter(
    "energy_forecast_requests_total",
    "Total energy-demand forecast requests.",
    ["backend"],
)
ERROR_COUNTER = Counter(
    "energy_forecast_errors_total",
    "Total energy-demand forecast errors.",
)
LATENCY_HISTOGRAM = Histogram(
    "energy_forecast_latency_seconds",
    "Energy-demand forecast request latency.",
)

app = FastAPI(
    title="Energy Demand Forecasting API",
    description="Validated energy-demand forecasting with explicit backend provenance.",
    version=APP_VERSION,
)


class PredictionRequest(BaseModel):
    """Validated inference request for engineered rolling features."""

    load_ma_3h: float = Field(
        ge=0.0,
        le=1_000_000.0,
        description="Three-hour moving-average demand in MW.",
    )
    temperature_ma_3h: float = Field(
        ge=-100.0,
        le=100.0,
        description="Three-hour moving-average temperature in degrees Celsius.",
    )


class PredictionResponse(BaseModel):
    """Prediction value and transparent inference provenance."""

    predicted_load: float
    backend: Literal["trained-model", "deterministic-baseline"]
    model_version: str = APP_VERSION


class HealthResponse(BaseModel):
    """Service health and model-artifact readiness metadata."""

    status: Literal["healthy"] = "healthy"
    service: str = "energy-demand-forecasting-api"
    version: str = APP_VERSION
    model_available: bool


def get_model_path() -> Path:
    """Resolve the model path at call time for testability and deployment overrides."""

    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the optional serialized forecasting model."""

    model_path = get_model_path()
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


def clear_model_cache() -> None:
    """Clear the model singleton, primarily for tests and controlled reloads."""

    get_model.cache_clear()


def deterministic_baseline(load_ma_3h: float, temperature_ma_3h: float) -> float:
    """Return a bounded, deterministic fallback demand estimate.

    The fallback preserves the recent load level and applies a modest weather
    sensitivity based on distance from a 20 °C comfort point. It is not a
    replacement for the trained model and is identified in every response.
    """

    weather_adjustment = abs(temperature_ma_3h - 20.0) * 2.5
    return max(float(load_ma_3h + weather_adjustment), 0.0)


def run_prediction(request: PredictionRequest) -> tuple[float, Literal["trained-model", "deterministic-baseline"]]:
    """Run trained-model inference or the deterministic baseline."""

    model = get_model()
    if model is None:
        return (
            deterministic_baseline(request.load_ma_3h, request.temperature_ma_3h),
            "deterministic-baseline",
        )

    features = np.asarray(
        [[request.load_ma_3h, request.temperature_ma_3h]],
        dtype=np.float64,
    )
    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise ValueError("Model returned an invalid prediction.")
    return max(float(prediction[0]), 0.0), "trained-model"


@app.get("/")
def root() -> dict[str, str]:
    """Return service identity metadata."""

    return {
        "status": "ok",
        "service": "Energy-Demand-Forecasting-DevMLOps",
        "version": APP_VERSION,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a lightweight liveness response for orchestrators."""

    return HealthResponse(model_available=get_model_path().is_file())


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in the standard text format."""

    return Response(prom.generate_latest(), media_type=prom.CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return a validated energy-demand prediction with backend provenance."""

    try:
        with LATENCY_HISTOGRAM.time():
            prediction, backend = run_prediction(request)
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail="Forecasting backend unavailable.") from exc

    REQUEST_COUNTER.labels(backend=backend).inc()
    return PredictionResponse(predicted_load=prediction, backend=backend)
