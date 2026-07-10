"""Contract and failure-mode tests for the energy forecasting API."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serve.app as app_module

client = TestClient(app_module.app)


@pytest.fixture(autouse=True)
def clear_cached_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep every test isolated from local model artifacts."""

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-model.pkl"))
    app_module.clear_model_cache()
    yield
    app_module.clear_model_cache()


def test_root_contract() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Energy-Demand-Forecasting-DevMLOps"
    assert "version" in payload


def test_health_endpoint_reports_optional_model_state() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_available"] is False


def test_predict_uses_deterministic_fallback_without_artifact() -> None:
    response = client.post(
        "/predict",
        json={"load_ma_3h": 1234.5, "temperature_ma_3h": 22.0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "deterministic-baseline"
    assert payload["predicted_load"] == pytest.approx(1239.5)
    assert "model_version" in payload


def test_predict_uses_trained_model_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.predict.return_value = np.array([1500.25])
    monkeypatch.setattr(app_module, "get_model", lambda: model)

    response = client.post(
        "/predict",
        json={"load_ma_3h": 1234.5, "temperature_ma_3h": 22.1},
    )

    assert response.status_code == 200
    assert response.json()["predicted_load"] == pytest.approx(1500.25)
    assert response.json()["backend"] == "trained-model"
    model.predict.assert_called_once()


def test_predict_rejects_missing_feature() -> None:
    response = client.post("/predict", json={"load_ma_3h": 1000.0})

    assert response.status_code == 422


def test_predict_rejects_out_of_range_values() -> None:
    response = client.post(
        "/predict",
        json={"load_ma_3h": -1.0, "temperature_ma_3h": 22.0},
    )

    assert response.status_code == 422


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "energy_forecast_latency_seconds" in response.text


def test_invalid_model_output_returns_service_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.predict.return_value = np.array([np.nan])
    monkeypatch.setattr(app_module, "get_model", lambda: model)

    response = client.post(
        "/predict",
        json={"load_ma_3h": 1200.0, "temperature_ma_3h": 20.0},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Forecasting backend unavailable."
