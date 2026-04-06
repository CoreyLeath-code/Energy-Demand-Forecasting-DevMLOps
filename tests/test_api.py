from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

# Patch the model loader before importing the app so no model file is needed
_mock_model = MagicMock()
_mock_model.predict.return_value = np.array([1234.5])

with patch("src.serve.app.get_model", return_value=_mock_model):
    from src.serve.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    with patch("src.serve.app.get_model", return_value=_mock_model):
        response = client.post(
            "/predict",
            json={"load_ma_3h": 1234.5, "temperature_ma_3h": 22.1},
        )
    assert response.status_code == 200
    assert "predicted_load" in response.json()


def test_predict_endpoint_invalid_input():
    response = client.post("/predict", json={"load_ma_3h": "not_a_float"})
    assert response.status_code == 422  # Unprocessable Entity
