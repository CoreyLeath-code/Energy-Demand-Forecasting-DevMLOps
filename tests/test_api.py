from fastapi.testclient import TestClient
from src.serve.app import app

client = TestClient(app)


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"load_ma_3h": 1234.5, "temperature_ma_3h": 22.1},
    )
    assert response.status_code == 200
    assert "predicted_load" in response.json()
