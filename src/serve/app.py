"""
app.py

Module: FastAPI Service for Energy Demand Forecasting
Author: Corey Leath

Description:
- Loads trained model on startup
- Provides REST API endpoint for inference

Endpoint:
/predict
    Input: JSON with feature values
    Output: Predicted energy demand (MW)

Example input:
{
    "load_ma_3h": 1234.5,
    "temperature_ma_3h": 22.1
}

Example output:
{
    "predicted_load": 1250.3
}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from typing import Dict

MODEL_PATH = os.environ.get("MODEL_PATH", "models/energy_forecast_model.pkl")

# Define FastAPI app
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="Serve trained energy forecasting model via REST API",
    version="1.0.0"
)

# Lazy-loaded model singleton
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Train the model first."
            )
        print(f"Loading model from {MODEL_PATH}...")
        _model = joblib.load(MODEL_PATH)
    return _model


# Define request schema
class PredictionRequest(BaseModel):
    load_ma_3h: float
    temperature_ma_3h: float

# Define response schema
class PredictionResponse(BaseModel):
    predicted_load: float

# Health check endpoint required by Kubernetes liveness/readiness probes
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# Define /predict endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()

    # Prepare input
    features = np.array([
        [
            request.load_ma_3h,
            request.temperature_ma_3h
        ]
    ])

    # Run inference
    prediction = model.predict(features)[0]

    # Return response
    return PredictionResponse(predicted_load=float(prediction))
