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

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model_path = 'models/energy_forecast_model.pkl'
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# Define FastAPI app
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="Serve trained energy forecasting model via REST API",
    version="1.0.0"
)

# Define request schema
class PredictionRequest(BaseModel):
    load_ma_3h: float
    temperature_ma_3h: float

# Define response schema
class PredictionResponse(BaseModel):
    predicted_load: float

# Define /predict endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
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
    return PredictionResponse(predicted_load=prediction)
