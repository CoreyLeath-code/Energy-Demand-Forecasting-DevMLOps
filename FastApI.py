"""
FastApI.py

FastAPI entry-point for the Energy Demand Forecasting API.
Delegates to src.serve.app — run with:
    uvicorn FastApI:app --reload --port 8000
"""

from src.serve.app import app  # noqa: F401 – re-export the app instance
