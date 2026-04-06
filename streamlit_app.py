import os

import streamlit as st
import numpy as np
import pandas as pd
import yaml
import torch

from src.utils import load_config, load_data, load_scaler
from src.predict import predict_multi
from src.model import build_model, get_device

st.title("⚡ Energy Demand Forecasting Dashboard")

try:
    config = load_config("config/config.yaml")
except FileNotFoundError as e:
    st.error(f"Configuration file not found: {e}")
    st.stop()

data_path = config.get("data", {}).get("path", config.get("data", {}).get("raw_path", "data/raw/energy.csv"))

try:
    df = load_data(data_path)
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}. Please run the data pipeline first.")
    st.stop()

try:
    scaler = load_scaler("models/scaler.pkl")
except FileNotFoundError:
    st.warning("Scaler not found at models/scaler.pkl. Predictions will not be inverse-transformed.")
    scaler = None

seq_len = config.get("model", {}).get("sequence_length", config.get("data", {}).get("sequence_length", 48))
device = get_device()

# Load latest model checkpoint
checkpoint_dir = "checkpoints"
model = None
if os.path.isdir(checkpoint_dir):
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") or f.endswith(".pth")])
    if checkpoints:
        model_path = os.path.join(checkpoint_dir, checkpoints[-1])
        try:
            model = build_model(config, df.shape[1], config.get("model", {}).get("forecast_horizon", 24))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            st.success(f"Loaded model: {checkpoints[-1]}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.warning("No model checkpoints found in checkpoints/. Please train a model first.")
else:
    st.warning("Checkpoints directory not found. Please train a model first.")

st.subheader("Latest Data Preview")
st.write(df.tail(10))

steps = st.slider("Forecast Horizon (hours)", 1, 72, 24)

if st.button("Run Forecast"):
    if model is None:
        st.error("No model loaded. Cannot run forecast.")
    elif len(df) < seq_len:
        st.error(f"Not enough data: need at least {seq_len} rows, got {len(df)}.")
    else:
        last_seq = df.values[-seq_len:]
        preds = predict_multi(model, last_seq, steps, scaler, device)

        st.subheader("Forecast Output")
        st.line_chart(preds)

