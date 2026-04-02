import streamlit as st
import numpy as np
import pandas as pd

from src.utils import load_data
from src.predict import predict_multi
from src.model import build_model, get_device
from src.utils import load_scaler
import yaml
import torch

st.title("⚡ Energy Demand Forecasting Dashboard")

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = load_data(config["data"]["path"])
scaler = load_scaler("models/scaler.pkl")

seq_len = config["data"]["sequence_length"]
device = get_device()

# Load model
model_path = sorted([f"checkpoints/{f}" for f in os.listdir("checkpoints")])[-1]
model = build_model(config, df.shape[1], config["data"]["forecast_horizon"])
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.subheader("Latest Data Preview")
st.write(df.tail(10))

steps = st.slider("Forecast Horizon (hours)", 1, 72, 24)

if st.button("Run Forecast"):
    last_seq = df[-seq_len:].values
    preds = predict_multi(model, last_seq, steps, scaler, device)

    st.subheader("Forecast Output")
    st.line_chart(preds)

