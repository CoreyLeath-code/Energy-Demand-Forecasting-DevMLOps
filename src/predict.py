"""
Energy-Demand-Forecasting-DevMLOps
Prediction Utilities

Author: Corey Leath (Trojan3877)

This module provides:
✔ Multi-step forecasting with LSTM/GRU/Transformer models
✔ Inverse scaling of predictions
"""

import numpy as np
import torch


def predict_multi(model, last_seq, steps, scaler, device):
    """
    Generate multi-step energy demand forecasts.

    Args:
        model: Trained PyTorch model (LSTM/GRU/Transformer)
        last_seq: numpy array of shape (seq_len, n_features) — most recent sequence
        steps: int — number of forecast steps (hours ahead)
        scaler: fitted MinMaxScaler (or None if not scaled)
        device: torch.device

    Returns:
        numpy array of shape (steps,) — predicted energy demand values
    """
    model.eval()
    predictions = []

    seq = last_seq.copy().astype(np.float32)

    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x)
            pred_val = pred.cpu().numpy().flatten()[0]
            predictions.append(pred_val)

            # Roll the window: drop oldest, append new prediction
            new_row = seq[-1].copy()
            new_row[0] = pred_val
            seq = np.vstack([seq[1:], new_row])

    predictions = np.array(predictions)

    if scaler is not None:
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, 0] = predictions
        predictions = scaler.inverse_transform(dummy)[:, 0]

    return predictions
