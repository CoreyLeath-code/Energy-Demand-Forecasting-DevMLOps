"""
Energy-Demand-Forecasting-DevMLOps
L6 Production Utilities Module

Author: Corey Leath (Trojan3877)

This file provides:
✔ Config loading
✔ Logging utilities
✔ Directory creation
✔ Random seed setting for reproducibility
✔ Scaler save/load utilities
✔ Environment (GPU/CPU) reporting
✔ Timestamp generator
"""

import os
import yaml
import json
import joblib
import random
import numpy as np
from datetime import datetime

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# -----------------------------------------------------------
# Load YAML Configuration
# -----------------------------------------------------------
def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"[ERROR] YAML parsing error: {e}")

    return config


# -----------------------------------------------------------
# Ensure Directory Exists
# -----------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")


# -----------------------------------------------------------
# Set Global Random Seed
# -----------------------------------------------------------
def set_seed(seed=42):
    print(f"[INFO] Setting global seed to: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    if _TF_AVAILABLE:
        tf.random.set_seed(seed)


# -----------------------------------------------------------
# Save Preprocessing Scaler
# -----------------------------------------------------------
def save_scaler(scaler, path="models/scaler.pkl"):
    ensure_dir(os.path.dirname(path))
    joblib.dump(scaler, path)
    print(f"[INFO] Saved scaler → {path}")


# -----------------------------------------------------------
# Load Preprocessing Scaler
# -----------------------------------------------------------
def load_scaler(path="models/scaler.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Scaler not found: {path}")

    print(f"[INFO] Loaded scaler from: {path}")
    return joblib.load(path)


# -----------------------------------------------------------
# JSON Helper (for exporting configs, results, metadata)
# -----------------------------------------------------------
def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Saved JSON → {path}")


# -----------------------------------------------------------
# Timestamp Helper
# -----------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# -----------------------------------------------------------
# GPU / CPU Environment Report
# -----------------------------------------------------------
def print_gpu_info():
    if _TF_AVAILABLE:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[INFO] GPU detected: {gpus}")
        else:
            print("[INFO] No GPU detected — running on CPU.")
    else:
        print("[INFO] TensorFlow not available — GPU detection skipped.")


# -----------------------------------------------------------
# Write Training Header to Console
# -----------------------------------------------------------
def print_training_header(config):
    print("\n" + "=" * 60)
    print(" ENERGY DEMAND FORECASTING — TRAINING SESSION STARTED ")
    print("=" * 60)
    print(f"Timestamp: {timestamp()}")
    print(f"Model Type: {config['model']['type']}")
    print(f"Sequence Length: {config['model']['sequence_length']}")
    print(f"Forecast Horizon: {config['model']['forecast_horizon']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print("=" * 60 + "\n")
