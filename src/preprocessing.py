import numpy as np
from typing import List, Optional
import os

# This file must contain the same preprocessing used in training.
# Two common approaches:
# 1) Load a saved scaler (scikit-learn StandardScaler/MinMaxScaler) from scaler.joblib
# 2) If no scaler exists, apply minimal checks and return numpy arrays

SCALER_PATH = "scaler.pkl"


def load_scaler_if_exists():
    import joblib
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def preprocess_single(features: List[float], scaler=None):
    x = np.array(features, dtype=float)
    # If the scaler was provided, apply transform
    if scaler is not None:
        try:
            x = scaler.transform(x.reshape(1, -1)).reshape(-1)
        except Exception:
            # fallback: try transform on array
            x = scaler.transform(np.array([x]))[0]
    # else: no scaler — the user must ensure the input is in the expected range
    return x


def preprocess_batch(data: List[List[float]], scaler=None):
    X = np.array(data, dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    return X