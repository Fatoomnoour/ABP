"""
Preprocessing utilities for ABP estimation.
Handles signal validation, normalization, and input preparation.
"""

import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 250


def load_scalers(scaler_x_path: str, scaler_y_path: str):
    """Load pre-fitted scalers from disk."""
    with open(scaler_x_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)
    logger.info("Scalers loaded.")
    return scaler_X, scaler_y


def validate_signal(signal: list, name: str, expected_size: int = SAMPLE_SIZE) -> np.ndarray:
    """
    Validate and convert a biosignal list to numpy array.

    Args:
        signal: Raw signal as list of floats.
        name: Signal name for error messages ('ppg' or 'ecg').
        expected_size: Expected number of samples.

    Returns:
        Validated numpy array of shape (expected_size,).

    Raises:
        ValueError: If signal size doesn't match expected.
    """
    arr = np.array(signal, dtype=np.float32)
    if arr.shape[0] != expected_size:
        raise ValueError(
            f"'{name}' must contain {expected_size} samples, got {arr.shape[0]}."
        )
    return arr


def prepare_input(ppg: np.ndarray, ecg: np.ndarray, scaler_X) -> np.ndarray:
    """
    Stack and scale PPG + ECG signals for model input.

    Args:
        ppg: PPG signal array of shape (250,).
        ecg: ECG signal array of shape (250,).
        scaler_X: Fitted StandardScaler for input.

    Returns:
        Scaled input array of shape (1, 250, 2).
    """
    ppg = ppg.reshape(1, SAMPLE_SIZE)
    ecg = ecg.reshape(1, SAMPLE_SIZE)

    # Stack to (1, 250, 2) — 2 channels: PPG and ECG
    X = np.stack((ppg, ecg), axis=-1)
    logger.debug(f"Stacked input shape: {X.shape}")

    X_scaled = scaler_X.transform(X.reshape(1, -1)).reshape(1, SAMPLE_SIZE, 2)
    logger.debug(f"Scaled input shape: {X_scaled.shape}")

    return X_scaled


def postprocess_output(prediction_scaled: np.ndarray, scaler_y) -> np.ndarray:
    """
    Inverse-transform model output to original ABP scale (mmHg).

    Args:
        prediction_scaled: Raw model output.
        scaler_y: Fitted StandardScaler for output.

    Returns:
        ABP values in mmHg.
    """
    return scaler_y.inverse_transform(prediction_scaled)
