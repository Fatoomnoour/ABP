"""Input validation and preprocessing for ABP inference."""

from __future__ import annotations

import numpy as np


class InputValidationError(ValueError):
    """Raised when an inference payload does not match the model contract."""


def _signal_array(values: object, name: str, sample_size: int) -> np.ndarray:
    """Validate one signal and return a finite float array of fixed length."""

    if not isinstance(values, (list, tuple)):
        raise InputValidationError(f"'{name}' must be a JSON list of numbers")

    if len(values) != sample_size:
        raise InputValidationError(
            f"'{name}' must contain exactly {sample_size} samples"
        )

    try:
        signal = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"'{name}' must contain only numeric values"
        ) from exc

    if not np.isfinite(signal).all():
        raise InputValidationError(f"'{name}' must contain only finite values")

    return signal


def prepare_features(
    payload: dict, scaler_x: object, sample_size: int = 250
) -> np.ndarray:
    """Validate PPG/ECG JSON and apply the training-time feature scaling."""

    if not isinstance(payload, dict):
        raise InputValidationError("Request body must be a JSON object")

    missing = [name for name in ("ppg", "ecg") if name not in payload]
    if missing:
        raise InputValidationError(f"Missing required field(s): {', '.join(missing)}")

    ppg = _signal_array(payload["ppg"], "ppg", sample_size)
    ecg = _signal_array(payload["ecg"], "ecg", sample_size)
    features = np.stack((ppg, ecg), axis=-1)
    flattened = features.reshape(1, -1)
    return scaler_x.transform(flattened).reshape(1, sample_size, 2)
