"""
Unit tests for preprocessing and inference pipeline.
Run with: pytest tests/test_inference.py -v
"""

import numpy as np
import pytest

from src.ml.preprocessing import prepare_input, validate_signal

# ─── validate_signal ──────────────────────────────────────────────────────────

def test_validate_signal_correct_size():
    signal = np.random.randn(250).tolist()
    result = validate_signal(signal, "ppg")
    assert result.shape == (250,)
    assert result.dtype == np.float32


def test_validate_signal_wrong_size():
    signal = np.random.randn(100).tolist()
    with pytest.raises(ValueError, match="250"):
        validate_signal(signal, "ppg")


def test_validate_signal_empty():
    with pytest.raises(ValueError):
        validate_signal([], "ecg")


# ─── prepare_input ────────────────────────────────────────────────────────────

class MockScaler:
    """Minimal scaler mock — returns input unchanged."""
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X


def test_prepare_input_output_shape():
    ppg = np.random.randn(250).astype(np.float32)
    ecg = np.random.randn(250).astype(np.float32)
    scaler = MockScaler()
    result = prepare_input(ppg, ecg, scaler)
    assert result.shape == (1, 250, 2), f"Expected (1, 250, 2), got {result.shape}"


def test_prepare_input_channel_order():
    """PPG should be channel 0, ECG should be channel 1."""
    ppg = np.ones(250, dtype=np.float32) * 2.0
    ecg = np.ones(250, dtype=np.float32) * 5.0
    scaler = MockScaler()
    prepare_input(ppg, ecg, scaler)
    # After reshape/transform, check original stacking order
    stacked = np.stack((ppg.reshape(1, 250), ecg.reshape(1, 250)), axis=-1)
    assert stacked.shape == (1, 250, 2)
    assert stacked[0, 0, 0] == pytest.approx(2.0)
    assert stacked[0, 0, 1] == pytest.approx(5.0)
