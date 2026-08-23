"""
Inference pipeline for ABP estimation.
Loads model and runs end-to-end prediction from raw biosignals.
"""

import logging
import os

import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

from src.ml.preprocessing import (
    load_scalers,
    postprocess_output,
    prepare_input,
    validate_signal,
)

logger = logging.getLogger(__name__)


class ABPInferenceEngine:
    """
    End-to-end inference engine for ABP estimation.

    Usage:
        engine = ABPInferenceEngine()
        result = engine.predict(ppg_signal, ecg_signal)
    """

    def __init__(
        self,
        model_path: str = None,
        scaler_x_path: str = None,
        scaler_y_path: str = None,
    ):
        self.model_path = model_path or os.getenv(
            "MODEL_PATH", "models/CNN_LSTM_Model_256.h5"
        )
        self.scaler_x_path = scaler_x_path or os.getenv(
            "SCALER_X_PATH", "models/scaler_X.pkl"
        )
        self.scaler_y_path = scaler_y_path or os.getenv(
            "SCALER_Y_PATH", "models/scaler_y.pkl"
        )

        self._model = None
        self._scaler_X = None
        self._scaler_y = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model weights and scalers from disk."""
        logger.info("Loading CNN-LSTM model...")
        self._model = load_model(
            self.model_path,
            custom_objects={"mse": MeanSquaredError()},
        )
        self._scaler_X, self._scaler_y = load_scalers(
            self.scaler_x_path, self.scaler_y_path
        )
        logger.info("All artifacts loaded successfully.")

    def predict(self, ppg: list, ecg: list) -> np.ndarray:
        """
        Run end-to-end ABP prediction.

        Args:
            ppg: List of 250 PPG float values.
            ecg: List of 250 ECG float values.

        Returns:
            np.ndarray: Predicted ABP waveform in mmHg.
        """
        # Validate
        ppg_arr = validate_signal(ppg, "ppg")
        ecg_arr = validate_signal(ecg, "ecg")

        # Preprocess
        X_scaled = prepare_input(ppg_arr, ecg_arr, self._scaler_X)

        # Inference
        logger.info("Running model inference...")
        prediction_scaled = self._model.predict(X_scaled, verbose=0)

        # Postprocess
        prediction = postprocess_output(prediction_scaled, self._scaler_y)
        logger.info(f"Prediction complete. Output shape: {prediction.shape}")

        return prediction
