"""Model loading and prediction orchestration."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import prepare_features

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Loaded model and preprocessing artifacts used for inference."""

    model: Any
    scaler_x: Any
    scaler_y: Any
    sample_size: int = 250

    def predict(self, payload: dict) -> np.ndarray:
        """Validate, scale, infer, and inverse-transform one request."""

        features = prepare_features(payload, self.scaler_x, self.sample_size)
        scaled_prediction = self.model.predict(features, verbose=0)
        return self.scaler_y.inverse_transform(scaled_prediction)


def load_bundle(
    model_path: Path,
    scaler_x_path: Path,
    scaler_y_path: Path,
    sample_size: int = 250,
) -> ModelBundle:
    """Load the trained Keras model and its fitted scalers from disk."""

    try:
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.models import load_model

        logger.info("Loading model from %s", model_path)
        model = load_model(
            model_path,
            custom_objects={"mse": MeanSquaredError()},
            compile=False,
        )
        with scaler_x_path.open("rb") as file:
            scaler_x = pickle.load(file)
        with scaler_y_path.open("rb") as file:
            scaler_y = pickle.load(file)
    except Exception:
        logger.exception("Failed to load model artifacts")
        raise

    logger.info("Model and preprocessing artifacts loaded successfully")
    return ModelBundle(model, scaler_x, scaler_y, sample_size)
