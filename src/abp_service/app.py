"""Flask application for arterial blood pressure estimation."""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, request

from .config import Settings
from .model_service import ModelBundle, load_bundle
from .preprocessing import InputValidationError

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    model_bundle: ModelBundle | None = None,
    load_model_on_startup: bool = True,
) -> Flask:
    """Create and configure the Flask application.

    ``model_bundle`` is injectable so API tests can run without downloading or
    loading the large production model artifact.
    """

    settings = settings or Settings.from_env()
    bundle = model_bundle
    if bundle is None and load_model_on_startup:
        bundle = load_bundle(
            settings.model_path,
            settings.scaler_x_path,
            settings.scaler_y_path,
            settings.sample_size,
        )

    app = Flask(__name__)
    app.config["ABP_SETTINGS"] = settings
    app.extensions["abp_model"] = bundle

    @app.get("/")
    def home() -> Any:
        """Return the service status and inference contract."""

        return jsonify(
            {
                "service": "ABP Estimation API",
                "status": "ok",
                "model_loaded": app.extensions["abp_model"] is not None,
                "endpoints": {
                    "/predict": {
                        "method": "POST",
                        "description": "Predict mean ABP from 250 PPG and ECG samples",
                        "input_format": {
                            "ppg": f"list of {settings.sample_size} numeric values",
                            "ecg": f"list of {settings.sample_size} numeric values",
                        },
                    }
                },
            }
        )

    @app.post("/predict")
    def predict() -> Any:
        """Validate a signal window and return the predicted mean ABP."""

        bundle = app.extensions["abp_model"]
        if bundle is None:
            logger.error("Prediction requested before the model was loaded")
            return jsonify({"error": "Model is not available"}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        try:
            prediction = bundle.predict(payload)
        except InputValidationError as exc:
            logger.warning("Invalid prediction request: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("Prediction failed")
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify(
            {
                "predicted_mean_abp": float(prediction.reshape(-1)[0]),
                "unit": "mmHg",
            }
        )

    return app


def start_ngrok(settings: Settings) -> str | None:
    """Start an ngrok tunnel only when explicitly enabled by environment."""

    if not settings.enable_ngrok:
        logger.info("ngrok is disabled; serving locally")
        return None
    if not settings.ngrok_auth_token:
        raise RuntimeError(
            "ENABLE_NGROK is true but NGROK_AUTHTOKEN is not configured"
        )

    from pyngrok import ngrok

    ngrok.set_auth_token(settings.ngrok_auth_token)
    tunnel = ngrok.connect(settings.port)
    logger.info("ngrok tunnel started at %s", tunnel.public_url)
    return tunnel.public_url


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runtime_settings = Settings.from_env()
    start_ngrok(runtime_settings)
    application = create_app(runtime_settings)
    application.run(
        debug=False,
        host=runtime_settings.host,
        port=runtime_settings.port,
    )
