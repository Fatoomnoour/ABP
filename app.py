"""Backward-compatible launcher for the refactored ABP service."""

from src.abp_service.app import create_app, start_ngrok
from src.abp_service.config import Settings

settings = Settings.from_env()
app = create_app(settings)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    start_ngrok(settings)
    app.run(debug=False, host=settings.host, port=settings.port)
