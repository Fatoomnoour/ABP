"""Container and local entry point for the ABP estimation API."""

from __future__ import annotations

import logging

from src.abp_service.app import create_app, start_ngrok
from src.abp_service.config import Settings

settings = Settings.from_env()
app = create_app(settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_ngrok(settings)
    app.run(debug=False, host=settings.host, port=settings.port)
