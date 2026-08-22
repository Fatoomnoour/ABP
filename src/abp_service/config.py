"""Environment-backed configuration for the ABP estimation service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    model_path: Path
    scaler_x_path: Path
    scaler_y_path: Path
    sample_size: int
    enable_ngrok: bool
    ngrok_auth_token: str | None
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with safe local defaults."""

        return cls(
            model_path=_resolve_path(
                os.getenv("MODEL_PATH", "models/CNN_LSTM_Model_256.h5"),
                PROJECT_ROOT / "models/CNN_LSTM_Model_256.h5",
            ),
            scaler_x_path=_resolve_path(
                os.getenv("SCALER_X_PATH", "models/scaler_X.pkl"),
                PROJECT_ROOT / "models/scaler_X.pkl",
            ),
            scaler_y_path=_resolve_path(
                os.getenv("SCALER_Y_PATH", "models/scaler_y.pkl"),
                PROJECT_ROOT / "models/scaler_y.pkl",
            ),
            sample_size=int(os.getenv("SAMPLE_SIZE", "250")),
            enable_ngrok=os.getenv("ENABLE_NGROK", "false").lower()
            in {"1", "true", "yes"},
            ngrok_auth_token=os.getenv("NGROK_AUTHTOKEN") or None,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
        )
