"""
Centralized logging configuration for the ABP Estimation API.
"""

import logging
import os
import sys


def setup_logger(name: str = "abp_api", level: str = None) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__ of the calling module).
        level: Log level string. Defaults to LOG_LEVEL env var or INFO.

    Returns:
        Configured logger instance.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level, logging.INFO))
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
