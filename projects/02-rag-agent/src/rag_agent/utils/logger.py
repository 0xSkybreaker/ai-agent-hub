"""Structured logging via loguru."""

from __future__ import annotations

import sys

from loguru import logger

from rag_agent.config import settings

# Remove default handler
logger.remove()

# Add console handler with configured level
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# File handler for persistent logs (stored in logs/ directory)
logger.add(
    "logs/rag_agent.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
)


def get_logger():
    """Return the configured loguru logger instance."""
    return logger
