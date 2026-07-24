"""Retry decorators using tenacity for resilient API calls."""

from __future__ import annotations

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import (
    APIError,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)

from rag_agent.utils.logger import get_logger

logger = get_logger()

# Retryable exceptions for NVIDIA NIM API
RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)


def create_retry_decorator(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 30.0):
    """Create a retry decorator with configurable parameters.

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.

    Returns:
        A tenacity retry decorator.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.fn.__name__} "
            f"(attempt {retry_state.attempt_number}/{max_attempts}) "
            f"after error: {retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
        ),
        reraise=True,
    )


# Default retry decorator for NVIDIA API calls
nvidia_retry = create_retry_decorator()
