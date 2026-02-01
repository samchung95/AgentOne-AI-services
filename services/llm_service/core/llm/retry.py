"""Retry logic for LLM API calls with exponential backoff.

Handles rate limiting (429), server errors (5xx), and transient failures
with configurable retry strategies.
"""

import asyncio
import random
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog

from services.llm_service.core.config.constants import (
    DEFAULT_MAX_BUFFER_CHUNKS,
    RETRYABLE_STATUS_CODES,
)

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Retry strategy for failed requests."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        initial_delay_seconds: Initial delay before first retry.
        max_delay_seconds: Maximum delay cap for backoff.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delays.
        jitter_factor: Maximum jitter as fraction of delay (0.0-1.0).
        strategy: Retry strategy to use.
        retryable_status_codes: HTTP status codes that should trigger retry.
        retryable_exceptions: Exception types that should trigger retry.
    """

    max_retries: int = 5
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_status_codes: set[int] = field(default_factory=lambda: set(RETRYABLE_STATUS_CODES))
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
    )
    max_buffer_chunks: int = DEFAULT_MAX_BUFFER_CHUNKS


# Default configuration for rate limit handling
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay_seconds=2.0,
    max_delay_seconds=120.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.25,
)


@dataclass
class RetryState:
    """Tracks retry state across attempts.

    Attributes:
        attempt: Current attempt number (0-indexed).
        total_delay: Total time spent waiting.
        last_error: Last error encountered.
        started_at: When retries started.
    """

    attempt: int = 0
    total_delay: float = 0.0
    last_error: Exception | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a given attempt number.

    Args:
        attempt: Current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    if config.strategy == RetryStrategy.FIXED_DELAY:
        delay = config.initial_delay_seconds
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.initial_delay_seconds * (attempt + 1)
    else:  # EXPONENTIAL_BACKOFF
        delay = config.initial_delay_seconds * (config.exponential_base**attempt)

    # Cap at maximum delay
    delay = min(delay, config.max_delay_seconds)

    # Add jitter if enabled
    if config.jitter:
        jitter_amount = delay * config.jitter_factor * random.random()
        delay += jitter_amount

    return delay


def is_retryable_error(
    error: Exception,
    config: RetryConfig,
) -> bool:
    """Check if an error should trigger a retry.

    Args:
        error: The exception to check.
        config: Retry configuration.

    Returns:
        True if the error is retryable.
    """
    # Check for OpenAI-style rate limit errors
    error_str = str(error).lower()
    if "429" in error_str or "rate" in error_str or "quota" in error_str:
        return True

    # Check status code if available (OpenAI errors have status_code attribute)
    status_code = getattr(error, "status_code", None)
    if status_code and status_code in config.retryable_status_codes:
        return True

    # Check exception type
    if isinstance(error, config.retryable_exceptions):
        return True

    return False


def extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value from error if present.

    Args:
        error: The exception to check.

    Returns:
        Retry-After value in seconds, or None if not present.
    """
    # Try to get response headers from various error types
    response = getattr(error, "response", None)
    if response:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Decorator to add retry logic to async functions.

    Args:
        config: Retry configuration. Uses DEFAULT_RETRY_CONFIG if not provided.

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(RetryConfig(max_retries=3))
        async def call_api():
            ...
    """
    config = config or DEFAULT_RETRY_CONFIG

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            state = RetryState()

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    state.last_error = e
                    state.attempt += 1

                    if not is_retryable_error(e, config):
                        logger.warning(
                            "non_retryable_error",
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        raise

                    if state.attempt >= config.max_retries:
                        logger.error(
                            "max_retries_exceeded",
                            attempts=state.attempt,
                            total_delay=state.total_delay,
                            error=str(e),
                        )
                        raise

                    # Check for Retry-After header
                    retry_after = extract_retry_after(e)
                    if retry_after:
                        delay = min(retry_after, config.max_delay_seconds)
                    else:
                        delay = calculate_delay(state.attempt - 1, config)

                    state.total_delay += delay

                    logger.warning(
                        "retrying_after_error",
                        attempt=state.attempt,
                        max_retries=config.max_retries,
                        delay_seconds=round(delay, 2),
                        total_delay=round(state.total_delay, 2),
                        error=str(e)[:200],
                        error_type=type(e).__name__,
                    )

                    await asyncio.sleep(delay)

        return wrapper

    return decorator


async def retry_async_generator[T](
    generator_factory: Callable[[], AsyncGenerator[T, None]],
    config: RetryConfig | None = None,
) -> AsyncGenerator[T, None]:
    """Retry wrapper for async generators with chunk buffering.

    Unlike regular functions, generators can fail mid-iteration.
    This wrapper buffers yielded chunks so they can be replayed on retry.

    Args:
        generator_factory: Function that creates the async generator.
        config: Retry configuration.

    Yields:
        Items from the generator.

    Note:
        Chunks are buffered up to max_buffer_chunks for potential replay.
        Buffer is cleared on successful completion.
    """
    config = config or DEFAULT_RETRY_CONFIG
    state = RetryState()
    chunk_buffer: list[T] = []

    while True:
        try:
            async for item in generator_factory():
                # Buffer the chunk if under limit
                if len(chunk_buffer) < config.max_buffer_chunks:
                    chunk_buffer.append(item)
                yield item
            # Successfully completed - clear buffer
            chunk_buffer.clear()
            return
        except Exception as e:
            state.last_error = e
            state.attempt += 1

            if not is_retryable_error(e, config):
                logger.warning(
                    "non_retryable_generator_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            if state.attempt >= config.max_retries:
                logger.error(
                    "generator_max_retries_exceeded",
                    attempts=state.attempt,
                    total_delay=state.total_delay,
                    error=str(e),
                    buffered_chunks=len(chunk_buffer),
                )
                raise

            retry_after = extract_retry_after(e)
            if retry_after:
                delay = min(retry_after, config.max_delay_seconds)
            else:
                delay = calculate_delay(state.attempt - 1, config)

            state.total_delay += delay

            logger.warning(
                "retrying_generator_after_error",
                attempt=state.attempt,
                max_retries=config.max_retries,
                delay_seconds=round(delay, 2),
                buffered_chunks=len(chunk_buffer),
                error=str(e)[:200],
            )

            await asyncio.sleep(delay)
