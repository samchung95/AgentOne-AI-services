"""Custom exception hierarchy for LLM Service.

This module defines exceptions for LLM-related errors.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ErrorInfo:
    """Structured error information for error responses."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class LLMServiceError(Exception):
    """Base exception for all LLM service errors."""

    def __init__(
        self,
        message: str,
        code: str = "LLM_SERVICE_ERROR",
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.retryable = retryable

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class ConfigurationError(LLMServiceError):
    """Error in application configuration."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            details=details,
            retryable=False,
        )


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""

    def __init__(self, provider: str, key_name: str):
        super().__init__(
            message=f"Missing API key for {provider}: {key_name} is required",
            details={"provider": provider, "key_name": key_name},
        )
        self.code = "MISSING_API_KEY"
        self.provider = provider
        self.key_name = key_name


class LLMError(LLMServiceError):
    """Base error for LLM-related issues."""

    def __init__(
        self,
        message: str,
        code: str = "LLM_ERROR",
        details: dict[str, Any] | None = None,
        retryable: bool = True,
    ):
        super().__init__(message, code, details, retryable)


class LLMProviderError(LLMError):
    """Error from LLM provider API."""

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        retryable: bool = True,
    ):
        super().__init__(
            message=f"{provider} API error: {message}",
            code="LLM_PROVIDER_ERROR",
            details={"provider": provider, "status_code": status_code},
            retryable=retryable,
        )
        self.provider = provider
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, provider: str, timeout_seconds: float):
        super().__init__(
            message=f"LLM request to {provider} timed out after {timeout_seconds}s",
            code="LLM_TIMEOUT",
            details={"provider": provider, "timeout_seconds": timeout_seconds},
            retryable=True,
        )


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            code="LLM_RATE_LIMIT",
            details={"provider": provider, "retry_after": retry_after},
            retryable=True,
        )
        self.retry_after = retry_after


class QueueFullError(LLMRateLimitError):
    """Dispatcher queue is full - cannot accept more requests.

    This error is raised when the request queue for a provider is full
    and no more requests can be queued. Clients should retry after the
    specified delay.
    """

    def __init__(self, provider: str, retry_after: float, queue_size: int):
        super().__init__(provider=provider, retry_after=retry_after)
        self.code = "QUEUE_FULL"
        self.message = (
            f"Request queue full for {provider} "
            f"(queue_size={queue_size}). Retry after {retry_after}s"
        )
        self.details = {
            "provider": provider,
            "retry_after": retry_after,
            "queue_size": queue_size,
        }
        self.queue_size = queue_size
