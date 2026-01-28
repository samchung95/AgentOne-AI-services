"""LLM Request Dispatcher with rate limiting and queue management.

Provides centralized control over LLM API requests:
- Rate limiting per provider/model
- Request queuing during rate limit windows
- Concurrent request limiting
- Provider health tracking
- Automatic failover (future)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import structlog

from services.llm_service.core.config.constants import (
    DEFAULT_MAX_CONCURRENT_LLM_REQUESTS,
    DISPATCHER_HEALTH_CHECK_INTERVAL_SECONDS,
    MAX_RATE_LIMITS_PER_WINDOW,
    QUEUE_TIMEOUT_SECONDS,
    RATE_LIMIT_WINDOW_SECONDS,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ProviderHealth(str, Enum):
    """Health status of an LLM provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Experiencing rate limits
    UNHEALTHY = "unhealthy"  # Multiple failures


@dataclass
class RateLimitState:
    """Tracks rate limit state for a provider.

    Attributes:
        is_limited: Whether currently rate limited.
        retry_after: When rate limit expires.
        consecutive_limits: Number of consecutive rate limits hit.
        total_limits: Total rate limits hit in current window.
        window_start: Start of current tracking window.
    """

    is_limited: bool = False
    retry_after: datetime | None = None
    consecutive_limits: int = 0
    total_limits: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))

    def mark_limited(self, retry_seconds: float = 60.0) -> None:
        """Mark provider as rate limited.

        Args:
            retry_seconds: Seconds until rate limit expires.
        """
        self.is_limited = True
        self.retry_after = datetime.now(UTC) + timedelta(seconds=retry_seconds)
        self.consecutive_limits += 1
        self.total_limits += 1

        logger.warning(
            "provider_rate_limited",
            retry_after=self.retry_after.isoformat(),
            consecutive_limits=self.consecutive_limits,
        )

    def mark_success(self) -> None:
        """Mark a successful request (clears consecutive limit count)."""
        self.consecutive_limits = 0
        self.is_limited = False
        self.retry_after = None

    def check_and_clear(self) -> bool:
        """Check if rate limit has expired and clear if so.

        Returns:
            True if still rate limited, False if cleared.
        """
        if not self.is_limited:
            return False

        if self.retry_after and datetime.now(UTC) >= self.retry_after:
            self.is_limited = False
            self.retry_after = None
            logger.info("rate_limit_expired")
            return False

        return True

    def reset_window(self) -> None:
        """Reset the tracking window (e.g., every hour)."""
        self.total_limits = 0
        self.window_start = datetime.now(UTC)


@dataclass
class DispatcherConfig:
    """Configuration for the LLM dispatcher.

    Attributes:
        max_concurrent_requests: Maximum concurrent requests per provider.
        rate_limit_window_seconds: Window for tracking rate limits.
        max_rate_limits_per_window: Max rate limits before marking unhealthy.
        queue_timeout_seconds: Timeout for queued requests.
        health_check_interval_seconds: How often to check provider health.
    """

    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_LLM_REQUESTS
    rate_limit_window_seconds: float = RATE_LIMIT_WINDOW_SECONDS
    max_rate_limits_per_window: int = MAX_RATE_LIMITS_PER_WINDOW
    queue_timeout_seconds: float = QUEUE_TIMEOUT_SECONDS
    health_check_interval_seconds: float = DISPATCHER_HEALTH_CHECK_INTERVAL_SECONDS


@dataclass
class ProviderState:
    """State for a single LLM provider.

    Attributes:
        name: Provider name.
        health: Current health status.
        rate_limit: Rate limit tracking.
        semaphore: Concurrency limiter.
        pending_requests: Number of requests waiting.
        active_requests: Number of currently executing requests.
        total_requests: Total requests processed.
        total_failures: Total request failures.
    """

    name: str
    health: ProviderHealth = ProviderHealth.HEALTHY
    rate_limit: RateLimitState = field(default_factory=RateLimitState)
    semaphore: asyncio.Semaphore | None = None
    pending_requests: int = 0
    active_requests: int = 0
    total_requests: int = 0
    total_failures: int = 0


class LLMDispatcher:
    """Centralized dispatcher for LLM API requests.

    Manages rate limiting, queuing, and concurrency control for LLM providers.
    Designed to be used as a singleton across the application.

    Example:
        dispatcher = LLMDispatcher()

        async def make_request():
            async with dispatcher.acquire("openrouter"):
                return await client.generate_stream(...)
    """

    def __init__(self, config: DispatcherConfig | None = None):
        """Initialize the dispatcher.

        Args:
            config: Dispatcher configuration.
        """
        self.config = config or DispatcherConfig()
        self._providers: dict[str, ProviderState] = {}
        self._lock = asyncio.Lock()

    async def _get_or_create_provider(self, name: str) -> ProviderState:
        """Get or create state for a provider.

        Args:
            name: Provider name.

        Returns:
            Provider state object.
        """
        async with self._lock:
            if name not in self._providers:
                self._providers[name] = ProviderState(
                    name=name,
                    semaphore=asyncio.Semaphore(self.config.max_concurrent_requests),
                )
                logger.info(
                    "dispatcher_provider_registered",
                    provider=name,
                    max_concurrent=self.config.max_concurrent_requests,
                )
            return self._providers[name]

    async def acquire(self, provider: str) -> "DispatcherContext":
        """Acquire a slot for making a request.

        This is an async context manager that:
        1. Waits if rate limited
        2. Acquires a concurrency slot
        3. Tracks request metrics

        Args:
            provider: Provider name (e.g., "openrouter", "openai").

        Returns:
            Context manager for the request.

        Example:
            async with dispatcher.acquire("openrouter"):
                response = await client.generate(...)
        """
        state = await self._get_or_create_provider(provider)
        return DispatcherContext(self, state)

    async def wait_for_rate_limit(self, state: ProviderState) -> None:
        """Wait if provider is rate limited.

        Args:
            state: Provider state to check.
        """
        while state.rate_limit.check_and_clear():
            wait_seconds = (state.rate_limit.retry_after - datetime.now(UTC)).total_seconds()

            if wait_seconds > 0:
                logger.info(
                    "waiting_for_rate_limit",
                    provider=state.name,
                    wait_seconds=round(wait_seconds, 1),
                )
                await asyncio.sleep(min(wait_seconds, 5.0))  # Check every 5s max

    def mark_rate_limited(
        self,
        provider: str,
        retry_seconds: float = 60.0,
    ) -> None:
        """Mark a provider as rate limited.

        Args:
            provider: Provider name.
            retry_seconds: Seconds until rate limit expires.
        """
        if provider in self._providers:
            state = self._providers[provider]
            state.rate_limit.mark_limited(retry_seconds)
            self._update_health(state)

    def mark_success(self, provider: str) -> None:
        """Mark a successful request.

        Args:
            provider: Provider name.
        """
        if provider in self._providers:
            state = self._providers[provider]
            state.rate_limit.mark_success()
            state.total_requests += 1
            self._update_health(state)

    def mark_failure(self, provider: str, is_rate_limit: bool = False) -> None:
        """Mark a failed request.

        Args:
            provider: Provider name.
            is_rate_limit: Whether this was a rate limit error.
        """
        if provider in self._providers:
            state = self._providers[provider]
            state.total_failures += 1
            if is_rate_limit:
                state.rate_limit.mark_limited()
            self._update_health(state)

    def _update_health(self, state: ProviderState) -> None:
        """Update provider health based on current state.

        Args:
            state: Provider state to update.
        """
        old_health = state.health

        if state.rate_limit.consecutive_limits >= 3:
            state.health = ProviderHealth.UNHEALTHY
        elif state.rate_limit.is_limited:
            state.health = ProviderHealth.DEGRADED
        else:
            state.health = ProviderHealth.HEALTHY

        if state.health != old_health:
            logger.info(
                "provider_health_changed",
                provider=state.name,
                old_health=old_health.value,
                new_health=state.health.value,
            )

    def get_health(self, provider: str) -> ProviderHealth:
        """Get current health of a provider.

        Args:
            provider: Provider name.

        Returns:
            Provider health status.
        """
        if provider in self._providers:
            return self._providers[provider].health
        return ProviderHealth.HEALTHY

    def get_stats(self, provider: str | None = None) -> dict[str, Any]:
        """Get dispatcher statistics.

        Args:
            provider: Specific provider, or None for all.

        Returns:
            Statistics dictionary.
        """
        if provider and provider in self._providers:
            state = self._providers[provider]
            return {
                "provider": state.name,
                "health": state.health.value,
                "is_rate_limited": state.rate_limit.is_limited,
                "consecutive_limits": state.rate_limit.consecutive_limits,
                "active_requests": state.active_requests,
                "pending_requests": state.pending_requests,
                "total_requests": state.total_requests,
                "total_failures": state.total_failures,
            }

        return {
            "providers": {
                name: {
                    "health": state.health.value,
                    "is_rate_limited": state.rate_limit.is_limited,
                    "active_requests": state.active_requests,
                    "total_requests": state.total_requests,
                }
                for name, state in self._providers.items()
            }
        }


class DispatcherContext:
    """Context manager for dispatched requests.

    Handles acquiring/releasing semaphore and tracking metrics.
    """

    def __init__(self, dispatcher: LLMDispatcher, state: ProviderState):
        """Initialize the context.

        Args:
            dispatcher: Parent dispatcher.
            state: Provider state.
        """
        self.dispatcher = dispatcher
        self.state = state

    async def __aenter__(self) -> "DispatcherContext":
        """Enter the context - wait for rate limit and acquire slot."""
        self.state.pending_requests += 1

        try:
            # Wait for any rate limit to clear
            await self.dispatcher.wait_for_rate_limit(self.state)

            # Acquire concurrency slot
            if self.state.semaphore:
                await self.state.semaphore.acquire()

            self.state.active_requests += 1
        finally:
            self.state.pending_requests -= 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context - release slot and track metrics."""
        self.state.active_requests -= 1

        # Release semaphore
        if self.state.semaphore:
            self.state.semaphore.release()

        # Track success/failure
        if exc_type is None:
            self.dispatcher.mark_success(self.state.name)
        else:
            # Check if it's a rate limit error
            is_rate_limit = exc_val and ("429" in str(exc_val) or "rate" in str(exc_val).lower())
            self.dispatcher.mark_failure(self.state.name, is_rate_limit)

        return False  # Don't suppress exceptions


# Global dispatcher instance
_dispatcher: LLMDispatcher | None = None


def get_dispatcher() -> LLMDispatcher:
    """Get the global LLM dispatcher instance.

    Returns:
        Global LLMDispatcher instance.
    """
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LLMDispatcher()
    return _dispatcher


def reset_dispatcher() -> None:
    """Reset the global dispatcher (for testing)."""
    global _dispatcher
    _dispatcher = None
