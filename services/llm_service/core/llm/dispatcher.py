"""LLM Request Dispatcher with rate limiting and queue management.

Provides centralized control over LLM API requests:
- Rate limiting per provider/model
- Request queuing during rate limit windows
- Concurrent request limiting
- Provider health tracking
- Automatic failover (future)
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import structlog

from services.llm_service.core.config.constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    CIRCUIT_BREAKER_WINDOW_SIZE,
    DEFAULT_MAX_CONCURRENT_LLM_REQUESTS,
    DEFAULT_MAX_QUEUE_SIZE,
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


class CircuitState(str, Enum):
    """Circuit breaker state for a provider.

    State machine:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Circuit is tripped, requests are rejected (fail fast)
    - HALF_OPEN: Testing if provider has recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


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
class CircuitBreaker:
    """Circuit breaker for a provider.

    Tracks request success/failure in a sliding window and manages
    circuit state transitions.

    Attributes:
        state: Current circuit state (closed, open, half_open).
        window_size: Number of requests to track in sliding window.
        failure_threshold: Failure rate (0.0-1.0) to open circuit.
        recovery_timeout_seconds: Seconds to wait before half-open.
        results: Sliding window of request results (True=success, False=failure).
        opened_at: When the circuit was opened.
    """

    state: CircuitState = CircuitState.CLOSED
    window_size: int = CIRCUIT_BREAKER_WINDOW_SIZE
    failure_threshold: float = CIRCUIT_BREAKER_FAILURE_THRESHOLD
    recovery_timeout_seconds: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
    results: deque[bool] = field(
        default_factory=lambda: deque(maxlen=CIRCUIT_BREAKER_WINDOW_SIZE)
    )
    opened_at: datetime | None = None

    def record_success(self) -> None:
        """Record a successful request."""
        self.results.append(True)

        # Half-open -> closed on success
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Record a failed request."""
        self.results.append(False)

        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self._should_open():
                self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            # Half-open -> open on failure
            self._transition_to(CircuitState.OPEN)

    def _should_open(self) -> bool:
        """Check if circuit should open based on failure rate.

        Returns:
            True if failure rate exceeds threshold with sufficient samples.
        """
        # Need at least 10 samples to make a decision
        if len(self.results) < 10:
            return False

        failure_count = sum(1 for r in self.results if not r)
        failure_rate = failure_count / len(self.results)
        return failure_rate >= self.failure_threshold

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state.

        Args:
            new_state: The new circuit state.
        """
        old_state = self.state
        self.state = new_state

        if new_state == CircuitState.OPEN:
            self.opened_at = datetime.now(UTC)
        elif new_state == CircuitState.CLOSED:
            self.opened_at = None
            # Clear the window on recovery
            self.results.clear()

        logger.info(
            "circuit_breaker_transition",
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def check_state(self) -> CircuitState:
        """Check and potentially update circuit state.

        If the circuit is open and recovery timeout has elapsed,
        transitions to half-open state.

        Returns:
            Current circuit state after any transitions.
        """
        if self.state == CircuitState.OPEN and self.opened_at:
            elapsed = (datetime.now(UTC) - self.opened_at).total_seconds()
            if elapsed >= self.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

        return self.state

    def is_request_allowed(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        state = self.check_state()

        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            # Allow one test request in half-open
            return True
        else:
            # OPEN - reject
            return False

    def get_failure_rate(self) -> float:
        """Get current failure rate.

        Returns:
            Failure rate as float 0.0-1.0, or 0.0 if no samples.
        """
        if not self.results:
            return 0.0
        failure_count = sum(1 for r in self.results if not r)
        return failure_count / len(self.results)


@dataclass
class DispatcherConfig:
    """Configuration for the LLM dispatcher.

    Attributes:
        max_concurrent_requests: Maximum concurrent requests per provider.
        max_queue_size: Maximum requests that can be queued per provider.
        rate_limit_window_seconds: Window for tracking rate limits.
        max_rate_limits_per_window: Max rate limits before marking unhealthy.
        queue_timeout_seconds: Timeout for queued requests.
        health_check_interval_seconds: How often to check provider health.
    """

    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_LLM_REQUESTS
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
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
        circuit_breaker: Circuit breaker for fail-fast behavior.
        semaphore: Concurrency limiter.
        queue: Queue for pending requests when rate limited.
        max_queue_size: Maximum queue size for this provider.
        pending_requests: Number of requests waiting.
        active_requests: Number of currently executing requests.
        total_requests: Total requests processed.
        total_failures: Total request failures.
    """

    name: str
    health: ProviderHealth = ProviderHealth.HEALTHY
    rate_limit: RateLimitState = field(default_factory=RateLimitState)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    semaphore: asyncio.Semaphore | None = None
    queue: asyncio.Queue[asyncio.Event] | None = None
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
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
                    queue=asyncio.Queue(maxsize=self.config.max_queue_size),
                    max_queue_size=self.config.max_queue_size,
                )
                logger.info(
                    "dispatcher_provider_registered",
                    provider=name,
                    max_concurrent=self.config.max_concurrent_requests,
                    max_queue_size=self.config.max_queue_size,
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
            state.circuit_breaker.record_success()
            state.total_requests += 1
            self._update_health(state)
            # Signal queued requests that rate limit is cleared
            self._process_queue(state)

    def mark_failure(self, provider: str, is_rate_limit: bool = False) -> None:
        """Mark a failed request.

        Args:
            provider: Provider name.
            is_rate_limit: Whether this was a rate limit error.
        """
        if provider in self._providers:
            state = self._providers[provider]
            state.total_failures += 1
            state.circuit_breaker.record_failure()
            if is_rate_limit:
                state.rate_limit.mark_limited()
            self._update_health(state)

    def _update_health(self, state: ProviderState) -> None:
        """Update provider health based on current state.

        Health is determined by both rate limit and circuit breaker states:
        - UNHEALTHY: Circuit is open OR 3+ consecutive rate limits
        - DEGRADED: Circuit is half-open OR currently rate limited
        - HEALTHY: Circuit is closed AND not rate limited

        Args:
            state: Provider state to update.
        """
        old_health = state.health
        circuit_state = state.circuit_breaker.check_state()

        # Check circuit breaker state first (takes precedence)
        if circuit_state == CircuitState.OPEN:
            state.health = ProviderHealth.UNHEALTHY
        elif state.rate_limit.consecutive_limits >= 3:
            state.health = ProviderHealth.UNHEALTHY
        elif circuit_state == CircuitState.HALF_OPEN:
            state.health = ProviderHealth.DEGRADED
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
                circuit_state=circuit_state.value,
            )

    def _process_queue(self, state: ProviderState) -> None:
        """Process queued requests when rate limit clears.

        Signals all waiting requests to proceed when rate limit is cleared.

        Args:
            state: Provider state with queue to process.
        """
        if not state.queue or state.queue.empty():
            return

        # Signal all waiting requests to proceed
        signaled = 0
        while not state.queue.empty():
            try:
                event = state.queue.get_nowait()
                event.set()
                signaled += 1
            except asyncio.QueueEmpty:
                break

        if signaled > 0:
            logger.info(
                "queue_processed",
                provider=state.name,
                requests_signaled=signaled,
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
                "circuit_state": state.circuit_breaker.state.value,
                "failure_rate": state.circuit_breaker.get_failure_rate(),
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
                    "circuit_state": state.circuit_breaker.state.value,
                    "failure_rate": state.circuit_breaker.get_failure_rate(),
                    "active_requests": state.active_requests,
                    "total_requests": state.total_requests,
                }
                for name, state in self._providers.items()
            }
        }

    def get_status(self) -> dict[str, Any]:
        """Get dispatcher status for health endpoint.

        Returns status fields:
        - active_requests: Total active requests across all providers
        - queue_depth: Total queued requests across all providers
        - rate_limit_remaining: Number of concurrent slots remaining

        Returns:
            Status dictionary for health endpoint.
        """
        active_requests = sum(
            state.active_requests for state in self._providers.values()
        )
        queue_depth = sum(
            state.queue.qsize() if state.queue else 0
            for state in self._providers.values()
        )

        # Calculate total capacity and remaining slots
        total_capacity = len(self._providers) * self.config.max_concurrent_requests
        rate_limit_remaining = max(0, total_capacity - active_requests)

        # If no providers registered yet, report full capacity based on config
        if not self._providers:
            rate_limit_remaining = self.config.max_concurrent_requests

        return {
            "active_requests": active_requests,
            "queue_depth": queue_depth,
            "rate_limit_remaining": rate_limit_remaining,
        }


class DispatcherContext:
    """Context manager for dispatched requests.

    Handles acquiring/releasing semaphore and tracking metrics.
    Queues requests when rate limited and rejects when queue is full.
    """

    def __init__(self, dispatcher: LLMDispatcher, state: ProviderState):
        """Initialize the context.

        Args:
            dispatcher: Parent dispatcher.
            state: Provider state.
        """
        self.dispatcher = dispatcher
        self.state = state
        self._queue_event: asyncio.Event | None = None

    async def __aenter__(self) -> "DispatcherContext":
        """Enter the context - check circuit breaker, queue if rate limited, wait for slot.

        If the circuit breaker is open, a CircuitOpenError is raised immediately.
        If the provider is rate limited, requests are queued. When the queue
        is full, a QueueFullError is raised with a Retry-After value.

        Raises:
            CircuitOpenError: When the circuit breaker is open.
            QueueFullError: When the queue is full and cannot accept more requests.
            asyncio.TimeoutError: When the queued request times out.
        """
        from services.llm_service.core.llm.exceptions import (
            CircuitOpenError,
            QueueFullError,
        )

        # Check circuit breaker first (fail fast)
        if not self.state.circuit_breaker.is_request_allowed():
            raise CircuitOpenError(
                provider=self.state.name,
                retry_after=self.state.circuit_breaker.recovery_timeout_seconds,
                failure_rate=self.state.circuit_breaker.get_failure_rate(),
            )

        self.state.pending_requests += 1

        try:
            # Check if rate limited and queue is needed
            if self.state.rate_limit.check_and_clear() and self.state.queue:
                # Calculate retry_after from rate limit state
                retry_after = self.dispatcher.config.queue_timeout_seconds
                if self.state.rate_limit.retry_after:
                    remaining = (
                        self.state.rate_limit.retry_after - datetime.now(UTC)
                    ).total_seconds()
                    if remaining > 0:
                        retry_after = remaining

                # Try to add to queue (non-blocking check if full)
                if self.state.queue.full():
                    raise QueueFullError(
                        provider=self.state.name,
                        retry_after=retry_after,
                        queue_size=self.state.max_queue_size,
                    )

                # Create event to wait on and add to queue
                self._queue_event = asyncio.Event()
                try:
                    self.state.queue.put_nowait(self._queue_event)
                except asyncio.QueueFull:
                    # Race condition - queue filled between check and put
                    raise QueueFullError(
                        provider=self.state.name,
                        retry_after=retry_after,
                        queue_size=self.state.max_queue_size,
                    )

                logger.info(
                    "request_queued",
                    provider=self.state.name,
                    queue_depth=self.state.queue.qsize(),
                    timeout=self.dispatcher.config.queue_timeout_seconds,
                )

                # Wait for signal, rate limit expiry, or timeout
                start_time = datetime.now(UTC)
                timeout_seconds = self.dispatcher.config.queue_timeout_seconds

                while True:
                    # Calculate remaining timeout
                    elapsed = (datetime.now(UTC) - start_time).total_seconds()
                    remaining_timeout = timeout_seconds - elapsed

                    if remaining_timeout <= 0:
                        logger.warning(
                            "queued_request_timeout",
                            provider=self.state.name,
                            timeout=timeout_seconds,
                        )
                        raise TimeoutError(
                            f"Queued request timed out after {timeout_seconds}s"
                        )

                    # Check if rate limit has cleared
                    if not self.state.rate_limit.check_and_clear():
                        # Rate limit cleared, can proceed
                        break

                    # Wait for event or timeout, checking periodically
                    wait_time = min(remaining_timeout, 0.5)  # Check every 0.5s
                    try:
                        await asyncio.wait_for(
                            self._queue_event.wait(),
                            timeout=wait_time,
                        )
                        # Event was signaled, can proceed
                        break
                    except TimeoutError:
                        # Timeout just means we need to check again
                        continue

            # Wait for any remaining rate limit to clear
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
