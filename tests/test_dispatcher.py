"""Tests for LLM dispatcher."""

import asyncio

import pytest

from services.llm_service.core.llm.dispatcher import (
    DispatcherConfig,
    LLMDispatcher,
    ProviderHealth,
    RateLimitState,
    get_dispatcher,
    reset_dispatcher,
)
from services.llm_service.core.llm.exceptions import QueueFullError


class TestRateLimitState:
    """Tests for RateLimitState."""

    def test_initial_state(self):
        """Test initial state is not limited."""
        state = RateLimitState()
        assert not state.is_limited
        assert state.consecutive_limits == 0
        assert state.check_and_clear() is False

    def test_mark_limited(self):
        """Test marking as rate limited."""
        state = RateLimitState()
        state.mark_limited(retry_seconds=60.0)
        assert state.is_limited
        assert state.consecutive_limits == 1
        assert state.retry_after is not None

    def test_mark_success_clears_consecutive(self):
        """Test that success clears consecutive limit count."""
        state = RateLimitState()
        state.mark_limited()
        state.mark_limited()
        assert state.consecutive_limits == 2

        state.mark_success()
        assert state.consecutive_limits == 0
        assert not state.is_limited

    def test_reset_window(self):
        """Test window reset."""
        state = RateLimitState()
        state.mark_limited()
        state.mark_limited()
        assert state.total_limits == 2

        state.reset_window()
        assert state.total_limits == 0


class TestLLMDispatcher:
    """Tests for LLMDispatcher."""

    @pytest.fixture
    def dispatcher(self):
        """Create a fresh dispatcher for each test."""
        reset_dispatcher()
        return LLMDispatcher(DispatcherConfig(max_concurrent_requests=2))

    @pytest.mark.asyncio
    async def test_acquire_creates_provider(self, dispatcher):
        """Test that acquire creates provider state."""
        async with await dispatcher.acquire("openrouter"):
            pass

        stats = dispatcher.get_stats("openrouter")
        assert stats["provider"] == "openrouter"
        assert stats["health"] == "healthy"

    @pytest.mark.asyncio
    async def test_concurrent_requests_limited(self, dispatcher):
        """Test that concurrent requests are limited by semaphore."""
        import asyncio

        active_count = 0
        max_active = 0

        async def make_request():
            nonlocal active_count, max_active
            async with await dispatcher.acquire("openrouter"):
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.1)
                active_count -= 1

        # Launch 5 concurrent requests with limit of 2
        await asyncio.gather(*[make_request() for _ in range(5)])

        # Max concurrent should be limited to 2
        assert max_active <= 2

    def test_mark_rate_limited(self, dispatcher):
        """Test marking provider as rate limited."""
        # First acquire to create provider
        import asyncio

        asyncio.get_event_loop().run_until_complete(dispatcher.acquire("openrouter"))

        dispatcher.mark_rate_limited("openrouter", retry_seconds=60.0)
        assert dispatcher.get_health("openrouter") == ProviderHealth.DEGRADED

    def test_health_transitions(self, dispatcher):
        """Test health state transitions."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(dispatcher.acquire("openrouter"))

        # Initially healthy
        assert dispatcher.get_health("openrouter") == ProviderHealth.HEALTHY

        # One rate limit -> degraded
        dispatcher.mark_rate_limited("openrouter")
        assert dispatcher.get_health("openrouter") == ProviderHealth.DEGRADED

        # Multiple consecutive limits -> unhealthy
        dispatcher.mark_rate_limited("openrouter")
        dispatcher.mark_rate_limited("openrouter")
        assert dispatcher.get_health("openrouter") == ProviderHealth.UNHEALTHY

        # Success clears consecutive and returns to healthy
        dispatcher.mark_success("openrouter")
        assert dispatcher.get_health("openrouter") == ProviderHealth.HEALTHY

    def test_get_stats_all_providers(self, dispatcher):
        """Test getting stats for all providers."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(dispatcher.acquire("openrouter"))
        asyncio.get_event_loop().run_until_complete(dispatcher.acquire("openai"))

        stats = dispatcher.get_stats()
        assert "providers" in stats
        assert "openrouter" in stats["providers"]
        assert "openai" in stats["providers"]


class TestGlobalDispatcher:
    """Tests for global dispatcher singleton."""

    def test_get_dispatcher_singleton(self):
        """Test that get_dispatcher returns same instance."""
        reset_dispatcher()
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        assert d1 is d2

    def test_reset_dispatcher(self):
        """Test that reset_dispatcher clears the singleton."""
        d1 = get_dispatcher()
        reset_dispatcher()
        d2 = get_dispatcher()
        assert d1 is not d2


class TestDispatcherQueue:
    """Tests for dispatcher queue behavior."""

    @pytest.fixture
    def dispatcher_with_queue(self):
        """Create a dispatcher with small queue for testing."""
        reset_dispatcher()
        return LLMDispatcher(
            DispatcherConfig(
                max_concurrent_requests=2,
                max_queue_size=3,
                queue_timeout_seconds=1.0,
            )
        )

    @pytest.mark.asyncio
    async def test_queue_created_for_provider(self, dispatcher_with_queue):
        """Test that queue is created when provider is registered."""
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        state = dispatcher_with_queue._providers["openrouter"]
        assert state.queue is not None
        assert state.max_queue_size == 3

    @pytest.mark.asyncio
    async def test_request_queued_when_rate_limited(self, dispatcher_with_queue):
        """Test that requests are queued when provider is rate limited."""
        # First acquire to create provider
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        # Mark as rate limited
        dispatcher_with_queue.mark_rate_limited("openrouter", retry_seconds=0.5)

        state = dispatcher_with_queue._providers["openrouter"]
        assert state.rate_limit.is_limited

        # Start a request that should be queued
        async def make_request():
            async with await dispatcher_with_queue.acquire("openrouter"):
                return True

        # The request should wait in queue then succeed after rate limit clears
        result = await asyncio.wait_for(make_request(), timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_queue_full_raises_error(self, dispatcher_with_queue):
        """Test that QueueFullError is raised when queue is full."""
        # First acquire to create provider
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        # Mark as rate limited with long timeout
        dispatcher_with_queue.mark_rate_limited("openrouter", retry_seconds=60.0)

        state = dispatcher_with_queue._providers["openrouter"]

        # Fill the queue with events
        for _ in range(3):  # max_queue_size = 3
            state.queue.put_nowait(asyncio.Event())

        # Next request should get QueueFullError
        with pytest.raises(QueueFullError) as exc_info:
            async with await dispatcher_with_queue.acquire("openrouter"):
                pass

        assert exc_info.value.queue_size == 3
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_queued_requests_signaled_on_success(self, dispatcher_with_queue):
        """Test that queued requests are signaled when rate limit clears."""
        # First acquire to create provider
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        state = dispatcher_with_queue._providers["openrouter"]

        # Add events to queue manually
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        state.queue.put_nowait(event1)
        state.queue.put_nowait(event2)

        # Events should not be set yet
        assert not event1.is_set()
        assert not event2.is_set()

        # Mark success should process queue
        dispatcher_with_queue.mark_success("openrouter")

        # Events should now be set
        assert event1.is_set()
        assert event2.is_set()

        # Queue should be empty
        assert state.queue.empty()

    @pytest.mark.asyncio
    async def test_queue_timeout_raises_error(self, dispatcher_with_queue):
        """Test that queued requests timeout after queue_timeout_seconds."""
        # First acquire to create provider
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        # Mark as rate limited with long timeout
        dispatcher_with_queue.mark_rate_limited("openrouter", retry_seconds=60.0)

        # Request should timeout (queue_timeout_seconds=1.0)
        with pytest.raises(asyncio.TimeoutError):
            async with await dispatcher_with_queue.acquire("openrouter"):
                pass

    @pytest.mark.asyncio
    async def test_queue_depth_in_status(self, dispatcher_with_queue):
        """Test that queue depth is reported correctly in status."""
        # First acquire to create provider
        async with await dispatcher_with_queue.acquire("openrouter"):
            pass

        state = dispatcher_with_queue._providers["openrouter"]

        # Add events to queue manually
        state.queue.put_nowait(asyncio.Event())
        state.queue.put_nowait(asyncio.Event())

        status = dispatcher_with_queue.get_status()
        assert status["queue_depth"] == 2
