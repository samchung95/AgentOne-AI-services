"""Tests for LLM dispatcher."""

import pytest

from services.llm_service.core.llm.dispatcher import (
    DispatcherConfig,
    LLMDispatcher,
    ProviderHealth,
    RateLimitState,
    get_dispatcher,
    reset_dispatcher,
)


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
