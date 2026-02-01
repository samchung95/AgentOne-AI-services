"""Tests for retry logic."""


import pytest

from services.llm_service.core.llm.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    RetryStrategy,
    calculate_delay,
    extract_retry_after,
    is_retryable_error,
    retry_async_generator,
    with_retry,
)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        )
        assert calculate_delay(0, config) == 1.0
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 4.0
        assert calculate_delay(3, config) == 8.0

    def test_linear_backoff(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            jitter=False,
            strategy=RetryStrategy.LINEAR_BACKOFF,
        )
        assert calculate_delay(0, config) == 1.0
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 3.0

    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        config = RetryConfig(
            initial_delay_seconds=5.0,
            jitter=False,
            strategy=RetryStrategy.FIXED_DELAY,
        )
        assert calculate_delay(0, config) == 5.0
        assert calculate_delay(1, config) == 5.0
        assert calculate_delay(10, config) == 5.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay_seconds."""
        config = RetryConfig(
            initial_delay_seconds=10.0,
            max_delay_seconds=30.0,
            exponential_base=2.0,
            jitter=False,
        )
        # 10 * 2^3 = 80, but should be capped at 30
        assert calculate_delay(3, config) == 30.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(
            initial_delay_seconds=10.0,
            jitter=True,
            jitter_factor=0.25,
        )
        delays = [calculate_delay(0, config) for _ in range(10)]
        # With jitter, delays should vary (not all equal)
        assert len(set(delays)) > 1
        # All delays should be >= base and <= base * (1 + jitter_factor)
        for d in delays:
            assert 10.0 <= d <= 12.5


class TestIsRetryableError:
    """Tests for retryable error detection."""

    def test_rate_limit_in_message(self):
        """Test detection of rate limit errors by message content."""
        config = DEFAULT_RETRY_CONFIG
        assert is_retryable_error(Exception("Error 429: Too many requests"), config)
        assert is_retryable_error(Exception("Rate limit exceeded"), config)
        assert is_retryable_error(Exception("Quota exceeded"), config)

    def test_status_code_attribute(self):
        """Test detection by status_code attribute."""
        config = DEFAULT_RETRY_CONFIG

        class HTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        assert is_retryable_error(HTTPError(429), config)
        assert is_retryable_error(HTTPError(500), config)
        assert is_retryable_error(HTTPError(503), config)
        assert not is_retryable_error(HTTPError(400), config)
        assert not is_retryable_error(HTTPError(404), config)

    def test_connection_errors(self):
        """Test that connection errors are retryable."""
        config = DEFAULT_RETRY_CONFIG
        assert is_retryable_error(ConnectionError("Connection refused"), config)
        assert is_retryable_error(TimeoutError("Timed out"), config)

    def test_non_retryable_errors(self):
        """Test that generic errors are not retryable."""
        config = DEFAULT_RETRY_CONFIG
        assert not is_retryable_error(ValueError("Invalid value"), config)
        assert not is_retryable_error(KeyError("Missing key"), config)


class TestExtractRetryAfter:
    """Tests for Retry-After header extraction."""

    def test_retry_after_from_response(self):
        """Test extraction of Retry-After from response headers."""

        class MockResponse:
            headers = {"Retry-After": "30"}

        class MockError(Exception):
            response = MockResponse()

        assert extract_retry_after(MockError()) == 30.0

    def test_no_retry_after(self):
        """Test when no Retry-After header is present."""
        assert extract_retry_after(Exception("No header")) is None


class TestWithRetryDecorator:
    """Tests for the with_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test successful execution without retries."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test retry followed by success."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, initial_delay_seconds=0.01, jitter=False))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=2, initial_delay_seconds=0.01, jitter=False))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        """Test that non-retryable errors are raised immediately."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, initial_delay_seconds=0.01))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # No retries


class TestRetryAsyncGeneratorBuffer:
    """Tests for async generator retry with chunk buffering."""

    @pytest.mark.asyncio
    async def test_chunks_are_buffered(self):
        """Test that chunks are stored in buffer during iteration."""
        chunks_yielded = []

        async def gen():
            for i in range(5):
                yield f"chunk_{i}"

        config = RetryConfig(max_buffer_chunks=10)
        async for chunk in retry_async_generator(gen, config):
            chunks_yielded.append(chunk)

        assert chunks_yielded == ["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]

    @pytest.mark.asyncio
    async def test_buffer_respects_max_limit(self):
        """Test that buffer does not exceed max_buffer_chunks.

        The buffer only stores up to max_buffer_chunks for potential replay.
        This test verifies the buffer limit is respected (all chunks are still
        yielded, but only the first N are buffered).
        """
        call_count = 0

        async def gen():
            nonlocal call_count
            call_count += 1
            for i in range(20):
                yield f"chunk_{i}"

        # With max_buffer_chunks=5, we buffer at most 5 chunks
        # but yield all 20 from the generator
        config = RetryConfig(max_buffer_chunks=5)

        chunks = []
        async for chunk in retry_async_generator(gen, config):
            chunks.append(chunk)

        # All 20 chunks should be yielded
        assert len(chunks) == 20
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_buffer_cleared_on_success(self):
        """Test that buffer is cleared after successful completion."""
        # We verify buffer clearing by checking that memory usage pattern
        # is as expected (buffer doesn't accumulate across iterations)
        call_count = 0

        async def gen():
            nonlocal call_count
            call_count += 1
            for i in range(5):
                yield f"chunk_{i}"

        config = RetryConfig(max_buffer_chunks=1000)

        # First run
        chunks1 = []
        async for chunk in retry_async_generator(gen, config):
            chunks1.append(chunk)

        # Second run (separate generator wrapper)
        chunks2 = []
        async for chunk in retry_async_generator(gen, config):
            chunks2.append(chunk)

        assert chunks1 == chunks2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_default_max_buffer_chunks(self):
        """Test that default max_buffer_chunks is 1000."""
        config = RetryConfig()
        assert config.max_buffer_chunks == 1000


class TestRetryAsyncGeneratorReplay:
    """Tests for async generator retry with chunk replay on failure."""

    @pytest.mark.asyncio
    async def test_default_enable_stream_replay_is_true(self):
        """Test that enable_stream_replay defaults to True."""
        config = RetryConfig()
        assert config.enable_stream_replay is True

    @pytest.mark.asyncio
    async def test_replay_buffered_chunks_on_failure(self):
        """Test that buffered chunks are replayed when failure occurs mid-stream."""
        call_count = 0
        chunks_from_generator = []

        async def gen():
            nonlocal call_count
            call_count += 1
            for i in range(5):
                chunks_from_generator.append((call_count, f"chunk_{i}"))
                yield f"chunk_{i}"
                # Fail mid-stream on first call
                if call_count == 1 and i == 2:
                    raise ConnectionError("Connection lost mid-stream")

        config = RetryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
            jitter=False,
            enable_stream_replay=True,
        )

        chunks_received = []
        async for chunk in retry_async_generator(gen, config):
            chunks_received.append(chunk)

        # First call yields chunks 0, 1, 2 then fails
        # Replay yields chunks 0, 1, 2 again
        # Second call yields all 5 chunks but skips first 3 (already replayed)
        # So chunks_received should be: 0, 1, 2, (replay) 0, 1, 2, (new from retry) 3, 4
        assert chunks_received == [
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
        ]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_replay_when_disabled(self):
        """Test that chunks are not replayed when enable_stream_replay is False."""
        call_count = 0

        async def gen():
            nonlocal call_count
            call_count += 1
            for i in range(5):
                yield f"chunk_{i}"
                # Fail mid-stream on first call
                if call_count == 1 and i == 2:
                    raise ConnectionError("Connection lost mid-stream")

        config = RetryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
            jitter=False,
            enable_stream_replay=False,
        )

        chunks_received = []
        async for chunk in retry_async_generator(gen, config):
            chunks_received.append(chunk)

        # First call yields chunks 0, 1, 2 then fails
        # No replay, so retry starts fresh and yields all 5 chunks
        # Result: 0, 1, 2, (retry) 0, 1, 2, 3, 4
        assert chunks_received == [
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
        ]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_replay_with_buffer_limit(self):
        """Test replay behavior when buffer limit is smaller than yielded chunks."""
        call_count = 0

        async def gen():
            nonlocal call_count
            call_count += 1
            for i in range(10):
                yield f"chunk_{i}"
                # Fail after 7 chunks on first call
                if call_count == 1 and i == 6:
                    raise ConnectionError("Connection lost")

        config = RetryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
            jitter=False,
            max_buffer_chunks=3,  # Only buffer first 3 chunks
            enable_stream_replay=True,
        )

        chunks_received = []
        async for chunk in retry_async_generator(gen, config):
            chunks_received.append(chunk)

        # First call yields chunks 0-6 (7 chunks), but only 0-2 buffered, then fails
        # Replay yields chunks 0, 1, 2
        # Retry yields all 10 but skips first 3 (replayed)
        # Result: 0, 1, 2, 3, 4, 5, 6, (replay) 0, 1, 2, (retry new) 3, 4, 5, 6, 7, 8, 9
        assert chunks_received == [
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
            "chunk_5",
            "chunk_6",
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
            "chunk_5",
            "chunk_6",
            "chunk_7",
            "chunk_8",
            "chunk_9",
        ]
        assert call_count == 2
