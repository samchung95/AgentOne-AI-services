"""Cross-provider test suite for validating consistent behavior across LLM providers.

This module contains parameterized tests that verify all LLM providers implement
the expected interface correctly and return consistent response types.

Usage:
    # Run with mocked providers (no API keys required)
    pytest --mock-providers tests/test_cross_provider.py

    # Run with real providers (requires API keys)
    pytest tests/test_cross_provider.py
"""

import pytest

from services.llm_service.core.config.constants import ProviderID
from services.llm_service.core.llm.base import (
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
)
from shared.protocol.tool_models import ToolCall

# All provider types to test
ALL_PROVIDERS = [
    ProviderID.OPENAI,
    ProviderID.OPENROUTER,
    ProviderID.AZURE_OPENAI,
    ProviderID.VERTEX_AI,
    ProviderID.REMOTE,
]


@pytest.fixture
def messages() -> list[LLMMessage]:
    """Sample messages for testing."""
    return [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def tools() -> list[LLMToolDefinition]:
    """Sample tool definitions for testing."""
    return [
        LLMToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
            audience="weather-api",
            scopes=["weather.read"],
        ),
        LLMToolDefinition(
            name="get_stock_price",
            description="Get the current stock price",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                },
                "required": ["symbol"],
            },
            audience="stock-api",
            scopes=["stocks.read"],
        ),
    ]


def _get_mock_client(provider: ProviderID) -> type:
    """Get the mocked client class for a provider.

    When --mock-providers flag is used, these are the MockLLMClient classes
    that have been patched in place of the real clients.
    """
    if provider == ProviderID.OPENAI:
        from services.llm_service.core.llm.openai_client import OpenAIClient

        return OpenAIClient
    elif provider == ProviderID.OPENROUTER:
        from services.llm_service.core.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient
    elif provider == ProviderID.AZURE_OPENAI:
        from services.llm_service.core.llm.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient
    elif provider == ProviderID.VERTEX_AI:
        from services.llm_service.core.llm.vertex import VertexAIClient

        return VertexAIClient
    elif provider == ProviderID.REMOTE:
        from services.llm_service.core.llm.remote_client import RemoteLLMClient

        return RemoteLLMClient
    else:
        raise ValueError(f"Unknown provider: {provider}")


class TestBasicCompletion:
    """Tests for basic completion functionality across providers."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_returns_llm_response(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that generate() returns an LLMResponse with content."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages)

        assert isinstance(response, LLMResponse), (
            f"{provider}: generate() should return LLMResponse"
        )
        assert isinstance(response.content, str), (
            f"{provider}: response.content should be a string"
        )
        assert len(response.content) > 0, (
            f"{provider}: response.content should not be empty"
        )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_returns_finish_reason(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that generate() returns a finish_reason."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages)

        assert response.finish_reason is not None, (
            f"{provider}: response.finish_reason should not be None"
        )
        assert response.finish_reason in ["stop", "tool_calls", "length"], (
            f"{provider}: unexpected finish_reason: {response.finish_reason}"
        )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_returns_usage_stats(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that generate() returns usage statistics."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages)

        assert response.usage is not None, (
            f"{provider}: response.usage should not be None"
        )
        assert hasattr(response.usage, "input_tokens"), (
            f"{provider}: usage should have input_tokens"
        )
        assert hasattr(response.usage, "output_tokens"), (
            f"{provider}: usage should have output_tokens"
        )
        assert hasattr(response.usage, "model_name"), (
            f"{provider}: usage should have model_name"
        )


class TestStreamingCompletion:
    """Tests for streaming completion functionality across providers."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_stream_yields_llm_chunks(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that generate_stream() yields LLMChunk objects."""
        client_class = _get_mock_client(provider)
        client = client_class()

        chunks = []
        async for chunk in client.generate_stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0, (
            f"{provider}: generate_stream() should yield at least one chunk"
        )
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, LLMChunk), (
                f"{provider}: chunk {i} should be LLMChunk, got {type(chunk)}"
            )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_stream_chunks_have_delta(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that streaming chunks have delta content."""
        client_class = _get_mock_client(provider)
        client = client_class()

        deltas = []
        async for chunk in client.generate_stream(messages):
            if chunk.delta:
                deltas.append(chunk.delta)

        assert len(deltas) > 0, (
            f"{provider}: at least one chunk should have delta content"
        )
        # All deltas should be strings
        for delta in deltas:
            assert isinstance(delta, str), (
                f"{provider}: delta should be a string"
            )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_stream_last_chunk_has_finish_reason(
        self, provider: ProviderID, messages: list[LLMMessage]
    ) -> None:
        """Verify that the last streaming chunk has a finish_reason."""
        client_class = _get_mock_client(provider)
        client = client_class()

        chunks = []
        async for chunk in client.generate_stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0, f"{provider}: should yield at least one chunk"
        last_chunk = chunks[-1]
        assert last_chunk.finish_reason is not None, (
            f"{provider}: last chunk should have finish_reason"
        )


class TestToolCalling:
    """Tests for tool calling functionality across providers."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_generate_with_tools_returns_tool_calls(
        self,
        provider: ProviderID,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
    ) -> None:
        """Verify that generate() with tools returns ToolCall objects."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages, tools=tools)

        assert isinstance(response.tool_calls, list), (
            f"{provider}: response.tool_calls should be a list"
        )
        assert len(response.tool_calls) > 0, (
            f"{provider}: should return at least one tool call"
        )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_tool_calls_are_valid_tool_call_objects(
        self,
        provider: ProviderID,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
    ) -> None:
        """Verify that each tool call is a valid ToolCall object."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages, tools=tools)

        for i, tc in enumerate(response.tool_calls):
            assert isinstance(tc, ToolCall), (
                f"{provider}: tool_call {i} should be ToolCall, got {type(tc)}"
            )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_tool_calls_have_valid_ids(
        self,
        provider: ProviderID,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
    ) -> None:
        """Verify that tool calls have valid normalized IDs."""
        import re

        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages, tools=tools)

        # Valid ID patterns: tc_*, call_*, toolu_*
        id_pattern = re.compile(r"^(tc_|call_|toolu_)[A-Za-z0-9_-]+$")

        for i, tc in enumerate(response.tool_calls):
            assert tc.tool_call_id is not None, (
                f"{provider}: tool_call {i} should have tool_call_id"
            )
            assert id_pattern.match(tc.tool_call_id), (
                f"{provider}: tool_call_id '{tc.tool_call_id}' should match pattern"
            )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_tool_calls_have_name_and_args(
        self,
        provider: ProviderID,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
    ) -> None:
        """Verify that tool calls have name and args."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages, tools=tools)

        for i, tc in enumerate(response.tool_calls):
            assert tc.name is not None and len(tc.name) > 0, (
                f"{provider}: tool_call {i} should have a name"
            )
            assert isinstance(tc.args, dict), (
                f"{provider}: tool_call {i} args should be a dict"
            )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_tool_calls_have_audience_and_scopes(
        self,
        provider: ProviderID,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
    ) -> None:
        """Verify that tool calls have audience and scopes from tool definitions."""
        client_class = _get_mock_client(provider)
        client = client_class()

        response = await client.generate(messages, tools=tools)

        for i, tc in enumerate(response.tool_calls):
            assert tc.audience is not None and len(tc.audience) > 0, (
                f"{provider}: tool_call {i} should have audience"
            )
            assert isinstance(tc.scopes, list) and len(tc.scopes) > 0, (
                f"{provider}: tool_call {i} should have scopes list"
            )


class TestClientLifecycle:
    """Tests for client lifecycle methods across providers."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_client_has_close_method(self, provider: ProviderID) -> None:
        """Verify that all clients have a close() method."""
        client_class = _get_mock_client(provider)
        client = client_class()

        assert hasattr(client, "close"), (
            f"{provider}: client should have close() method"
        )
        # Calling close should not raise
        await client.close()

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    @pytest.mark.usefixtures("mock_providers")
    async def test_client_has_from_model_config_classmethod(
        self, provider: ProviderID
    ) -> None:
        """Verify that all clients have from_model_config class method."""
        client_class = _get_mock_client(provider)

        assert hasattr(client_class, "from_model_config"), (
            f"{provider}: client should have from_model_config classmethod"
        )
