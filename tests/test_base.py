"""Tests for LLM base types."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from services.llm_service.core.llm.base import (
    BaseLLMClient,
    ImageContent,
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
    TextContent,
    content_to_string,
    to_langchain_content,
)
from services.llm_service.core.llm.credentials import CredentialProvider


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_simple_text_content(self):
        """Test message with simple string content."""
        msg = LLMMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.get_text_content() == "Hello, world!"
        assert not msg.is_multimodal()

    def test_multimodal_content(self):
        """Test message with text and image content."""
        msg = LLMMessage(
            role="user",
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(source_type="base64", media_type="image/png", data="abc123"),
            ],
        )
        assert msg.is_multimodal()
        assert msg.get_text_content() == "What's in this image?"

    def test_tool_calls(self):
        """Test message with tool calls."""
        msg = LLMMessage(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "tc_123", "name": "get_weather", "args": {"location": "NYC"}},
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    def test_tool_response(self):
        """Test tool response message."""
        msg = LLMMessage(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="tc_123",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_123"
        assert msg.name == "get_weather"


class TestContentConversion:
    """Tests for content conversion utilities."""

    def test_content_to_string_none(self):
        """Test with None content."""
        assert content_to_string(None) == ""

    def test_content_to_string_simple(self):
        """Test with simple string."""
        assert content_to_string("Hello") == "Hello"

    def test_content_to_string_multimodal(self):
        """Test with multimodal content blocks."""
        content = [
            TextContent(text="First"),
            ImageContent(source_type="url", media_type="image/png", data="http://example.com/img.png"),
            TextContent(text="Second"),
        ]
        result = content_to_string(content)
        assert "First" in result
        assert "Second" in result

    def test_to_langchain_content_string(self):
        """Test LangChain conversion with string."""
        result = to_langchain_content("Hello")
        assert result == [{"type": "text", "text": "Hello"}]

    def test_to_langchain_content_multimodal(self):
        """Test LangChain conversion with multimodal content."""
        content = [
            TextContent(text="Check this:"),
            ImageContent(source_type="base64", media_type="image/png", data="abc123"),
        ]
        result = to_langchain_content(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in result[1]["image_url"]["url"]


class TestLLMToolDefinition:
    """Tests for LLMToolDefinition dataclass."""

    def test_basic_tool(self):
        """Test basic tool definition."""
        tool = LLMToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
            audience="search-api",
            scopes=["search.read"],
        )
        assert tool.name == "search"
        assert tool.audience == "search-api"
        assert tool.scopes == ["search.read"]


class TestLLMChunk:
    """Tests for LLMChunk dataclass."""

    def test_text_delta(self):
        """Test chunk with text delta."""
        chunk = LLMChunk(delta="Hello")
        assert chunk.delta == "Hello"
        assert chunk.tool_calls is None
        assert chunk.finish_reason is None

    def test_final_chunk(self):
        """Test final chunk with finish reason."""
        chunk = LLMChunk(finish_reason="stop")
        assert chunk.finish_reason == "stop"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_text_response(self):
        """Test response with text content."""
        response = LLMResponse(content="Hello!", finish_reason="stop")
        assert response.content == "Hello!"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_tool_response(self):
        """Test response with tool calls."""
        from shared.protocol.tool_models import ToolCall

        response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    tool_call_id="tc_123",
                    name="get_weather",
                    args={"location": "NYC"},
                    audience="weather-api",
                    scopes=[],
                ),
            ],
            finish_reason="tool_calls",
        )
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"


class TestBaseLLMClientTemplateMethod:
    """Tests for BaseLLMClient template method pattern."""

    def test_initialize_client_calls_hooks_in_order(self):
        """Test that _initialize_client() calls hooks in correct order."""
        call_order: list[str] = []

        class MockCredentialProvider:
            """Mock credential provider for testing."""

            def get_credentials(self) -> dict[str, Any]:
                call_order.append("get_credentials")
                return {"api_key": "test-key"}

        class TestClient(BaseLLMClient):
            """Test client that tracks method call order."""

            def _get_credential_provider(self) -> CredentialProvider:
                call_order.append("get_credential_provider")
                return MockCredentialProvider()  # type: ignore[return-value]

            def _get_endpoint(self) -> str:
                call_order.append("get_endpoint")
                return "https://api.example.com"

            def _create_client_instance(
                self, credentials: dict[str, Any], endpoint: str
            ) -> Any:
                call_order.append("create_client_instance")
                return {"credentials": credentials, "endpoint": endpoint}

            async def generate_stream(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> AsyncGenerator[LLMChunk, None]:
                yield LLMChunk(delta="test")

            async def generate(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                return LLMResponse(content="test")

            async def close(self) -> None:
                pass

        client = TestClient()
        result = client._initialize_client()

        # Verify call order: get_credential_provider -> get_credentials -> get_endpoint -> create_client_instance
        assert call_order == [
            "get_credential_provider",
            "get_credentials",
            "get_endpoint",
            "create_client_instance",
        ]

        # Verify the result contains the correct data
        assert result["credentials"] == {"api_key": "test-key"}
        assert result["endpoint"] == "https://api.example.com"

    def test_client_property_lazy_initialization(self):
        """Test that client property lazily initializes the client."""
        create_count = 0

        class MockCredentialProvider:
            def get_credentials(self) -> dict[str, Any]:
                return {"api_key": "test-key"}

        class TestClient(BaseLLMClient):
            def _get_credential_provider(self) -> CredentialProvider:
                return MockCredentialProvider()  # type: ignore[return-value]

            def _get_endpoint(self) -> str:
                return "https://api.example.com"

            def _create_client_instance(
                self, credentials: dict[str, Any], endpoint: str
            ) -> Any:
                nonlocal create_count
                create_count += 1
                return MagicMock()

            async def generate_stream(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> AsyncGenerator[LLMChunk, None]:
                yield LLMChunk(delta="test")

            async def generate(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                return LLMResponse(content="test")

            async def close(self) -> None:
                pass

        client = TestClient()

        # Client not created yet
        assert create_count == 0

        # First access creates the client
        _ = client.client
        assert create_count == 1

        # Subsequent accesses return cached instance
        _ = client.client
        _ = client.client
        assert create_count == 1

    def test_initialize_client_without_credential_provider_raises(self):
        """Test that _initialize_client() raises if no credential provider."""

        class TestClient(BaseLLMClient):
            # Does not override _get_credential_provider, so it returns None

            async def generate_stream(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> AsyncGenerator[LLMChunk, None]:
                yield LLMChunk(delta="test")

            async def generate(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                return LLMResponse(content="test")

            async def close(self) -> None:
                pass

        client = TestClient()

        with pytest.raises(NotImplementedError) as exc_info:
            client._initialize_client()

        assert "_get_credential_provider()" in str(exc_info.value)

    def test_create_client_instance_not_implemented_raises(self):
        """Test that _create_client_instance() raises NotImplementedError by default."""

        class MockCredentialProvider:
            def get_credentials(self) -> dict[str, Any]:
                return {"api_key": "test-key"}

        class TestClient(BaseLLMClient):
            def _get_credential_provider(self) -> CredentialProvider:
                return MockCredentialProvider()  # type: ignore[return-value]

            # Does not override _create_client_instance

            async def generate_stream(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> AsyncGenerator[LLMChunk, None]:
                yield LLMChunk(delta="test")

            async def generate(
                self,
                messages: list[LLMMessage],
                tools: list[LLMToolDefinition] | None = None,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                return LLMResponse(content="test")

            async def close(self) -> None:
                pass

        client = TestClient()

        with pytest.raises(NotImplementedError) as exc_info:
            client._initialize_client()

        assert "_create_client_instance()" in str(exc_info.value)
