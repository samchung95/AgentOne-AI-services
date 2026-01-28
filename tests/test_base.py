"""Tests for LLM base types."""

import pytest

from services.llm_service.core.llm.base import (
    ContentBlock,
    ImageContent,
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
    TextContent,
    content_to_string,
    to_langchain_content,
)


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
