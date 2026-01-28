"""Tests for OpenAI-compatible mixin."""

import pytest

from services.llm_service.core.llm.base import LLMMessage, LLMToolDefinition
from services.llm_service.core.llm.mixin import OpenAICompatibleMixin, StreamingToolCallAggregator


class TestOpenAICompatibleMixin:
    """Tests for OpenAICompatibleMixin."""

    @pytest.fixture
    def mixin(self):
        """Create a mixin instance."""

        class TestClient(OpenAICompatibleMixin):
            pass

        return TestClient()

    def test_convert_messages_simple(self, mixin):
        """Test converting simple messages."""
        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there!"),
        ]

        result = mixin._convert_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"

    def test_convert_messages_with_system_prompt(self, mixin):
        """Test converting messages with system prompt prepended."""
        messages = [LLMMessage(role="user", content="Hello")]

        result = mixin._convert_messages(messages, system_prompt="You are helpful")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"

    def test_convert_messages_with_tool_calls(self, mixin):
        """Test converting messages with tool calls."""
        messages = [
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"location": "NYC"}},
                ],
            ),
        ]

        result = mixin._convert_messages(messages)

        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_convert_messages_tool_response(self, mixin):
        """Test converting tool response messages."""
        messages = [
            LLMMessage(
                role="tool",
                content='{"temp": 72}',
                tool_call_id="tc_123",
                name="get_weather",
            ),
        ]

        result = mixin._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_123"
        assert result[0]["name"] == "get_weather"

    def test_convert_tools(self, mixin):
        """Test converting tool definitions."""
        tools = [
            LLMToolDefinition(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {}},
                audience="search-api",
                scopes=[],
            ),
        ]

        result = mixin._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"

    def test_convert_tools_none(self, mixin):
        """Test converting None tools."""
        assert mixin._convert_tools(None) is None
        assert mixin._convert_tools([]) is None

    def test_convert_tool_calls(self, mixin):
        """Test converting raw tool calls to ToolCall objects."""
        tools = [
            LLMToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={},
                audience="weather-api",
                scopes=["weather.read"],
            ),
        ]

        raw_calls = [
            {
                "id": "tc_123",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            },
        ]

        result = mixin._convert_tool_calls(raw_calls, tools)

        assert len(result) == 1
        assert result[0].tool_call_id == "tc_123"
        assert result[0].name == "get_weather"
        assert result[0].args == {"location": "NYC"}
        assert result[0].audience == "weather-api"
        assert result[0].scopes == ["weather.read"]

    def test_convert_tool_calls_unknown_tool(self, mixin):
        """Test converting tool calls for unknown tools."""
        raw_calls = [
            {
                "id": "tc_456",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            },
        ]

        result = mixin._convert_tool_calls(raw_calls, [])

        assert len(result) == 1
        assert result[0].name == "unknown_tool"
        assert result[0].audience == "internal"  # Default
        assert result[0].scopes == []  # Default


class TestStreamingToolCallAggregator:
    """Tests for StreamingToolCallAggregator."""

    def test_empty_aggregator(self):
        """Test empty aggregator."""
        agg = StreamingToolCallAggregator()
        assert not agg.has_calls()
        assert agg.get_complete_calls() is None

    def test_aggregate_single_tool_call(self):
        """Test aggregating a single tool call."""

        class MockChunk:
            def __init__(self, index, id=None, name=None, arguments=None):
                self.index = index
                self.id = id

                class MockFunction:
                    pass

                self.function = MockFunction()
                self.function.name = name
                self.function.arguments = arguments

        agg = StreamingToolCallAggregator()

        # First chunk with ID and name
        agg.add_chunk(MockChunk(0, id="tc_123", name="search"))

        # Subsequent chunks with arguments
        agg.add_chunk(MockChunk(0, arguments='{"query":'))
        agg.add_chunk(MockChunk(0, arguments=' "test"}'))

        assert agg.has_calls()
        calls = agg.get_complete_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == "tc_123"
        assert calls[0]["function"]["name"] == "search"
        assert calls[0]["function"]["arguments"] == '{"query": "test"}'

    def test_aggregate_multiple_tool_calls(self):
        """Test aggregating multiple tool calls."""

        class MockChunk:
            def __init__(self, index, id=None, name=None, arguments=None):
                self.index = index
                self.id = id

                class MockFunction:
                    pass

                self.function = MockFunction()
                self.function.name = name
                self.function.arguments = arguments

        agg = StreamingToolCallAggregator()

        # First tool call
        agg.add_chunk(MockChunk(0, id="tc_1", name="tool_a", arguments="{}"))

        # Second tool call
        agg.add_chunk(MockChunk(1, id="tc_2", name="tool_b", arguments="{}"))

        calls = agg.get_complete_calls()
        assert len(calls) == 2
        assert calls[0]["id"] == "tc_1"
        assert calls[1]["id"] == "tc_2"

    def test_clear(self):
        """Test clearing aggregated calls."""

        class MockChunk:
            def __init__(self):
                self.index = 0
                self.id = "tc_123"

                class MockFunction:
                    name = "test"
                    arguments = None

                self.function = MockFunction()

        agg = StreamingToolCallAggregator()
        agg.add_chunk(MockChunk())
        assert agg.has_calls()

        agg.clear()
        assert not agg.has_calls()
