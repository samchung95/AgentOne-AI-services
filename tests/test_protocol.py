"""Tests for shared protocol models."""

import pytest

from shared.protocol.common import (
    CanvasSource,
    CanvasUpdate,
    ErrorCode,
    ErrorInfo,
    Usage,
)
from shared.protocol.tool_models import (
    DelegationResult,
    DelegationStatus,
    ToolCall,
    ToolResult,
    ToolStatus,
)


class TestUsage:
    """Tests for Usage model."""

    def test_valid_usage(self):
        """Test creating valid usage."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model_name="gpt-4",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model_name == "gpt-4"

    def test_usage_without_total(self):
        """Test usage without total_tokens (optional)."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            model_name="gpt-4",
        )
        assert usage.total_tokens is None

    def test_usage_negative_tokens_rejected(self):
        """Test that negative token counts are rejected."""
        with pytest.raises(ValueError):
            Usage(input_tokens=-1, output_tokens=50, model_name="gpt-4")


class TestErrorInfo:
    """Tests for ErrorInfo model."""

    def test_valid_error_info(self):
        """Test creating valid error info."""
        error = ErrorInfo(
            code="LLM_ERROR",
            message="Something went wrong",
            retryable=True,
            details={"provider": "openai"},
        )
        assert error.code == "LLM_ERROR"
        assert error.message == "Something went wrong"
        assert error.retryable is True
        assert error.details == {"provider": "openai"}

    def test_error_code_enum(self):
        """Test using ErrorCode enum."""
        error = ErrorInfo(
            code=ErrorCode.LLM_TIMEOUT.value,
            message="Request timed out",
        )
        assert error.code == "LLM_TIMEOUT"


class TestToolCall:
    """Tests for ToolCall model."""

    def test_valid_tool_call(self):
        """Test creating valid tool call."""
        tc = ToolCall(
            tool_call_id="tc_123abc",
            name="get_weather",
            args={"location": "NYC"},
            audience="weather-api",
            scopes=["weather.read"],
        )
        assert tc.tool_call_id == "tc_123abc"
        assert tc.name == "get_weather"
        assert tc.args == {"location": "NYC"}

    def test_tool_call_with_extensions(self):
        """Test tool call with extensions."""
        tc = ToolCall(
            tool_call_id="tc_456def",
            name="search",
            args={},
            audience="search",
            scopes=[],
            extensions={"custom_field": "value"},
        )
        assert tc.extensions == {"custom_field": "value"}

    def test_tool_call_id_patterns(self):
        """Test various valid tool call ID patterns."""
        # tc_ prefix (our internal format)
        tc1 = ToolCall(
            tool_call_id="tc_abc123",
            name="test",
            args={},
            audience="test",
            scopes=[],
        )
        assert tc1.tool_call_id == "tc_abc123"

        # call_ prefix (OpenAI format)
        tc2 = ToolCall(
            tool_call_id="call_abc123XYZ",
            name="test",
            args={},
            audience="test",
            scopes=[],
        )
        assert tc2.tool_call_id == "call_abc123XYZ"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_successful_result(self):
        """Test creating successful tool result."""
        result = ToolResult(
            tool_call_id="tc_123abc",
            name="get_weather",
            status=ToolStatus.SUCCESS,
            data={"temperature": 72, "conditions": "sunny"},
            duration_ms=150,
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.data == {"temperature": 72, "conditions": "sunny"}

    def test_error_result_requires_error(self):
        """Test that error result requires error field."""
        with pytest.raises(ValueError, match="error is required"):
            ToolResult(
                tool_call_id="tc_123abc",
                name="failing_tool",
                status=ToolStatus.ERROR,
            )

    def test_error_result_with_error(self):
        """Test error result with error info."""
        result = ToolResult(
            tool_call_id="tc_123abc",
            name="failing_tool",
            status=ToolStatus.ERROR,
            error=ErrorInfo(code="TOOL_ERROR", message="Tool failed"),
        )
        assert result.status == ToolStatus.ERROR
        assert result.error is not None

    def test_result_with_canvas_update(self):
        """Test result with canvas update."""
        from shared.protocol.common import CanvasData

        result = ToolResult(
            tool_call_id="tc_123abc",
            name="update_document",
            status=ToolStatus.SUCCESS,
            canvas_update=CanvasUpdate(
                component="document-editor",
                data=CanvasData(source=CanvasSource.TOOL, content="Updated"),
            ),
        )
        assert result.canvas_update is not None
        assert result.canvas_update.component == "document-editor"


class TestDelegationResult:
    """Tests for DelegationResult model."""

    def test_successful_delegation(self):
        """Test creating successful delegation result."""
        result = DelegationResult(
            agent_id="summarizer",
            task="Summarize the document",
            status=DelegationStatus.SUCCESS,
            response="Here is the summary...",
            input_tokens=100,
            output_tokens=50,
            duration_ms=2000,
        )
        assert result.status == DelegationStatus.SUCCESS
        assert result.response == "Here is the summary..."

    def test_failed_delegation_requires_error(self):
        """Test that failed delegation requires error."""
        with pytest.raises(ValueError, match="error is required"):
            DelegationResult(
                agent_id="summarizer",
                task="Summarize",
                status=DelegationStatus.ERROR,
            )

    def test_timeout_delegation(self):
        """Test timeout delegation result."""
        result = DelegationResult(
            agent_id="slow_agent",
            task="Long running task",
            status=DelegationStatus.TIMEOUT,
            error=ErrorInfo(code="TIMEOUT", message="Agent timed out"),
            duration_ms=30000,
        )
        assert result.status == DelegationStatus.TIMEOUT
        assert result.error is not None
