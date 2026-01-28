"""OpenAI-compatible mixin for LLM clients.

This mixin provides common functionality for OpenAI-compatible API clients:
- Message format conversion
- Tool format conversion
- Tool call aggregation from streaming chunks
- Non-streaming generate() from streaming

The mixin reduces code duplication across OpenAI, OpenRouter, and Azure OpenAI clients.
"""

import json
from typing import Any

from services.llm_service.core.llm.base import (
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
)
from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall
from shared.validators.id_generators import generate_tool_call_id


class OpenAICompatibleMixin:
    """Mixin providing common OpenAI-compatible functionality.

    This mixin should be used with BaseLLMClient subclasses that use
    the OpenAI API format (including OpenRouter and Azure OpenAI).

    Provides:
    - _convert_messages(): Convert LLMMessage to OpenAI format
    - _convert_tools(): Convert LLMToolDefinition to OpenAI format
    - _convert_tool_calls(): Convert raw tool calls to ToolCall objects
    - _aggregate_stream_to_response(): Build LLMResponse from stream chunks
    """

    def _convert_messages(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert LLMMessage list to OpenAI message format.

        Args:
            messages: List of LLM messages.
            system_prompt: Optional system prompt to prepend.

        Returns:
            List of dicts in OpenAI message format.
        """
        result = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            converted: dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                converted["content"] = msg.content

            if msg.tool_calls:
                # Ensure each tool call has required 'id' and 'type' fields
                # Note: ToolCall model uses 'tool_call_id' but OpenAI expects 'id'
                tool_calls_converted = []
                for tc in msg.tool_calls:
                    # Convert args to JSON string if it's a dict
                    args = tc.get("args", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else (args or "{}")
                    tool_calls_converted.append(
                        {
                            "id": tc.get("id") or tc.get("tool_call_id") or generate_tool_call_id(),
                            "type": tc.get("type", "function"),
                            "function": tc.get(
                                "function",
                                {
                                    "name": tc.get("name", ""),
                                    "arguments": args_str,
                                },
                            ),
                        }
                    )
                converted["tool_calls"] = tool_calls_converted

            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id

            if msg.name:
                converted["name"] = msg.name

            result.append(converted)

        return result

    def _convert_tools(
        self,
        tools: list[LLMToolDefinition] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert LLMToolDefinition list to OpenAI tool format.

        Args:
            tools: List of tool definitions.

        Returns:
            List of dicts in OpenAI tool format, or None if no tools.
        """
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _convert_tool_calls(
        self,
        tool_calls_raw: list[dict[str, Any]],
        tools: list[LLMToolDefinition] | None,
    ) -> list[ToolCall]:
        """Convert raw tool call dicts to ToolCall objects.

        Args:
            tool_calls_raw: Raw tool call data from API response.
            tools: Tool definitions (for audience/scopes lookup).

        Returns:
            List of ToolCall objects.
        """
        if not tool_calls_raw:
            return []

        # Build lookup for tool definitions
        tool_defs = {t.name: t for t in (tools or [])}

        result = []
        for tc in tool_calls_raw:
            func = tc.get("function", {})
            name = func.get("name", "")
            tool_def = tool_defs.get(name)

            result.append(
                ToolCall(
                    tool_call_id=tc.get("id") or generate_tool_call_id(),
                    name=name,
                    args=json.loads(func.get("arguments", "{}")),
                    audience=tool_def.audience if tool_def else "internal",
                    scopes=tool_def.scopes if tool_def else [],
                )
            )

        return result

    async def _aggregate_stream_to_response(
        self,
        stream_generator,
        tools: list[LLMToolDefinition] | None = None,
    ) -> LLMResponse:
        """Aggregate streaming chunks into a complete LLMResponse.

        Args:
            stream_generator: Async generator yielding LLMChunk objects.
            tools: Tool definitions (for audience/scopes in tool calls).

        Returns:
            Complete LLMResponse with aggregated content.
        """
        content_parts = []
        tool_calls_raw: list[dict[str, Any]] = []
        finish_reason = "stop"
        usage: Usage | None = None

        async for chunk in stream_generator:
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.tool_calls:
                tool_calls_raw = chunk.tool_calls
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        # Convert raw tool calls to ToolCall objects
        tool_calls = self._convert_tool_calls(tool_calls_raw, tools)

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


class StreamingToolCallAggregator:
    """Aggregates tool call chunks from streaming responses.

    Tool calls arrive in multiple chunks during streaming. This class
    tracks and combines them into complete tool call dictionaries.

    Usage:
        aggregator = StreamingToolCallAggregator()
        for tc in delta.tool_calls:
            aggregator.add_chunk(tc)
        complete_calls = aggregator.get_complete_calls()
    """

    def __init__(self):
        """Initialize the aggregator."""
        self._chunks: dict[int, dict[str, Any]] = {}

    def add_chunk(self, tool_call_chunk: Any) -> None:
        """Add a tool call chunk to the aggregator.

        Args:
            tool_call_chunk: Tool call delta from streaming response.
                Expected to have: index, id, function.name, function.arguments
        """
        idx = tool_call_chunk.index

        if idx not in self._chunks:
            self._chunks[idx] = {
                "id": tool_call_chunk.id or "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        if tool_call_chunk.id:
            self._chunks[idx]["id"] = tool_call_chunk.id

        if tool_call_chunk.function:
            if tool_call_chunk.function.name:
                self._chunks[idx]["function"]["name"] = tool_call_chunk.function.name
            if tool_call_chunk.function.arguments:
                self._chunks[idx]["function"]["arguments"] += tool_call_chunk.function.arguments

    def get_complete_calls(self) -> list[dict[str, Any]] | None:
        """Get all complete tool calls.

        Returns:
            List of complete tool call dicts, or None if no calls.
        """
        if not self._chunks:
            return None
        return list(self._chunks.values())

    def has_calls(self) -> bool:
        """Check if any tool calls have been aggregated.

        Returns:
            True if there are tool calls.
        """
        return bool(self._chunks)

    def clear(self) -> None:
        """Clear all aggregated chunks."""
        self._chunks.clear()
