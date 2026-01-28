"""Type definitions for LLM tool calls.

This module provides TypedDict definitions for structured typing of
tool call data from LLM responses.
"""

from typing import TypedDict


class LLMToolCallFunction(TypedDict):
    """Function component of an LLM tool call.

    Attributes:
        name: The name of the tool being called.
        arguments: JSON-encoded string of the tool arguments.
    """

    name: str
    arguments: str


class LLMToolCallChunk(TypedDict):
    """Raw tool call data from an LLM response.

    Attributes:
        id: Unique identifier for the tool call.
        function: The function details (name and arguments).
    """

    id: str
    function: LLMToolCallFunction


class ToolCallExtensions(TypedDict, total=False):
    """Optional extensions for tool call processing.

    These are attached to ToolCall objects when additional metadata
    is needed during execution.

    Attributes:
        raw_tool_call_id: Original tool call ID before normalization.
        parse_error: Error message if JSON argument parsing failed.
    """

    raw_tool_call_id: str
    parse_error: str
