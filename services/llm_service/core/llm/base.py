"""Base LLM client interface.

Defines the abstract interface for LLM providers and multimodal content types.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall

# =============================================================================
# Multimodal Content Types
# =============================================================================


class TextContent(BaseModel):
    """Text content block for multimodal messages."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block for multimodal messages.

    Supports two source types:
    - base64: Image data encoded as base64 string
    - url: Direct URL to the image (for providers that support it)
    """

    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"]
    media_type: str  # MIME type: "image/png", "image/jpeg", "image/gif", "image/webp"
    data: str  # base64 encoded data or URL depending on source_type


# Union type for all content block types
ContentBlock = TextContent | ImageContent


def to_langchain_content(content: str | list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert LLMMessage content to LangChain multimodal format.

    Args:
        content: Either a string or list of ContentBlock objects.

    Returns:
        List of content dictionaries in LangChain format:
        - {"type": "text", "text": "..."} for text
        - {"type": "image_url", "image_url": {"url": "..."}} for images
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            if block.source_type == "base64":
                # Format: data:{media_type};base64,{data}
                image_url = f"data:{block.media_type};base64,{block.data}"
            else:
                # Direct URL
                image_url = block.data
            result.append({"type": "image_url", "image_url": {"url": image_url}})
    return result


def content_to_string(content: str | list[ContentBlock] | None) -> str:
    """Extract text content as a string.

    Useful for cases where only text content is needed (e.g., logging, display).

    Args:
        content: String, list of ContentBlock, or None.

    Returns:
        Concatenated text from all TextContent blocks, or empty string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
    return "\n".join(text_parts)


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class LLMMessage:
    """A message in the LLM conversation.

    Supports both simple string content and multimodal content (text + images).
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[ContentBlock] | None = None  # String or multimodal blocks
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Tool name for tool responses

    def get_text_content(self) -> str:
        """Get the text content as a string.

        For multimodal content, concatenates all TextContent blocks.
        """
        return content_to_string(self.content)

    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content (images)."""
        if not isinstance(self.content, list):
            return False
        return any(isinstance(block, ImageContent) for block in self.content)


@dataclass
class LLMToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    audience: str  # API/service identifier for OBO token
    scopes: list[str]  # OAuth scopes required


@dataclass
class LLMChunk:
    """A streaming chunk from the LLM."""

    delta: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None  # "stop", "tool_calls", "length", "content_filter"
    usage: Usage | None = None


@dataclass
class LLMResponse:
    """Complete response from the LLM."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage | None = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            system_prompt: System prompt to prepend.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Yields:
            LLMChunk objects with deltas and metadata.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            system_prompt: System prompt to prepend.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Returns:
            Complete LLMResponse.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        ...
