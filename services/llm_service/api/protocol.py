"""HTTP protocol models for LLM Service API."""

from typing import Any

from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    """Content block for multimodal messages."""

    type: str  # "text" or "image"
    text: str | None = None
    source_type: str | None = None  # "base64" or "url" for images
    media_type: str | None = None
    data: str | None = None


class MessageRequest(BaseModel):
    """Message in the request."""

    role: str
    content: str | list[ContentBlock] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinitionRequest(BaseModel):
    """Tool definition in the request."""

    name: str
    description: str
    parameters: dict[str, Any]
    audience: str = "internal"
    scopes: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Request for LLM generation."""

    use_case: str = "chat"
    model: str | None = None
    messages: list[MessageRequest]
    tools: list[ToolDefinitionRequest] | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class UsageResponse(BaseModel):
    """Token usage in the response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str


class ToolCallResponse(BaseModel):
    """Tool call in the response."""

    tool_call_id: str
    name: str
    args: dict[str, Any]
    audience: str
    scopes: list[str]
    extensions: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    """Complete response from LLM generation."""

    content: str
    tool_calls: list[ToolCallResponse] = Field(default_factory=list)
    finish_reason: str
    usage: UsageResponse | None = None


class StreamChunk(BaseModel):
    """Streaming chunk response (NDJSON line)."""

    delta: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: UsageResponse | None = None
    error: str | None = None
