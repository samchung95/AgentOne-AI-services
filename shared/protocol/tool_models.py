"""Tool call and result models matching doc/contracts/schemas/v1/tool_*.schema.json."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.protocol.common import (
    SCHEMA_VERSION,
    CanvasUpdate,
    CorrelationId,
    ErrorInfo,
    ToolCallId,
    Traceparent,
)


class ToolCall(BaseModel):
    """Tool call request matching tool_call.schema.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    tool_call_id: ToolCallId
    name: Annotated[str, Field(min_length=1)]
    args: dict[str, Any]
    audience: Annotated[str, Field(min_length=1)]
    scopes: list[Annotated[str, Field(min_length=1)]]
    timeout_ms: Annotated[int, Field(ge=1)] | None = None
    endpoint: Annotated[str, Field(min_length=1)] | None = None
    idempotency_key: Annotated[str, Field(min_length=1)] | None = None
    correlation_id: CorrelationId | None = None
    traceparent: Traceparent | None = None
    extensions: dict[str, Any] | None = None
    provider_id: str | None = None  # Original tool call ID from the provider (for debugging)


class ToolStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    ERROR = "error"


class ToolResult(BaseModel):
    """Tool execution result matching tool_result.schema.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    tool_call_id: ToolCallId
    name: Annotated[str, Field(min_length=1)]
    status: ToolStatus
    data: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: ErrorInfo | None = None
    duration_ms: Annotated[int, Field(ge=0)] | None = None
    canvas_update: CanvasUpdate | None = None
    correlation_id: CorrelationId | None = None
    traceparent: Traceparent | None = None
    extensions: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_error_required_on_failure(self) -> "ToolResult":
        """Ensure error is present when status is error."""
        if self.status == ToolStatus.ERROR and self.error is None:
            raise ValueError("error is required when status is 'error'")
        return self


class DelegationStatus(str, Enum):
    """Status of a sub-agent delegation."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class DelegationResult(BaseModel):
    """Result from a sub-agent delegation.

    Used when an orchestrator agent delegates a task to a specialized sub-agent.
    The sub-agent runs in non-streaming mode and returns a complete response.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: Annotated[str, Field(min_length=1, description="ID of the sub-agent that executed")]
    task: Annotated[str, Field(min_length=1, description="The task that was delegated")]
    status: DelegationStatus
    response: str = ""  # Sub-agent's complete response (empty on error)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: Annotated[int, Field(ge=0)] = 0
    error: ErrorInfo | None = None
    correlation_id: CorrelationId | None = None
    traceparent: Traceparent | None = None
    canvas_updates: list[CanvasUpdate] = Field(
        default_factory=list, description="Canvas updates from sub-agent tool calls"
    )

    @model_validator(mode="after")
    def validate_error_required_on_failure(self) -> "DelegationResult":
        """Ensure error is present when status is error."""
        if self.status in (DelegationStatus.ERROR, DelegationStatus.TIMEOUT) and self.error is None:
            raise ValueError("error is required when status is 'error' or 'timeout'")
        return self
