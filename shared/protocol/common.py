"""Common types matching doc/contracts/schemas/v1/common.schema.json."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# Schema version constant
SCHEMA_VERSION = "1.0"

# ID pattern types using Annotated for validation
EventId = Annotated[str, Field(pattern=r"^evt_[A-Za-z0-9_-]+$")]
CorrelationId = Annotated[str, Field(pattern=r"^corr_[A-Za-z0-9_-]+$")]
SessionId = Annotated[str, Field(pattern=r"^s_[A-Za-z0-9_-]+$")]
MessageId = Annotated[str, Field(pattern=r"^msg_[A-Za-z0-9_-]+$")]
# Accept various LLM provider tool call ID formats:
# - tc_... (our internal format)
# - call_... (OpenAI/OpenRouter format)
# - toolu_... (Anthropic format)
ToolCallId = Annotated[str, Field(pattern=r"^(tc_|call_|toolu_)[A-Za-z0-9_-]+$")]
StreamId = Annotated[str, Field(pattern=r"^str_[A-Za-z0-9_-]+$")]
PackageId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
AgentId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
UserId = Annotated[str, Field(min_length=1)]
UserIdHash = Annotated[str, Field(pattern=r"^uh_[A-Za-z0-9_-]+$")]
Traceparent = Annotated[str, Field(pattern=r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")]


class CanvasSource(str, Enum):
    """Source of canvas data."""

    TOOL = "tool"
    MODEL = "model"


class CanvasData(BaseModel):
    """Canvas data with source tag and arbitrary properties."""

    model_config = ConfigDict(extra="allow")

    source: CanvasSource


class CanvasUpdate(BaseModel):
    """Canvas update with component name and data."""

    model_config = ConfigDict(extra="forbid")

    component: Annotated[str, Field(min_length=1)]
    data: CanvasData


class Usage(BaseModel):
    """Token usage metrics."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)] | None = None
    model_name: Annotated[str, Field(min_length=1)]


class ErrorCode(str, Enum):
    """Predefined error codes."""

    RBAC_DENY = "RBAC_DENY"
    MCP_ALLOWLIST_DENY = "MCP_ALLOWLIST_DENY"
    MCP_TIMEOUT = "MCP_TIMEOUT"
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_ERROR = "LLM_ERROR"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT = "RATE_LIMIT"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"


class ErrorInfo(BaseModel):
    """Error information structure."""

    model_config = ConfigDict(extra="forbid")

    code: str  # Can be ErrorCode or custom pattern ^[A-Z0-9_]+$
    message: Annotated[str, Field(min_length=1)]
    retryable: bool | None = None
    details: dict[str, Any] | None = None


class ErrorStage(str, Enum):
    """Stage where error occurred."""

    AUTH = "auth"
    LLM = "llm"
    TOOL = "tool"
    PERSISTENCE = "persistence"
    PROTOCOL = "protocol"


class StopReason(str, Enum):
    """Reason for stopping text generation."""

    END_TURN = "end_turn"
    STOP = "stop"
    TOOL_CALL = "tool_call"
    LENGTH = "length"
    ERROR = "error"


class RefreshReason(str, Enum):
    """Reason for token refresh."""

    EXPIRING = "expiring"
    MANUAL = "manual"
    REAUTH = "reauth"
