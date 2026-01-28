"""Shared protocol definitions."""

from shared.protocol.common import (
    SCHEMA_VERSION,
    CanvasData,
    CanvasSource,
    CanvasUpdate,
    CorrelationId,
    ErrorCode,
    ErrorInfo,
    ErrorStage,
    EventId,
    MessageId,
    RefreshReason,
    SessionId,
    StopReason,
    StreamId,
    ToolCallId,
    Traceparent,
    Usage,
    UserId,
    UserIdHash,
)
from shared.protocol.tool_models import (
    DelegationResult,
    DelegationStatus,
    ToolCall,
    ToolResult,
    ToolStatus,
)

__all__ = [
    # Common types
    "SCHEMA_VERSION",
    "EventId",
    "CorrelationId",
    "SessionId",
    "MessageId",
    "ToolCallId",
    "StreamId",
    "UserId",
    "UserIdHash",
    "Traceparent",
    "CanvasSource",
    "CanvasData",
    "CanvasUpdate",
    "Usage",
    "ErrorCode",
    "ErrorInfo",
    "ErrorStage",
    "StopReason",
    "RefreshReason",
    # Tool models
    "ToolCall",
    "ToolStatus",
    "ToolResult",
    "DelegationStatus",
    "DelegationResult",
]
