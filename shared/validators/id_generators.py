"""ID generators for AgentOne protocol types.

All IDs follow the patterns defined in doc/contracts/schemas/v1/common.schema.json:
- event_id: evt_[A-Za-z0-9_-]+
- correlation_id: corr_[A-Za-z0-9_-]+
- session_id: s_[A-Za-z0-9_-]+
- message_id: msg_[A-Za-z0-9_-]+
- tool_call_id: (tc_|call_|toolu_|tool_)[A-Za-z0-9_-]+
- stream_id: str_[A-Za-z0-9_-]+
- user_id_hash: uh_[A-Za-z0-9_-]+
"""

import hashlib
import re

from ulid import ULID


def _generate_ulid() -> str:
    """Generate a ULID string."""
    return str(ULID())


def generate_event_id() -> str:
    """Generate a unique event ID (evt_<ulid>)."""
    return f"evt_{_generate_ulid()}"


def generate_correlation_id() -> str:
    """Generate a unique correlation ID (corr_<ulid>)."""
    return f"corr_{_generate_ulid()}"


def generate_session_id() -> str:
    """Generate a unique session ID (s_<ulid>)."""
    return f"s_{_generate_ulid()}"


def generate_message_id() -> str:
    """Generate a unique message ID (msg_<ulid>)."""
    return f"msg_{_generate_ulid()}"


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID (tc_<ulid>)."""
    return f"tc_{_generate_ulid()}"


_TOOL_CALL_ID_PATTERN = re.compile(r"^(tc_|call_|toolu_|tool_)[A-Za-z0-9_-]+$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_tool_call_id(raw_id: str | None) -> tuple[str, str | None]:
    """Normalize a provider tool call ID into an AgentOne ToolCallId.

    Our ToolCallId schema requires one of these prefixes: `tc_`, `call_`, `toolu_`, `tool_`.
    Some providers (notably Gemini/Vertex) can emit UUID-like IDs without a prefix.
    Other providers (like Kimi) may use formats like `get_weather:0`.
    In those cases, we sanitize and prefix the value with `tc_`.

    Returns:
        A tuple of (normalized_id, original_id) where original_id is the raw provider ID
        for debugging purposes. original_id is None if raw_id was None/empty or already
        matched the normalized format.
    """
    if raw_id is None:
        return (generate_tool_call_id(), None)

    raw = str(raw_id).strip()
    if not raw:
        return (generate_tool_call_id(), None)

    # Already has valid prefix - no transformation needed
    if _TOOL_CALL_ID_PATTERN.match(raw):
        return (raw, None)

    # Safe chars only - just add prefix
    if _SAFE_ID_PATTERN.match(raw):
        return (f"tc_{raw}", raw)

    # Sanitize invalid chars (like colons in Kimi's format `get_weather:0`)
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", raw).strip("_")
    if not sanitized:
        return (generate_tool_call_id(), raw)
    return (f"tc_{sanitized}", raw)


def generate_stream_id() -> str:
    """Generate a unique stream ID (str_<ulid>)."""
    return f"str_{_generate_ulid()}"


def generate_user_id_hash(user_id: str) -> str:
    """Generate a hashed user ID (uh_<hash>).

    Args:
        user_id: The original user ID to hash.

    Returns:
        A hashed user ID in the format uh_<first 16 chars of sha256>.
    """
    hash_bytes = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return f"uh_{hash_bytes}"


def generate_traceparent(
    trace_id: str | None = None,
    parent_id: str | None = None,
    sampled: bool = True,
) -> str:
    """Generate a W3C traceparent header.

    Format: version-trace_id-parent_id-trace_flags
    Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

    Args:
        trace_id: 32 hex character trace ID. Generated if not provided.
        parent_id: 16 hex character parent span ID. Generated if not provided.
        sampled: Whether the trace is sampled (default True).

    Returns:
        A W3C traceparent string.
    """
    import secrets

    version = "00"
    trace_id = trace_id or secrets.token_hex(16)
    parent_id = parent_id or secrets.token_hex(8)
    trace_flags = "01" if sampled else "00"

    return f"{version}-{trace_id}-{parent_id}-{trace_flags}"
