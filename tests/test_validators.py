"""Tests for ID validators and generators."""

import re

from shared.validators.id_generators import (
    generate_correlation_id,
    generate_event_id,
    generate_message_id,
    generate_session_id,
    generate_stream_id,
    generate_tool_call_id,
    generate_traceparent,
    generate_user_id_hash,
    normalize_tool_call_id,
)


class TestIDGenerators:
    """Tests for ID generation functions."""

    def test_generate_event_id(self):
        """Test event ID generation."""
        event_id = generate_event_id()
        assert event_id.startswith("evt_")
        assert len(event_id) > 4

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        assert corr_id.startswith("corr_")
        assert len(corr_id) > 5

    def test_generate_session_id(self):
        """Test session ID generation."""
        session_id = generate_session_id()
        assert session_id.startswith("s_")
        assert len(session_id) > 2

    def test_generate_message_id(self):
        """Test message ID generation."""
        msg_id = generate_message_id()
        assert msg_id.startswith("msg_")
        assert len(msg_id) > 4

    def test_generate_tool_call_id(self):
        """Test tool call ID generation."""
        tc_id = generate_tool_call_id()
        assert tc_id.startswith("tc_")
        assert len(tc_id) > 3

    def test_generate_stream_id(self):
        """Test stream ID generation."""
        stream_id = generate_stream_id()
        assert stream_id.startswith("str_")
        assert len(stream_id) > 4

    def test_ids_are_unique(self):
        """Test that generated IDs are unique."""
        ids = [generate_event_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestUserIdHash:
    """Tests for user ID hashing."""

    def test_generate_user_id_hash(self):
        """Test user ID hash generation."""
        hash1 = generate_user_id_hash("user@example.com")
        assert hash1.startswith("uh_")
        assert len(hash1) == 19  # "uh_" + 16 hex chars

    def test_user_id_hash_deterministic(self):
        """Test that same input produces same hash."""
        hash1 = generate_user_id_hash("test_user")
        hash2 = generate_user_id_hash("test_user")
        assert hash1 == hash2

    def test_different_users_different_hashes(self):
        """Test that different users have different hashes."""
        hash1 = generate_user_id_hash("user1")
        hash2 = generate_user_id_hash("user2")
        assert hash1 != hash2


class TestTraceparent:
    """Tests for W3C traceparent generation."""

    def test_generate_traceparent(self):
        """Test traceparent generation."""
        tp = generate_traceparent()
        # Format: 00-{32 hex}-{16 hex}-{2 hex}
        pattern = r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$"
        assert re.match(pattern, tp)

    def test_traceparent_with_custom_values(self):
        """Test traceparent with custom trace and parent IDs."""
        tp = generate_traceparent(
            trace_id="0" * 32,
            parent_id="1" * 16,
            sampled=True,
        )
        assert tp == "00-" + "0" * 32 + "-" + "1" * 16 + "-01"

    def test_traceparent_not_sampled(self):
        """Test traceparent with sampled=False."""
        tp = generate_traceparent(sampled=False)
        assert tp.endswith("-00")


class TestNormalizeToolCallId:
    """Tests for tool call ID normalization."""

    def test_normalize_tc_prefix(self):
        """Test that tc_ prefix is preserved and original_id is None."""
        normalized, original = normalize_tool_call_id("tc_abc123")
        assert normalized == "tc_abc123"
        assert original is None  # Already valid, no transformation

    def test_normalize_call_prefix(self):
        """Test that call_ prefix is preserved and original_id is None."""
        normalized, original = normalize_tool_call_id("call_xyz789")
        assert normalized == "call_xyz789"
        assert original is None

    def test_normalize_toolu_prefix(self):
        """Test that toolu_ prefix is preserved and original_id is None."""
        normalized, original = normalize_tool_call_id("toolu_def456")
        assert normalized == "toolu_def456"
        assert original is None

    def test_normalize_tool_prefix(self):
        """Test that tool_ prefix is preserved and original_id is None."""
        normalized, original = normalize_tool_call_id("tool_abc123")
        assert normalized == "tool_abc123"
        assert original is None

    def test_normalize_kimi_format(self):
        """Test that Kimi-style IDs (with colons) are sanitized and original preserved."""
        normalized, original = normalize_tool_call_id("get_weather:0")
        assert normalized == "tc_get_weather_0"
        assert original == "get_weather:0"  # Original preserved for debugging

    def test_normalize_adds_tc_prefix(self):
        """Test that unprefixed IDs get tc_ prefix and original is preserved."""
        normalized, original = normalize_tool_call_id("abc123")
        assert normalized == "tc_abc123"
        assert original == "abc123"  # Original preserved for debugging

    def test_normalize_none_generates_new(self):
        """Test that None generates a new ID with None original."""
        normalized, original = normalize_tool_call_id(None)
        assert normalized.startswith("tc_")
        assert len(normalized) > 3
        assert original is None

    def test_normalize_empty_generates_new(self):
        """Test that empty string generates a new ID with None original."""
        normalized, original = normalize_tool_call_id("")
        assert normalized.startswith("tc_")
        assert original is None

    def test_normalize_sanitizes_special_chars(self):
        """Test that special characters are sanitized and original preserved."""
        normalized, original = normalize_tool_call_id("abc!@#$%123")
        assert normalized.startswith("tc_")
        # Only alphanumeric, underscore, and hyphen should remain
        assert re.match(r"^tc_[A-Za-z0-9_-]+$", normalized)
        assert original == "abc!@#$%123"  # Original preserved for debugging

    def test_normalize_uuid_like_id(self):
        """Test that UUID-like IDs from Gemini/Vertex are prefixed with original preserved."""
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        normalized, original = normalize_tool_call_id(uuid_id)
        assert normalized == f"tc_{uuid_id}"
        assert original == uuid_id

    def test_normalize_only_special_chars(self):
        """Test that IDs with only special chars get new ID with original preserved."""
        normalized, original = normalize_tool_call_id("!@#$%")
        assert normalized.startswith("tc_")
        # The sanitized string would be empty after stripping underscores
        assert len(normalized) > 3  # Should be a new generated ID
        assert original == "!@#$%"
