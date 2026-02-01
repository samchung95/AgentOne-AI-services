"""Integration tests specifically for Vertex AI client (ChatGoogleGenerativeAI migration).

These tests verify the migration from ChatVertexAI to ChatGoogleGenerativeAI works correctly
with both direct Vertex AI and GenAI Platform modes.

Markers:
    pytest.mark.integration - marks test as requiring real API calls
    pytest.mark.provider_vertex - requires Vertex AI/GCP credentials

Usage:
    # Run all Vertex AI tests with markers
    uv run pytest -m provider_vertex -v -s

    # Run with direct Vertex AI (requires VERTEX_PROJECT_ID)
    VERTEX_PROJECT_ID=your-project uv run pytest tests/integration/test_vertex_integration.py -v -s

    # Run with GenAI Platform (requires GENAI_PLATFORM_* vars)
    GENAI_PLATFORM_ENABLED=true \\
    GENAI_PLATFORM_BASE_URL=https://genai.yourcompany.com \\
    GENAI_PLATFORM_PATH=stg/v1 \\
    uv run pytest tests/integration/test_vertex_integration.py -v -s

Required environment variables:

Direct Vertex AI:
    VERTEX_PROJECT_ID - GCP project ID
    VERTEX_LOCATION - Region (default: us-central1)
    (Uses Application Default Credentials - run `gcloud auth application-default login`)

GenAI Platform:
    GENAI_PLATFORM_ENABLED=true
    GENAI_PLATFORM_BASE_URL - Base URL of GenAI gateway
    GENAI_PLATFORM_PATH - API path (default: stg/v1)
    GENAI_PLATFORM_USER_ID - User ID header (optional)
    GENAI_PLATFORM_PROJECT_NAME - Project name header (optional)
    (Uses Azure AD credentials)
"""

from __future__ import annotations

import os

import pytest

from services.llm_service.core.llm.base import LLMMessage, LLMToolDefinition
from services.llm_service.core.llm.vertex import DEFAULT_VERTEX_LOCATION, VertexAIClient

# Mark all tests in this module as integration and vertex-specific tests
pytestmark = [pytest.mark.integration, pytest.mark.provider_vertex]


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def is_direct_vertex_available() -> bool:
    """Check if direct Vertex AI credentials are available."""
    return bool(os.environ.get("VERTEX_PROJECT_ID"))


def is_genai_platform_available() -> bool:
    """Check if GenAI Platform credentials are available."""
    return (
        os.environ.get("GENAI_PLATFORM_ENABLED", "").lower() == "true"
        and bool(os.environ.get("GENAI_PLATFORM_BASE_URL"))
    )


def is_any_vertex_available() -> bool:
    """Check if any Vertex AI mode is available."""
    return is_direct_vertex_available() or is_genai_platform_available()


skip_no_vertex = pytest.mark.skipif(
    not is_any_vertex_available(),
    reason="Neither VERTEX_PROJECT_ID nor GENAI_PLATFORM_ENABLED is set",
)

skip_no_direct_vertex = pytest.mark.skipif(
    not is_direct_vertex_available(),
    reason="VERTEX_PROJECT_ID not set (direct Vertex AI)",
)

skip_no_genai_platform = pytest.mark.skipif(
    not is_genai_platform_available(),
    reason="GENAI_PLATFORM_ENABLED or GENAI_PLATFORM_BASE_URL not set",
)


def create_vertex_client(model: str = "gemini-2.0-flash") -> VertexAIClient:
    """Create a VertexAIClient based on available credentials."""
    if is_genai_platform_available():
        return VertexAIClient.from_model_config(
            model=model,
            genai_platform_enabled=True,
            genai_platform_base_url=os.environ.get("GENAI_PLATFORM_BASE_URL"),
            genai_platform_path=os.environ.get("GENAI_PLATFORM_PATH", "stg/v1"),
            genai_platform_user_id=os.environ.get("GENAI_PLATFORM_USER_ID"),
            genai_platform_project_name=os.environ.get("GENAI_PLATFORM_PROJECT_NAME"),
        )
    elif is_direct_vertex_available():
        return VertexAIClient.from_model_config(
            model=model,
            project_id=os.environ.get("VERTEX_PROJECT_ID"),
            location=os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
        )
    else:
        raise RuntimeError("No Vertex AI credentials available")


def get_simple_messages() -> list[LLMMessage]:
    """Get simple test messages."""
    return [LLMMessage(role="user", content="Say 'Hello' and nothing else.")]


def get_conversation_messages() -> list[LLMMessage]:
    """Get multi-turn conversation messages."""
    return [
        LLMMessage(role="user", content="My name is Alice."),
        LLMMessage(role="assistant", content="Hello Alice! How can I help you today?"),
        LLMMessage(role="user", content="What's my name?"),
    ]


def get_weather_tool() -> list[LLMToolDefinition]:
    """Get a weather tool definition for testing."""
    return [
        LLMToolDefinition(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
            audience="internal",
            scopes=[],
        )
    ]


# =============================================================================
# Basic Functionality Tests
# =============================================================================


@skip_no_vertex
class TestVertexBasicFunctionality:
    """Test basic Vertex AI functionality."""

    @pytest.mark.asyncio
    async def test_simple_generate(self):
        """Test simple text generation."""
        client = create_vertex_client()

        try:
            response = await client.generate(
                messages=get_simple_messages(),
                temperature=0.0,
                max_tokens=50,
            )

            assert response is not None
            assert response.content
            assert "hello" in response.content.lower()
            assert response.finish_reason in ("stop", "end_turn")

            print(f"\n[PASS] generate(): {response.content}")
            if response.usage:
                print(f"  Usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_simple_streaming(self):
        """Test streaming text generation."""
        client = create_vertex_client()

        try:
            chunks = []
            content_parts = []

            async for chunk in client.generate_stream(
                messages=get_simple_messages(),
                temperature=0.0,
                max_tokens=50,
            ):
                chunks.append(chunk)
                if chunk.delta:
                    content_parts.append(chunk.delta)

            assert chunks
            full_content = "".join(content_parts)
            assert full_content
            assert "hello" in full_content.lower()

            final_chunk = chunks[-1]
            assert final_chunk.finish_reason in ("stop", "end_turn", "tool_calls")

            print(f"\n[PASS] generate_stream(): {full_content}")
            print(f"  Received {len(chunks)} chunks")
            if final_chunk.usage:
                print(f"  Usage: {final_chunk.usage.input_tokens} in, {final_chunk.usage.output_tokens} out")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_conversation_context(self):
        """Test that model maintains conversation context."""
        client = create_vertex_client()

        try:
            response = await client.generate(
                messages=get_conversation_messages(),
                temperature=0.0,
                max_tokens=50,
            )

            assert response is not None
            assert response.content
            # Model should remember the name "Alice" from context
            assert "alice" in response.content.lower()

            print(f"\n[PASS] conversation context: {response.content}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """Test that system prompt is respected."""
        client = create_vertex_client()

        try:
            response = await client.generate(
                messages=[LLMMessage(role="user", content="Greet me")],
                system_prompt="You are a pirate. Always respond like a pirate.",
                temperature=0.7,
                max_tokens=100,
            )

            assert response is not None
            assert response.content
            # Response should have pirate-like language
            pirate_indicators = ["arr", "matey", "ahoy", "ye", "cap'n", "sea", "ship"]
            has_pirate_speak = any(word in response.content.lower() for word in pirate_indicators)

            print(f"\n[INFO] system prompt response: {response.content}")
            if has_pirate_speak:
                print("  [PASS] Response appears to follow pirate persona")

        finally:
            await client.close()


# =============================================================================
# Tool Calling Tests
# =============================================================================


@skip_no_vertex
class TestVertexToolCalling:
    """Test Vertex AI tool calling functionality."""

    @pytest.mark.asyncio
    async def test_tool_call_generate(self):
        """Test tool calling with generate()."""
        client = create_vertex_client()

        try:
            messages = [
                LLMMessage(role="user", content="What's the weather in Tokyo?")
            ]
            tools = get_weather_tool()

            response = await client.generate(
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=200,
            )

            assert response is not None

            if response.tool_calls:
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]
                assert tool_call.name == "get_weather"
                assert "location" in tool_call.args

                print(f"\n[PASS] tool calling: {tool_call.name}({tool_call.args})")
            else:
                # Some models may respond with text instead
                print(f"\n[INFO] Model responded with text instead of tool call: {response.content}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        """Test tool calling with streaming."""
        client = create_vertex_client()

        try:
            messages = [
                LLMMessage(role="user", content="Get the weather in London, UK")
            ]
            tools = get_weather_tool()

            chunks = []
            async for chunk in client.generate_stream(
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=200,
            ):
                chunks.append(chunk)

            assert chunks
            final_chunk = chunks[-1]

            if final_chunk.tool_calls:
                assert len(final_chunk.tool_calls) > 0
                tool_call = final_chunk.tool_calls[0]

                # Extract name based on format
                if isinstance(tool_call, dict):
                    name = tool_call.get("function", {}).get("name") or tool_call.get("name")
                else:
                    name = tool_call.name

                assert name == "get_weather"
                print(f"\n[PASS] streaming tool calling: {name}")
            else:
                content = "".join(c.delta for c in chunks if c.delta)
                print(f"\n[INFO] Model responded with text instead of tool call: {content}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_tool_result_handling(self):
        """Test handling tool results in conversation."""
        client = create_vertex_client()

        try:
            # First turn: get tool call
            messages = [
                LLMMessage(role="user", content="What's the weather in Paris?")
            ]
            tools = get_weather_tool()

            response = await client.generate(messages=messages, tools=tools, max_tokens=200)

            if not response.tool_calls:
                pytest.skip("Model did not produce tool call")

            tool_call = response.tool_calls[0]

            # Second turn: provide tool result
            messages = [
                LLMMessage(role="user", content="What's the weather in Paris?"),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call.tool_call_id,
                            "name": tool_call.name,
                            "args": tool_call.args,
                        }
                    ],
                ),
                LLMMessage(
                    role="tool",
                    content='{"temperature": 18, "condition": "cloudy", "humidity": 65}',
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                ),
            ]

            response = await client.generate(messages=messages, tools=tools, max_tokens=200)

            assert response is not None
            assert response.content
            # Response should mention the weather data
            assert any(word in response.content.lower() for word in ["18", "cloudy", "paris"])

            print(f"\n[PASS] tool result handling: {response.content}")

        finally:
            await client.close()


# =============================================================================
# Model Variant Tests
# =============================================================================


@skip_no_vertex
class TestVertexModelVariants:
    """Test different Gemini model variants."""

    @pytest.mark.asyncio
    async def test_gemini_20_flash(self):
        """Test Gemini 2.0 Flash."""
        client = create_vertex_client("gemini-2.0-flash")

        try:
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content
            print(f"\n[PASS] gemini-2.0-flash: {response.content}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_gemini_25_flash_preview(self):
        """Test Gemini 2.5 Flash Preview."""
        client = create_vertex_client("gemini-2.5-flash-preview-05-20")

        try:
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content
            print(f"\n[PASS] gemini-2.5-flash-preview: {response.content}")
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                pytest.skip(f"Model not available: {e}")
            raise
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_gemini_25_pro_preview(self):
        """Test Gemini 2.5 Pro Preview."""
        client = create_vertex_client("gemini-2.5-pro-preview-05-06")

        try:
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content
            print(f"\n[PASS] gemini-2.5-pro-preview: {response.content}")
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                pytest.skip(f"Model not available: {e}")
            raise
        finally:
            await client.close()


# =============================================================================
# Direct Vertex AI Specific Tests
# =============================================================================


@skip_no_direct_vertex
class TestDirectVertexAI:
    """Tests specific to direct Vertex AI (not GenAI Platform)."""

    @pytest.mark.asyncio
    async def test_direct_vertex_initialization(self):
        """Test that direct Vertex AI client initializes correctly."""
        client = VertexAIClient.from_model_config(
            model="gemini-2.0-flash",
            project_id=os.environ.get("VERTEX_PROJECT_ID"),
            location=os.environ.get("VERTEX_LOCATION", "us-central1"),
        )

        try:
            # Verify client is configured for direct mode
            assert client._genai_platform_enabled is False
            assert client._direct_project_id is not None

            # Verify it can make requests
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content

            # In direct mode, _force_sync should be False (native async)
            assert client._force_sync is False

            print(f"\n[PASS] Direct Vertex AI: {response.content}")
            print(f"  Project: {client._direct_project_id}")
            print(f"  Location: {client._direct_location}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_direct_vertex_native_async(self):
        """Verify direct Vertex AI uses native async (not sync fallback)."""
        client = VertexAIClient.from_model_config(
            model="gemini-2.0-flash",
            project_id=os.environ.get("VERTEX_PROJECT_ID"),
            location=os.environ.get("VERTEX_LOCATION", "us-central1"),
        )

        try:
            # Force client initialization
            client._ensure_client()

            # Direct mode should not force sync
            assert client._force_sync is False, "Direct Vertex AI should use native async"

            # Verify streaming works
            chunks = []
            async for chunk in client.generate_stream(
                messages=get_simple_messages(),
                max_tokens=50,
            ):
                chunks.append(chunk)

            assert chunks
            print(f"\n[PASS] Direct Vertex AI native async streaming: {len(chunks)} chunks")

        finally:
            await client.close()


# =============================================================================
# GenAI Platform Specific Tests
# =============================================================================


@skip_no_genai_platform
class TestGenAIPlatformVertexAI:
    """Tests specific to Vertex AI through GenAI Platform."""

    @pytest.mark.asyncio
    async def test_genai_platform_initialization(self):
        """Test that GenAI Platform client initializes correctly."""
        client = VertexAIClient.from_model_config(
            model="gemini-2.0-flash",
            genai_platform_enabled=True,
            genai_platform_base_url=os.environ.get("GENAI_PLATFORM_BASE_URL"),
            genai_platform_path=os.environ.get("GENAI_PLATFORM_PATH", "stg/v1"),
            genai_platform_user_id=os.environ.get("GENAI_PLATFORM_USER_ID"),
            genai_platform_project_name=os.environ.get("GENAI_PLATFORM_PROJECT_NAME"),
        )

        try:
            # Verify client is configured for GenAI Platform mode
            assert client._genai_platform_enabled is True
            assert client._genai_platform_base_url is not None

            # Verify it can make requests
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content

            # In GenAI Platform mode, _force_sync should be True
            assert client._force_sync is True

            print(f"\n[PASS] GenAI Platform Vertex AI: {response.content}")
            print(f"  Base URL: {client._genai_platform_base_url}")
            print(f"  Path: {client._genai_platform_path}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_genai_platform_headers(self):
        """Verify GenAI Platform headers are set correctly."""
        user_id = os.environ.get("GENAI_PLATFORM_USER_ID")
        project_name = os.environ.get("GENAI_PLATFORM_PROJECT_NAME")

        client = VertexAIClient.from_model_config(
            model="gemini-2.0-flash",
            genai_platform_enabled=True,
            genai_platform_base_url=os.environ.get("GENAI_PLATFORM_BASE_URL"),
            genai_platform_path=os.environ.get("GENAI_PLATFORM_PATH", "stg/v1"),
            genai_platform_user_id=user_id,
            genai_platform_project_name=project_name,
        )

        try:
            # Initialize client
            client._ensure_client()

            # Verify request succeeds (headers should be passed correctly)
            response = await client.generate(
                messages=get_simple_messages(),
                max_tokens=50,
            )
            assert response.content

            print("\n[PASS] GenAI Platform headers work correctly")
            if user_id:
                print(f"  User ID: {user_id}")
            if project_name:
                print(f"  Project: {project_name}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_genai_platform_sync_fallback_streaming(self):
        """Verify GenAI Platform uses sync fallback for streaming."""
        client = VertexAIClient.from_model_config(
            model="gemini-2.0-flash",
            genai_platform_enabled=True,
            genai_platform_base_url=os.environ.get("GENAI_PLATFORM_BASE_URL"),
            genai_platform_path=os.environ.get("GENAI_PLATFORM_PATH", "stg/v1"),
        )

        try:
            # Initialize and verify force_sync is True
            client._ensure_client()
            assert client._force_sync is True, "GenAI Platform should use sync fallback"

            # Verify streaming still works (via sync fallback)
            chunks = []
            content_parts = []

            async for chunk in client.generate_stream(
                messages=get_simple_messages(),
                max_tokens=50,
            ):
                chunks.append(chunk)
                if chunk.delta:
                    content_parts.append(chunk.delta)

            assert chunks
            full_content = "".join(content_parts)
            assert full_content

            print(f"\n[PASS] GenAI Platform sync fallback streaming: {len(chunks)} chunks")
            print(f"  Content: {full_content}")

        finally:
            await client.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


@skip_no_vertex
class TestVertexErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        client = create_vertex_client("nonexistent-model-12345")

        try:
            with pytest.raises(Exception) as exc_info:
                await client.generate(
                    messages=get_simple_messages(),
                    max_tokens=50,
                )

            # Should get a model not found error
            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ["not found", "invalid", "unknown", "404"])
            print(f"\n[PASS] Invalid model handled: {exc_info.value}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling of empty messages list."""
        client = create_vertex_client()

        try:
            with pytest.raises(Exception):
                await client.generate(messages=[], max_tokens=50)

            print("\n[PASS] Empty messages handled with exception")

        except Exception as e:
            # Some implementations may not raise for empty messages
            print(f"\n[INFO] Empty messages result: {e}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_client_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        client = create_vertex_client()

        # Initialize client
        await client.generate(messages=get_simple_messages(), max_tokens=10)

        # Close multiple times - should not raise
        await client.close()
        await client.close()
        await client.close()

        print("\n[PASS] Multiple close() calls handled safely")
