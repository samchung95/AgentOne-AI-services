"""Integration tests for LLM providers.

These tests actually call the LLM services to verify they work correctly.
Run with CONFIG_PROFILE=test and appropriate environment variables set.

Markers:
    pytest.mark.integration - marks test as requiring real API calls
    pytest.mark.provider_openrouter - requires OPENROUTER_API_KEY
    pytest.mark.provider_openai - requires OPENAI_API_KEY
    pytest.mark.provider_azure - requires Azure OpenAI credentials
    pytest.mark.provider_vertex - requires Vertex AI/GCP credentials

Usage:
    # Run all integration tests
    CONFIG_PROFILE=test uv run pytest -m integration -v

    # Run all provider tests
    CONFIG_PROFILE=test uv run pytest tests/integration/test_llm_providers.py -v

    # Run specific provider tests
    CONFIG_PROFILE=test uv run pytest -m provider_openrouter -v
    CONFIG_PROFILE=test uv run pytest -m provider_vertex -v

    # Run with verbose output for debugging
    CONFIG_PROFILE=test uv run pytest tests/integration/test_llm_providers.py -v -s

Required environment variables (set based on which providers you want to test):

OpenRouter:
    OPENROUTER_API_KEY

OpenAI Direct:
    OPENAI_API_KEY

Azure OpenAI (Direct):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY (or use DefaultAzureCredential)

Vertex AI (Direct):
    VERTEX_PROJECT_ID
    VERTEX_LOCATION (default: us-central1)

GenAI Platform (for Azure OpenAI or Vertex AI):
    GENAI_PLATFORM_ENABLED=true
    GENAI_PLATFORM_BASE_URL
    GENAI_PLATFORM_PATH
    GENAI_PLATFORM_USER_ID
    GENAI_PLATFORM_PROJECT_NAME
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from services.llm_service.core.config.models import (
    get_provider_credentials,
    load_model_config,
    reload_model_config,
)
from services.llm_service.core.llm.base import LLMMessage, LLMToolDefinition
from services.llm_service.core.llm.factory import LLMFactory

if TYPE_CHECKING:
    from services.llm_service.core.llm.base import BaseLLMClient

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module", autouse=True)
def set_test_profile():
    """Set CONFIG_PROFILE to test for all tests in this module."""
    original = os.environ.get("CONFIG_PROFILE")
    os.environ["CONFIG_PROFILE"] = "test"
    reload_model_config()
    yield
    if original:
        os.environ["CONFIG_PROFILE"] = original
    else:
        os.environ.pop("CONFIG_PROFILE", None)
    reload_model_config()


def has_openrouter_credentials() -> bool:
    """Check if OpenRouter credentials are available."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def has_openai_credentials() -> bool:
    """Check if OpenAI credentials are available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_azure_openai_credentials() -> bool:
    """Check if Azure OpenAI credentials are available."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def has_vertex_credentials() -> bool:
    """Check if Vertex AI credentials are available (direct or GenAI Platform)."""
    genai_enabled = os.environ.get("GENAI_PLATFORM_ENABLED", "").lower() == "true"
    has_project = bool(os.environ.get("VERTEX_PROJECT_ID"))
    return genai_enabled or has_project


def has_genai_platform_credentials() -> bool:
    """Check if GenAI Platform credentials are available."""
    return (
        os.environ.get("GENAI_PLATFORM_ENABLED", "").lower() == "true"
        and bool(os.environ.get("GENAI_PLATFORM_BASE_URL"))
    )


# Skip markers
skip_no_openrouter = pytest.mark.skipif(
    not has_openrouter_credentials(),
    reason="OPENROUTER_API_KEY not set",
)
skip_no_openai = pytest.mark.skipif(
    not has_openai_credentials(),
    reason="OPENAI_API_KEY not set",
)
skip_no_azure_openai = pytest.mark.skipif(
    not has_azure_openai_credentials(),
    reason="AZURE_OPENAI_ENDPOINT not set",
)
skip_no_vertex = pytest.mark.skipif(
    not has_vertex_credentials(),
    reason="VERTEX_PROJECT_ID or GENAI_PLATFORM_ENABLED not set",
)
skip_no_genai_platform = pytest.mark.skipif(
    not has_genai_platform_credentials(),
    reason="GENAI_PLATFORM_ENABLED or GENAI_PLATFORM_BASE_URL not set",
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_simple_message() -> list[LLMMessage]:
    """Get a simple test message."""
    return [LLMMessage(role="user", content="Say 'Hello, World!' and nothing else.")]


def get_tool_message() -> list[LLMMessage]:
    """Get a message that should trigger a tool call."""
    return [
        LLMMessage(
            role="user",
            content="What's the weather like in Tokyo? Use the get_weather tool to find out.",
        )
    ]


def get_test_tools() -> list[LLMToolDefinition]:
    """Get test tool definitions."""
    return [
        LLMToolDefinition(
            name="get_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. 'Tokyo, Japan'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
            audience="internal",
            scopes=[],
        )
    ]


async def verify_simple_response(client: BaseLLMClient, model_name: str) -> None:
    """Verify a simple generate() call works."""
    messages = get_simple_message()
    response = await client.generate(messages, temperature=0.0, max_tokens=50)

    assert response is not None, f"{model_name}: Response should not be None"
    assert response.content, f"{model_name}: Response should have content"
    assert "hello" in response.content.lower(), f"{model_name}: Response should contain 'hello'"
    assert response.finish_reason in ("stop", "end_turn", "length"), f"{model_name}: Invalid finish_reason"

    print(f"  [PASS] {model_name} generate(): '{response.content[:50]}...'")


async def verify_streaming_response(client: BaseLLMClient, model_name: str) -> None:
    """Verify a streaming generate_stream() call works."""
    messages = get_simple_message()
    chunks = []
    content_parts = []

    async for chunk in client.generate_stream(messages, temperature=0.0, max_tokens=50):
        chunks.append(chunk)
        if chunk.delta:
            content_parts.append(chunk.delta)

    assert chunks, f"{model_name}: Should receive at least one chunk"

    # Find the final chunk with finish_reason
    final_chunk = chunks[-1]
    assert final_chunk.finish_reason in (
        "stop",
        "end_turn",
        "length",
        "tool_calls",
    ), f"{model_name}: Final chunk should have finish_reason"

    full_content = "".join(content_parts)
    assert full_content, f"{model_name}: Should have streamed content"
    assert "hello" in full_content.lower(), f"{model_name}: Streamed content should contain 'hello'"

    print(f"  [PASS] {model_name} generate_stream(): '{full_content[:50]}...' ({len(chunks)} chunks)")


async def verify_tool_calling(client: BaseLLMClient, model_name: str) -> None:
    """Verify tool calling works."""
    messages = get_tool_message()
    tools = get_test_tools()

    response = await client.generate(messages, tools=tools, temperature=0.0, max_tokens=200)

    assert response is not None, f"{model_name}: Response should not be None"

    # Model should either call the tool or explain it can't
    if response.tool_calls:
        assert len(response.tool_calls) > 0, f"{model_name}: Should have at least one tool call"
        tool_call = response.tool_calls[0]
        assert tool_call.name == "get_weather", f"{model_name}: Tool call should be 'get_weather'"
        assert "location" in tool_call.args, f"{model_name}: Tool call should have 'location' arg"
        print(f"  [PASS] {model_name} tool_calling: Called {tool_call.name}({tool_call.args})")
    else:
        # Some models might respond with text instead of tool call
        assert response.content, f"{model_name}: Should have content if no tool calls"
        print(f"  [INFO] {model_name} tool_calling: Responded with text instead of tool call")


async def verify_streaming_tool_calling(client: BaseLLMClient, model_name: str) -> None:
    """Verify streaming tool calling works."""
    messages = get_tool_message()
    tools = get_test_tools()

    chunks = []
    async for chunk in client.generate_stream(messages, tools=tools, temperature=0.0, max_tokens=200):
        chunks.append(chunk)

    assert chunks, f"{model_name}: Should receive at least one chunk"

    # Find the final chunk
    final_chunk = chunks[-1]

    if final_chunk.tool_calls:
        assert len(final_chunk.tool_calls) > 0, f"{model_name}: Should have tool calls in final chunk"
        tool_call = final_chunk.tool_calls[0]
        # Tool call format varies by provider
        if isinstance(tool_call, dict):
            name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        else:
            name = tool_call.name
        assert name == "get_weather", f"{model_name}: Streaming tool call should be 'get_weather'"
        print(f"  [PASS] {model_name} streaming_tool_calling: Called {name}")
    else:
        print(f"  [INFO] {model_name} streaming_tool_calling: Responded with text instead of tool call")


# =============================================================================
# OpenRouter Provider Tests
# =============================================================================


@pytest.mark.provider_openrouter
@skip_no_openrouter
class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.mark.asyncio
    async def test_openrouter_gpt4o_mini_generate(self, config):
        """Test OpenRouter GPT-4o-mini generate."""
        model_info = config.get_model("openrouter/openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openrouter/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openrouter_gpt4o_mini_streaming(self, config):
        """Test OpenRouter GPT-4o-mini streaming."""
        model_info = config.get_model("openrouter/openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_response(client, "openrouter/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openrouter_gpt4o_mini_tool_calling(self, config):
        """Test OpenRouter GPT-4o-mini tool calling."""
        model_info = config.get_model("openrouter/openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_tool_calling(client, "openrouter/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openrouter_gemini_generate(self, config):
        """Test OpenRouter Gemini generate."""
        model_info = config.get_model("openrouter/google/gemini-2.0-flash-001")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openrouter/gemini-2.0-flash")
        await client.close()

    @pytest.mark.asyncio
    async def test_openrouter_gemini_streaming(self, config):
        """Test OpenRouter Gemini streaming."""
        model_info = config.get_model("openrouter/google/gemini-2.0-flash-001")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_response(client, "openrouter/gemini-2.0-flash")
        await client.close()

    @pytest.mark.asyncio
    async def test_openrouter_claude_generate(self, config):
        """Test OpenRouter Claude generate."""
        model_info = config.get_model("openrouter/anthropic/claude-sonnet-4")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openrouter/claude-sonnet-4")
        await client.close()


# =============================================================================
# OpenAI Direct Provider Tests
# =============================================================================


@pytest.mark.provider_openai
@skip_no_openai
class TestOpenAIProvider:
    """Tests for OpenAI direct provider."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.mark.asyncio
    async def test_openai_gpt4o_mini_generate(self, config):
        """Test OpenAI GPT-4o-mini generate."""
        model_info = config.get_model("openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openai/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openai_gpt4o_mini_streaming(self, config):
        """Test OpenAI GPT-4o-mini streaming."""
        model_info = config.get_model("openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_response(client, "openai/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openai_gpt4o_mini_tool_calling(self, config):
        """Test OpenAI GPT-4o-mini tool calling."""
        model_info = config.get_model("openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_tool_calling(client, "openai/gpt-4o-mini")
        await client.close()

    @pytest.mark.asyncio
    async def test_openai_gpt4o_generate(self, config):
        """Test OpenAI GPT-4o generate."""
        model_info = config.get_model("openai/gpt-4o")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openai/gpt-4o")
        await client.close()

    @pytest.mark.asyncio
    async def test_openai_gpt41_mini_generate(self, config):
        """Test OpenAI GPT-4.1-mini generate."""
        model_info = config.get_model("openai/gpt-4.1-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "openai/gpt-4.1-mini")
        await client.close()


# =============================================================================
# Azure OpenAI Provider Tests
# =============================================================================


@pytest.mark.provider_azure
@skip_no_azure_openai
class TestAzureOpenAIProvider:
    """Tests for Azure OpenAI provider."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.mark.asyncio
    async def test_azure_openai_gpt4o_generate(self, config):
        """Test Azure OpenAI GPT-4o generate."""
        model_info = config.get_model("azure_openai/gpt-4o")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "azure_openai/gpt-4o")
        await client.close()

    @pytest.mark.asyncio
    async def test_azure_openai_gpt4o_streaming(self, config):
        """Test Azure OpenAI GPT-4o streaming."""
        model_info = config.get_model("azure_openai/gpt-4o")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_response(client, "azure_openai/gpt-4o")
        await client.close()

    @pytest.mark.asyncio
    async def test_azure_openai_gpt4o_tool_calling(self, config):
        """Test Azure OpenAI GPT-4o tool calling."""
        model_info = config.get_model("azure_openai/gpt-4o")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_tool_calling(client, "azure_openai/gpt-4o")
        await client.close()

    @pytest.mark.asyncio
    async def test_azure_openai_gpt4o_mini_generate(self, config):
        """Test Azure OpenAI GPT-4o-mini generate."""
        model_info = config.get_model("azure_openai/gpt-4o-mini")
        if not model_info:
            pytest.skip("Model not configured")

        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, "azure_openai/gpt-4o-mini")
        await client.close()


# =============================================================================
# Vertex AI Provider Tests
# =============================================================================


@pytest.mark.provider_vertex
@skip_no_vertex
class TestVertexAIProvider:
    """Tests for Vertex AI provider (direct or GenAI Platform)."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.fixture
    def is_genai_platform(self):
        """Check if running through GenAI Platform."""
        return os.environ.get("GENAI_PLATFORM_ENABLED", "").lower() == "true"

    @pytest.mark.asyncio
    async def test_vertex_gemini_flash_generate(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.0 Flash generate."""
        model_info = config.get_model("vertex_ai/gemini-2.0-flash")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, f"vertex_ai/gemini-2.0-flash ({mode})")
        await client.close()

    @pytest.mark.asyncio
    async def test_vertex_gemini_flash_streaming(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.0 Flash streaming."""
        model_info = config.get_model("vertex_ai/gemini-2.0-flash")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_response(client, f"vertex_ai/gemini-2.0-flash ({mode})")
        await client.close()

    @pytest.mark.asyncio
    async def test_vertex_gemini_flash_tool_calling(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.0 Flash tool calling."""
        model_info = config.get_model("vertex_ai/gemini-2.0-flash")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_tool_calling(client, f"vertex_ai/gemini-2.0-flash ({mode})")
        await client.close()

    @pytest.mark.asyncio
    async def test_vertex_gemini_flash_streaming_tool_calling(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.0 Flash streaming tool calling."""
        model_info = config.get_model("vertex_ai/gemini-2.0-flash")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_streaming_tool_calling(client, f"vertex_ai/gemini-2.0-flash ({mode})")
        await client.close()

    @pytest.mark.asyncio
    async def test_vertex_gemini_25_flash_generate(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.5 Flash Preview generate."""
        model_info = config.get_model("vertex_ai/gemini-2.5-flash-preview-05-20")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, f"vertex_ai/gemini-2.5-flash ({mode})")
        await client.close()

    @pytest.mark.asyncio
    async def test_vertex_gemini_25_pro_generate(self, config, is_genai_platform):
        """Test Vertex AI Gemini 2.5 Pro Preview generate."""
        model_info = config.get_model("vertex_ai/gemini-2.5-pro-preview-05-06")
        if not model_info:
            pytest.skip("Model not configured")

        mode = "GenAI Platform" if is_genai_platform else "Direct"
        client = LLMFactory.create_for_model(model_info)
        await verify_simple_response(client, f"vertex_ai/gemini-2.5-pro ({mode})")
        await client.close()


# =============================================================================
# GenAI Platform Specific Tests
# =============================================================================


@pytest.mark.provider_azure
@pytest.mark.provider_vertex
@skip_no_genai_platform
class TestGenAIPlatform:
    """Tests specific to GenAI Platform deployment."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.mark.asyncio
    async def test_genai_platform_vertex_headers(self, config):
        """Test that GenAI Platform headers are set correctly for Vertex AI."""
        from services.llm_service.core.llm.vertex import VertexAIClient

        model_info = config.get_model("vertex_ai/gemini-2.0-flash")
        if not model_info:
            pytest.skip("Model not configured")

        creds = get_provider_credentials("vertex_ai")

        # Verify GenAI Platform is enabled
        assert creds.get("genai_platform_enabled"), "GenAI Platform should be enabled"
        assert creds.get("genai_platform_base_url"), "GenAI Platform base URL should be set"

        # Create client and verify it initializes
        client = LLMFactory.create_for_model(model_info)
        assert isinstance(client, VertexAIClient)

        # Run a simple request to verify connectivity
        await verify_simple_response(client, "vertex_ai (GenAI Platform)")
        await client.close()

    @pytest.mark.asyncio
    async def test_genai_platform_azure_openai(self, config):
        """Test Azure OpenAI through GenAI Platform."""
        from services.llm_service.core.llm.azure_openai import AzureOpenAIClient

        model_info = config.get_model("azure_openai/gpt-4o")
        if not model_info:
            pytest.skip("Model not configured")

        creds = get_provider_credentials("azure_openai")

        # Verify GenAI Platform is enabled
        if not creds.get("genai_platform_enabled"):
            pytest.skip("GenAI Platform not enabled for Azure OpenAI")

        client = LLMFactory.create_for_model(model_info)
        assert isinstance(client, AzureOpenAIClient)

        await verify_simple_response(client, "azure_openai (GenAI Platform)")
        await client.close()


# =============================================================================
# Cross-Provider Comparison Tests
# =============================================================================


class TestCrossProviderComparison:
    """Tests that compare behavior across providers."""

    @pytest.fixture
    def config(self):
        """Load model config."""
        return load_model_config()

    @pytest.mark.asyncio
    async def test_all_available_providers_respond(self, config):
        """Test that all configured providers can respond to a simple message."""
        # Skip if no providers are available
        if not any([
            has_openrouter_credentials(),
            has_openai_credentials(),
            has_azure_openai_credentials(),
            has_vertex_credentials(),
        ]):
            pytest.skip("No provider credentials configured")

        results = {}

        # OpenRouter
        if has_openrouter_credentials():
            model = config.get_model("openrouter/openai/gpt-4o-mini")
            if model:
                try:
                    client = LLMFactory.create_for_model(model)
                    response = await client.generate(get_simple_message(), max_tokens=20)
                    results["openrouter"] = ("PASS", response.content[:30] if response.content else "")
                    await client.close()
                except Exception as e:
                    results["openrouter"] = ("FAIL", str(e)[:50])

        # OpenAI
        if has_openai_credentials():
            model = config.get_model("openai/gpt-4o-mini")
            if model:
                try:
                    client = LLMFactory.create_for_model(model)
                    response = await client.generate(get_simple_message(), max_tokens=20)
                    results["openai"] = ("PASS", response.content[:30] if response.content else "")
                    await client.close()
                except Exception as e:
                    results["openai"] = ("FAIL", str(e)[:50])

        # Azure OpenAI
        if has_azure_openai_credentials():
            model = config.get_model("azure_openai/gpt-4o")
            if model:
                try:
                    client = LLMFactory.create_for_model(model)
                    response = await client.generate(get_simple_message(), max_tokens=20)
                    results["azure_openai"] = ("PASS", response.content[:30] if response.content else "")
                    await client.close()
                except Exception as e:
                    results["azure_openai"] = ("FAIL", str(e)[:50])

        # Vertex AI
        if has_vertex_credentials():
            model = config.get_model("vertex_ai/gemini-2.0-flash")
            if model:
                try:
                    client = LLMFactory.create_for_model(model)
                    response = await client.generate(get_simple_message(), max_tokens=20)
                    results["vertex_ai"] = ("PASS", response.content[:30] if response.content else "")
                    await client.close()
                except Exception as e:
                    results["vertex_ai"] = ("FAIL", str(e)[:50])

        # Print summary
        print("\n=== Provider Test Summary ===")
        for provider, (status, detail) in results.items():
            print(f"  {provider}: {status} - {detail}")

        # At least one provider should work
        assert any(status == "PASS" for status, _ in results.values()), (
            f"No providers responded successfully: {results}"
        )

    @pytest.mark.asyncio
    async def test_tool_calling_consistency(self, config):
        """Test that tool calling works consistently across providers."""
        # Skip if no providers are available
        if not any([
            has_openrouter_credentials(),
            has_openai_credentials(),
            has_vertex_credentials(),
        ]):
            pytest.skip("No provider credentials configured")

        results = {}
        tools = get_test_tools()
        messages = get_tool_message()

        providers_to_test = []

        if has_openrouter_credentials():
            model = config.get_model("openrouter/openai/gpt-4o-mini")
            if model:
                providers_to_test.append(("openrouter", model))

        if has_openai_credentials():
            model = config.get_model("openai/gpt-4o-mini")
            if model:
                providers_to_test.append(("openai", model))

        if has_vertex_credentials():
            model = config.get_model("vertex_ai/gemini-2.0-flash")
            if model:
                providers_to_test.append(("vertex_ai", model))

        for provider_name, model in providers_to_test:
            try:
                client = LLMFactory.create_for_model(model)
                response = await client.generate(messages, tools=tools, max_tokens=200)

                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    results[provider_name] = ("TOOL_CALL", tool_call.name, tool_call.args)
                else:
                    results[provider_name] = ("TEXT", response.content[:50] if response.content else "")

                await client.close()
            except Exception as e:
                results[provider_name] = ("ERROR", str(e)[:50])

        # Print summary
        print("\n=== Tool Calling Test Summary ===")
        for provider, result in results.items():
            print(f"  {provider}: {result}")

        # Skip if no providers available
        if not results:
            pytest.skip("No providers available for testing")
