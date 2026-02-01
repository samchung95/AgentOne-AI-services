"""Pytest configuration and fixtures.

Mock Providers Flag
-------------------
Use the `--mock-providers` flag to run tests with mocked LLM providers:

    pytest --mock-providers tests/

When enabled, the `mock_providers` fixture auto-applies to all tests and patches
all LLM client classes (OpenAIClient, OpenRouterClient, AzureOpenAIClient,
VertexAIClient, RemoteLLMClient) to return responses from fixture files in
tests/fixtures/:

- mock_openai_response.json: Standard chat completion response
- mock_openai_stream.json: Streaming chunks array
- mock_tool_call_response.json: Response with tool calls
- mock_error_response.json: Error response structure

This allows running the full test suite without real API keys or network access.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from services.llm_service.core.llm.base import LLMResponse

# Set test environment before importing settings
os.environ.setdefault("CONFIG_PROFILE", "openrouter")
os.environ.setdefault("LLM_SERVICE_LOAD_ENV_FILE", "false")


# =============================================================================
# Pytest CLI Option
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for pytest."""
    parser.addoption(
        "--mock-providers",
        action="store_true",
        default=False,
        help="Run tests with mocked LLM providers using fixture files",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "mock_providers: mark test to use mocked LLM providers",
    )


# =============================================================================
# Fixture File Loading
# =============================================================================


def _load_fixture(filename: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/.

    Args:
        filename: Name of the fixture file (e.g., 'mock_openai_response.json')

    Returns:
        Parsed JSON content as a dictionary.
    """
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path) as f:
        return json.load(f)


def _get_mock_completion_response() -> dict[str, Any]:
    """Get the mock completion response fixture."""
    return _load_fixture("mock_openai_response.json")


def _get_mock_stream_chunks() -> list[dict[str, Any]]:
    """Get the mock streaming chunks fixture."""
    return _load_fixture("mock_openai_stream.json")


def _get_mock_tool_call_response() -> dict[str, Any]:
    """Get the mock tool call response fixture."""
    return _load_fixture("mock_tool_call_response.json")


# =============================================================================
# Mock Response Builders
# =============================================================================


def _build_mock_llm_response(
    response_data: dict[str, Any], tools: list[Any] | None = None
) -> "LLMResponse":
    """Build a mock LLMResponse from fixture data.

    Args:
        response_data: OpenAI-format response dict from fixture file.
        tools: Optional list of LLMToolDefinition objects for audience/scopes.

    Returns:
        A properly constructed LLMResponse object.
    """
    import json

    from services.llm_service.core.llm.base import LLMResponse
    from shared.protocol.common import Usage
    from shared.protocol.tool_models import ToolCall
    from shared.validators.id_generators import normalize_tool_call_id

    choice = response_data["choices"][0]
    message = choice["message"]
    usage_data = response_data.get("usage", {})

    # Build tool lookup from provided tools for audience/scopes
    tool_lookup: dict[str, Any] = {}
    if tools:
        for tool in tools:
            tool_lookup[tool.name] = tool

    tool_calls = []
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func_name = tc["function"]["name"]
            tool_def = tool_lookup.get(func_name)

            # Parse arguments from JSON string
            args_str = tc["function"]["arguments"]
            args = json.loads(args_str) if isinstance(args_str, str) else args_str

            # Normalize the tool call ID
            normalized_id, original_id = normalize_tool_call_id(tc["id"])

            tool_calls.append(
                ToolCall(
                    tool_call_id=normalized_id,
                    name=func_name,
                    args=args,
                    audience=tool_def.audience if tool_def else "default",
                    scopes=tool_def.scopes if tool_def else ["default.read"],
                    provider_id=original_id,
                )
            )

    return LLMResponse(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason", "stop"),
        usage=Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            model_name=response_data.get("model", "mock-model"),
        ),
    )


async def _mock_generate_stream(
    chunks: list[dict[str, Any]],
) -> "AsyncGenerator[Any, None]":
    """Create an async generator that yields mock chunks."""
    from services.llm_service.core.llm.base import LLMChunk
    from shared.protocol.common import Usage

    for chunk_data in chunks:
        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})
        usage_data = chunk_data.get("usage")

        chunk = LLMChunk(
            delta=delta.get("content", ""),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                model_name=chunk_data.get("model", "mock-model"),
            )
            if usage_data
            else None,
        )
        yield chunk


def _create_mock_client_class(client_name: str) -> type:
    """Create a mock client class that returns fixture data.

    Args:
        client_name: Name of the client being mocked (for logging)

    Returns:
        A mock class that can be instantiated and used like the real client.
    """

    class MockLLMClient:
        """Mock LLM client that returns fixture data."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._client_name = client_name
            self._args = args
            self._kwargs = kwargs

        @classmethod
        def from_model_config(cls, *args: Any, **kwargs: Any) -> "MockLLMClient":
            """Mock factory method."""
            return cls(*args, **kwargs)

        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            """Return mock completion response."""
            # Check if tools are provided to return tool call response
            tools = kwargs.get("tools") or (args[1] if len(args) > 1 else None)
            if tools:
                return _build_mock_llm_response(_get_mock_tool_call_response(), tools)
            return _build_mock_llm_response(_get_mock_completion_response())

        async def generate_stream(
            self, *args: Any, **kwargs: Any
        ) -> "AsyncGenerator[Any, None]":
            """Return mock streaming response."""
            chunks = _get_mock_stream_chunks()
            async for chunk in _mock_generate_stream(chunks):
                yield chunk

        async def close(self) -> None:
            """Mock close method."""
            pass

    return MockLLMClient


# =============================================================================
# Mock Providers Fixture
# =============================================================================


@pytest.fixture(autouse=False)
def mock_providers(request: pytest.FixtureRequest) -> Any:
    """Fixture that patches all LLM client classes to return mock responses.

    This fixture is automatically applied when running pytest with
    the --mock-providers flag. It patches:

    - OpenAIClient
    - OpenRouterClient
    - AzureOpenAIClient
    - VertexAIClient
    - RemoteLLMClient

    All mocked clients return responses from the fixture files in tests/fixtures/.

    Usage:
        pytest --mock-providers tests/

    Or apply to specific tests:
        @pytest.mark.usefixtures("mock_providers")
        def test_something():
            ...
    """
    # Client module paths - these are the primary definitions that factory.py imports
    client_paths = [
        "services.llm_service.core.llm.openai_client.OpenAIClient",
        "services.llm_service.core.llm.openrouter_client.OpenRouterClient",
        "services.llm_service.core.llm.azure_openai.AzureOpenAIClient",
        "services.llm_service.core.llm.vertex.VertexAIClient",
        "services.llm_service.core.llm.remote_client.RemoteLLMClient",
    ]

    patches = []
    mock_clients = {}

    # Create mock classes for each client
    for path in client_paths:
        client_name = path.split(".")[-1]
        mock_class = _create_mock_client_class(client_name)
        p = patch(path, mock_class)
        patches.append(p)
        mock_clients[client_name] = mock_class

    # Start all patches
    for p in patches:
        p.start()

    yield mock_clients

    # Stop all patches
    for p in patches:
        p.stop()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply mock_providers fixture when --mock-providers flag is set."""
    if config.getoption("--mock-providers"):
        for item in items:
            # Add the mock_providers fixture to all tests
            item.fixturenames.append("mock_providers")


@pytest.fixture
def sample_messages():
    """Sample LLM messages for testing."""
    from services.llm_service.core.llm.base import LLMMessage

    return [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Hello!"),
    ]


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    from services.llm_service.core.llm.base import LLMToolDefinition

    return [
        LLMToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
            audience="weather-api",
            scopes=["weather.read"],
        ),
    ]


@pytest.fixture
def mock_openrouter_api_key(monkeypatch):
    """Set a mock OpenRouter API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
