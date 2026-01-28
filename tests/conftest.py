"""Pytest configuration and fixtures."""

import os

import pytest

# Set test environment before importing settings
os.environ.setdefault("CONFIG_PROFILE", "openrouter")
os.environ.setdefault("LLM_SERVICE_LOAD_ENV_FILE", "false")


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
