"""Tests for main FastAPI application with dispatcher integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.llm_service.core.llm.dispatcher import reset_dispatcher


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singletons before each test."""
    reset_dispatcher()
    # Reset registry
    from services.llm_service.api import registry

    registry._registry = None
    yield
    reset_dispatcher()
    registry._registry = None


@pytest.fixture
def mock_model_info():
    """Create a mock ModelInfo."""
    model_info = MagicMock()
    model_info.provider = "openrouter"
    model_info.model_name = "openai/gpt-4o"
    model_info.id = "openrouter/openai/gpt-4o"
    return model_info


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    response = MagicMock()
    response.content = "Hello! How can I help you?"
    response.tool_calls = []
    response.finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20
    response.usage.total_tokens = 30
    response.usage.model_name = "gpt-4o"
    return response


class TestDispatcherIntegration:
    """Tests for dispatcher integration with endpoints."""

    @pytest.mark.asyncio
    async def test_generate_endpoint_uses_dispatcher(
        self, mock_model_info, mock_llm_response
    ):
        """Test that /v1/generate endpoint uses the dispatcher."""
        with (
            patch(
                "services.llm_service.main.get_registry"
            ) as mock_get_registry,
            patch(
                "services.llm_service.main.get_dispatcher"
            ) as mock_get_dispatcher,
        ):
            # Set up mocks
            mock_registry = MagicMock()
            mock_registry.resolve_model.return_value = mock_model_info

            mock_client = AsyncMock()
            mock_client.generate.return_value = mock_llm_response
            mock_registry.get_client = AsyncMock(return_value=mock_client)

            mock_get_registry.return_value = mock_registry

            # Set up dispatcher mock with async context manager
            mock_dispatcher = MagicMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=False)
            mock_dispatcher.acquire = AsyncMock(return_value=mock_context)
            mock_get_dispatcher.return_value = mock_dispatcher

            # Import app after mocking
            from services.llm_service.main import app

            client = TestClient(app)

            # Make request
            response = client.post(
                "/v1/generate",
                json={
                    "use_case": "chat",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Hello! How can I help you?"

            # Verify dispatcher was called with correct provider
            mock_dispatcher.acquire.assert_called_once_with("openrouter")

            # Verify context manager was entered and exited
            mock_context.__aenter__.assert_called_once()
            mock_context.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_endpoint_uses_dispatcher(
        self, mock_model_info
    ):
        """Test that /v1/generate-stream endpoint uses the dispatcher."""
        with (
            patch(
                "services.llm_service.main.get_registry"
            ) as mock_get_registry,
            patch(
                "services.llm_service.main.get_dispatcher"
            ) as mock_get_dispatcher,
        ):
            # Set up mocks
            mock_registry = MagicMock()
            mock_registry.resolve_model.return_value = mock_model_info

            # Create async generator for streaming
            async def mock_stream(*args, **kwargs):
                chunk = MagicMock()
                chunk.delta = "Hello"
                chunk.tool_calls = None
                chunk.finish_reason = None
                chunk.usage = None
                yield chunk

                final_chunk = MagicMock()
                final_chunk.delta = "!"
                final_chunk.tool_calls = None
                final_chunk.finish_reason = "stop"
                final_chunk.usage = MagicMock()
                final_chunk.usage.input_tokens = 10
                final_chunk.usage.output_tokens = 5
                final_chunk.usage.total_tokens = 15
                final_chunk.usage.model_name = "gpt-4o"
                yield final_chunk

            mock_client = AsyncMock()
            mock_client.generate_stream = mock_stream
            mock_registry.get_client = AsyncMock(return_value=mock_client)

            mock_get_registry.return_value = mock_registry

            # Set up dispatcher mock with async context manager
            mock_dispatcher = MagicMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock(return_value=False)
            mock_dispatcher.acquire = AsyncMock(return_value=mock_context)
            mock_get_dispatcher.return_value = mock_dispatcher

            # Import app after mocking
            from services.llm_service.main import app

            client = TestClient(app)

            # Make streaming request
            response = client.post(
                "/v1/generate-stream",
                json={
                    "use_case": "chat",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # Verify response
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-ndjson"

            # Verify dispatcher was called with correct provider
            mock_dispatcher.acquire.assert_called_once_with("openrouter")

    def test_health_endpoint_returns_registry_stats(self):
        """Test that /health endpoint includes registry stats."""
        with patch(
            "services.llm_service.main.get_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get_stats.return_value = {
                "cached_clients": 2,
                "client_keys": ["openrouter:gpt-4o", "openai:gpt-4o"],
            }
            mock_get_registry.return_value = mock_registry

            from services.llm_service.main import app

            client = TestClient(app)

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert data["service"] == "llm_service"
            assert "registry" in data
            assert data["registry"]["cached_clients"] == 2
