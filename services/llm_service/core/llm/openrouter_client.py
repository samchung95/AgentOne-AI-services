"""OpenRouter LLM client implementation.

OpenRouter provides access to multiple LLM providers through a unified API.
Uses the OpenAI-compatible format with custom headers.
"""

from collections.abc import AsyncGenerator
from typing import Any

import structlog
from openai import AsyncOpenAI

from services.llm_service.core.config.settings import Settings, get_settings
from services.llm_service.core.llm.base import (
    BaseLLMClient,
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
)
from services.llm_service.core.llm.mixin import OpenAICompatibleMixin, StreamingToolCallAggregator
from services.llm_service.core.llm.retry import DEFAULT_RETRY_CONFIG, with_retry
from shared.protocol.common import Usage

logger = structlog.get_logger()


class OpenRouterClient(BaseLLMClient, OpenAICompatibleMixin):
    """OpenRouter API client with streaming, tool calling, and retry support.

    OpenRouter provides access to models from various providers (OpenAI, Anthropic,
    Google, Meta, etc.) through a unified API compatible with OpenAI's format.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize OpenRouter client.

        Args:
            settings: Application settings. Uses get_settings() if not provided.
        """
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        self._model_name = self._settings.openrouter_model

    @classmethod
    def from_model_config(
        cls,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        app_name: str = "AgentOne",
    ) -> "OpenRouterClient":
        """Create client from model configuration.

        Args:
            model: Model name (e.g., "openai/gpt-4o")
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            site_url: Optional site URL for OpenRouter headers
            app_name: Application name for OpenRouter headers

        Returns:
            Configured OpenRouterClient
        """
        instance = cls.__new__(cls)
        instance._settings = None
        instance._client = None
        instance._model_name = model
        instance._direct_api_key = api_key
        instance._direct_base_url = base_url
        instance._direct_site_url = site_url
        instance._direct_app_name = app_name
        return instance

    def _ensure_client(self) -> AsyncOpenAI:
        """Ensure the async client is initialized."""
        if self._client is None:
            if self._settings:
                api_key = self._settings.openrouter_api_key
                base_url = self._settings.openrouter_base_url
                site_url = self._settings.openrouter_site_url
                app_name = self._settings.openrouter_app_name or "AgentOne"
            else:
                api_key = getattr(self, "_direct_api_key", "")
                base_url = getattr(self, "_direct_base_url", "https://openrouter.ai/api/v1")
                site_url = getattr(self, "_direct_site_url", None)
                app_name = getattr(self, "_direct_app_name", "AgentOne")

            # Build default headers for OpenRouter
            default_headers = {"X-Title": app_name}
            if site_url:
                default_headers["HTTP-Referer"] = site_url

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            )

            logger.info(
                "openrouter_client_initialized",
                model=self._model_name,
                base_url=base_url,
            )

        return self._client

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response from OpenRouter."""
        client = self._ensure_client()

        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system_prompt)

        # Convert tools if provided
        openai_tools = self._convert_tools(tools)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if openai_tools:
            kwargs["tools"] = openai_tools
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Track aggregated content
        aggregator = StreamingToolCallAggregator()
        input_tokens = 0
        output_tokens = 0

        @with_retry(DEFAULT_RETRY_CONFIG)
        async def _stream_with_retry():
            return await client.chat.completions.create(**kwargs)

        response = await _stream_with_retry()

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

            # Track usage from stream events
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

            if delta:
                # Yield content delta
                if delta.content:
                    yield LLMChunk(delta=delta.content)

                # Aggregate tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        aggregator.add_chunk(tc)

            # Final chunk with finish reason
            if finish_reason:
                yield LLMChunk(
                    finish_reason=finish_reason,
                    tool_calls=aggregator.get_complete_calls(),
                    usage=Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        model_name=self._model_name,
                    ),
                )

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response from OpenRouter."""
        # Use streaming and aggregate
        return await self._aggregate_stream_to_response(
            self.generate_stream(messages, tools, system_prompt, temperature, max_tokens),
            tools,
        )

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("openrouter_client_closed")
