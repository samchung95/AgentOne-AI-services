"""OpenAI LLM client implementation."""

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
from services.llm_service.core.llm.credentials import APIKeyCredentialProvider, CredentialProvider
from services.llm_service.core.llm.mixin import OpenAICompatibleMixin, StreamingToolCallAggregator
from services.llm_service.core.llm.retry import DEFAULT_RETRY_CONFIG, with_retry
from shared.protocol.common import Usage

logger = structlog.get_logger()


class OpenAIClient(BaseLLMClient, OpenAICompatibleMixin):
    """OpenAI API client with streaming, tool calling, and retry support."""

    def __init__(self, settings: Settings | None = None):
        """Initialize OpenAI client.

        Args:
            settings: Application settings. Uses get_settings() if not provided.
        """
        self._settings = settings or get_settings()
        self._client_instance: AsyncOpenAI | None = None
        self._model_name = self._settings.openai.model
        self._direct_api_key: str | None = None
        self._direct_base_url: str | None = None

    @classmethod
    def from_model_config(
        cls,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> "OpenAIClient":
        """Create client from model configuration.

        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: OpenAI API key
            base_url: Optional base URL override

        Returns:
            Configured OpenAIClient
        """
        instance = cls.__new__(cls)
        instance._settings = None
        instance._client_instance = None
        instance._model_name = model
        instance._direct_api_key = api_key
        instance._direct_base_url = base_url
        return instance

    def _get_endpoint(self) -> str:
        """Return the OpenAI API endpoint URL.

        Returns:
            The base URL for OpenAI API, or empty string for default.
        """
        if self._settings:
            return self._settings.openai.base_url or ""
        return self._direct_base_url or ""

    def _get_credential_provider(self) -> CredentialProvider:
        """Return the credential provider for OpenAI.

        Returns:
            APIKeyCredentialProvider with the configured API key.
        """
        if self._settings:
            api_key = self._settings.openai.api_key or ""
        else:
            api_key = self._direct_api_key or ""
        return APIKeyCredentialProvider(api_key)

    def _create_client_instance(
        self, credentials: dict[str, Any], endpoint: str
    ) -> AsyncOpenAI:
        """Create the AsyncOpenAI client instance.

        Args:
            credentials: Dictionary with 'api_key' from credential provider.
            endpoint: The API endpoint URL (empty string for default).

        Returns:
            Configured AsyncOpenAI client.
        """
        api_key = credentials.get("api_key", "")
        base_url = endpoint if endpoint else None

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        logger.info(
            "openai_client_initialized",
            model=self._model_name,
            base_url=base_url or "default",
        )

        return client

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response from OpenAI."""
        client = self.client

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
        """Generate a complete response from OpenAI."""
        # Use streaming and aggregate
        return await self._aggregate_stream_to_response(
            self.generate_stream(messages, tools, system_prompt, temperature, max_tokens),
            tools,
        )

    async def close(self) -> None:
        """Close the client."""
        if self._client_instance:
            await self._client_instance.close()
            self._client_instance = None
            logger.info("openai_client_closed")
