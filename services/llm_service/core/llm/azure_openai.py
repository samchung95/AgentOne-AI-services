"""Azure OpenAI LLM client implementation using LangChain."""

from collections.abc import AsyncGenerator
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

from services.llm_service.core.config.constants import ProviderID
from services.llm_service.core.config.settings import Settings, get_settings
from services.llm_service.core.llm.azure_token import get_genai_token_provider
from services.llm_service.core.llm.base import (
    BaseLLMClient,
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
    to_langchain_content,
)
from services.llm_service.core.llm.credentials import (
    APIKeyCredentialProvider,
    AzureADCredentialProvider,
    CredentialProvider,
)
from services.llm_service.core.llm.genai_platform import (
    build_genai_headers,
    resolve_genai_endpoint,
)
from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall
from shared.validators.id_generators import generate_tool_call_id

logger = structlog.get_logger()


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI LLM client using LangChain with GenAI Platform support."""

    def __init__(self, settings: Settings | None = None):
        """Initialize Azure OpenAI client.

        Args:
            settings: Application settings. Uses get_settings() if not provided.
        """
        self._settings = settings or get_settings()
        self._client_instance: AzureChatOpenAI | None = None
        self._model_name = self._settings.azure_openai_deployment
        # Direct config attributes (set by from_model_config)
        self._direct_endpoint: str | None = None
        self._direct_api_key: str | None = None
        self._direct_api_version: str | None = None
        self._genai_platform_enabled: bool = False
        self._genai_platform_base_url: str | None = None
        self._genai_platform_path: str | None = None
        self._genai_platform_user_id: str | None = None
        self._genai_platform_project_name: str | None = None

    @classmethod
    def from_model_config(
        cls,
        deployment: str,
        endpoint: str,
        api_key: str | None = None,
        api_version: str | None = None,
        genai_platform_enabled: bool = False,
        genai_platform_base_url: str | None = None,
        genai_platform_path: str | None = None,
        genai_platform_user_id: str | None = None,
        genai_platform_project_name: str | None = None,
    ) -> "AzureOpenAIClient":
        """Create client from model configuration."""
        instance = cls.__new__(cls)
        instance._settings = None
        instance._client_instance = None
        instance._model_name = deployment
        instance._direct_endpoint = endpoint
        instance._direct_api_key = api_key
        instance._direct_api_version = api_version or "2024-08-01-preview"
        # GenAI Platform config
        instance._genai_platform_enabled = genai_platform_enabled
        instance._genai_platform_base_url = genai_platform_base_url
        instance._genai_platform_path = genai_platform_path
        instance._genai_platform_user_id = genai_platform_user_id
        instance._genai_platform_project_name = genai_platform_project_name
        return instance

    def _get_endpoint(self) -> str:
        """Return the Azure OpenAI endpoint URL.

        Handles three modes:
        - GenAI Platform: Constructs endpoint from base URL and path using resolve_genai_endpoint()
        - Direct/Settings: Returns the configured Azure endpoint

        Returns:
            The Azure OpenAI endpoint URL.
        """
        if self._settings:
            genai_enabled = self._settings.genai_platform_enabled
            genai_base_url = self._settings.genai_platform_base_url
            genai_path = self._settings.genai_platform_path
            direct_endpoint = self._settings.azure_openai_endpoint
        else:
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url
            genai_path = self._genai_platform_path
            direct_endpoint = self._direct_endpoint or ""

        if genai_enabled and genai_base_url:
            return resolve_genai_endpoint(
                base_url=genai_base_url,
                path=genai_path,
                provider=ProviderID.AZURE_OPENAI,
            )

        return direct_endpoint

    def _get_credential_provider(self) -> CredentialProvider:
        """Return the credential provider for Azure OpenAI.

        Handles three modes:
        - GenAI Platform: Uses AzureADCredentialProvider
        - Direct API key: Uses APIKeyCredentialProvider
        - Default: Uses AzureADCredentialProvider (GenAI token without platform)

        Returns:
            The appropriate CredentialProvider for the configuration.
        """
        if self._settings:
            genai_enabled = self._settings.genai_platform_enabled
            genai_base_url = self._settings.genai_platform_base_url
            direct_api_key = self._settings.azure_openai_api_key
        else:
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url
            direct_api_key = self._direct_api_key

        # GenAI Platform mode - use Azure AD token
        if genai_enabled and genai_base_url:
            return AzureADCredentialProvider(get_genai_token_provider())

        # Direct API key mode
        if direct_api_key:
            return APIKeyCredentialProvider(direct_api_key)

        # Default: Azure AD token without GenAI Platform endpoint
        return AzureADCredentialProvider(get_genai_token_provider())

    def _create_client_instance(
        self, credentials: dict[str, Any], endpoint: str
    ) -> AzureChatOpenAI:
        """Create the AzureChatOpenAI client instance.

        Args:
            credentials: Dictionary with either 'api_key' or 'token' from credential provider.
            endpoint: The Azure OpenAI endpoint URL.

        Returns:
            Configured AzureChatOpenAI client.
        """
        # Get API version and GenAI Platform settings
        if self._settings:
            api_version = self._settings.azure_openai_api_version
            genai_enabled = self._settings.genai_platform_enabled
            genai_base_url = self._settings.genai_platform_base_url
            genai_user_id = self._settings.genai_platform_user_id
            genai_project_name = self._settings.genai_platform_project_name
        else:
            api_version = self._direct_api_version or "2024-08-01-preview"
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url
            genai_user_id = self._genai_platform_user_id
            genai_project_name = self._genai_platform_project_name

        # Build headers for GenAI Platform using shared module
        headers = build_genai_headers(
            user_id=genai_user_id,
            project_name=genai_project_name,
        )

        # Extract API key from credentials (could be 'api_key' or 'token')
        api_key = credentials.get("api_key") or credentials.get("token", "")

        client = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=self._model_name,
            api_version=api_version,
            api_key=api_key,
            default_headers=headers if headers else None,
        )

        # Log initialization based on mode
        if genai_enabled and genai_base_url:
            logger.info(
                "azure_openai_client_initialized_genai_platform",
                endpoint=endpoint,
                deployment=self._model_name,
            )
        elif credentials.get("api_key"):
            logger.info(
                "azure_openai_client_initialized",
                endpoint=endpoint,
                deployment=self._model_name,
            )
        else:
            logger.info(
                "azure_openai_client_initialized_genai_token",
                endpoint=endpoint,
                deployment=self._model_name,
            )

        return client

    def _convert_to_langchain_messages(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> list[Any]:
        """Convert LLMMessage list to LangChain messages."""
        lc_messages: list[Any] = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg.role == "system":
                content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                lc_messages.append(SystemMessage(content=content))
            elif msg.role == "user":
                if hasattr(msg, "is_multimodal") and msg.is_multimodal():
                    lc_content = to_langchain_content(msg.content)
                    lc_messages.append(HumanMessage(content=lc_content))
                else:
                    content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                    lc_messages.append(HumanMessage(content=content))
            elif msg.role == "assistant":
                content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        tool_calls.append(
                            {
                                "id": tc.get("id") or tc.get("tool_call_id") or generate_tool_call_id(),
                                "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                                "args": tc.get("args") or tc.get("function", {}).get("arguments", {}),
                            }
                        )
                    lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif msg.role == "tool":
                content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.tool_call_id or "",
                        name=msg.name,
                    )
                )

        return lc_messages

    def _convert_tools_for_binding(
        self,
        tools: list[LLMToolDefinition] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert LLMToolDefinition list for LangChain bind_tools."""
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response using LangChain."""
        chat = self.client

        lc_messages = self._convert_to_langchain_messages(messages, system_prompt)

        if tools:
            tool_schemas = self._convert_tools_for_binding(tools)
            chat = chat.bind_tools(tool_schemas)

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0

        async for chunk in chat.astream(lc_messages, **kwargs):
            if chunk.content:
                content_parts.append(str(chunk.content))
                yield LLMChunk(delta=str(chunk.content))

            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc_chunk in chunk.tool_call_chunks:
                    idx = tc_chunk.get("index", 0)
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                    if tc_chunk.get("id"):
                        tool_calls[idx]["id"] = tc_chunk["id"]
                    if tc_chunk.get("name"):
                        tool_calls[idx]["function"]["name"] = tc_chunk["name"]
                    if tc_chunk.get("args"):
                        tool_calls[idx]["function"]["arguments"] += tc_chunk["args"]

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.get("input_tokens", 0)
                output_tokens = chunk.usage_metadata.get("output_tokens", 0)

        finish_reason = "tool_calls" if tool_calls else "stop"
        yield LLMChunk(
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
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
        """Generate a complete response using LangChain."""
        chat = self.client

        lc_messages = self._convert_to_langchain_messages(messages, system_prompt)

        if tools:
            tool_schemas = self._convert_tools_for_binding(tools)
            chat = chat.bind_tools(tool_schemas)

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await chat.ainvoke(lc_messages, **kwargs)

        content = str(response.content) if response.content else ""

        tool_calls_result: list[ToolCall] = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_defs = {t.name: t for t in (tools or [])}
            for tc in response.tool_calls:
                name = tc.get("name", "")
                tool_def = tool_defs.get(name)
                tool_calls_result.append(
                    ToolCall(
                        tool_call_id=tc.get("id") or generate_tool_call_id(),
                        name=name,
                        args=tc.get("args", {}),
                        audience=tool_def.audience if tool_def else "internal",
                        scopes=tool_def.scopes if tool_def else [],
                    )
                )

        usage: Usage | None = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = Usage(
                input_tokens=response.usage_metadata.get("input_tokens", 0),
                output_tokens=response.usage_metadata.get("output_tokens", 0),
                total_tokens=response.usage_metadata.get("total_tokens", 0),
                model_name=self._model_name,
            )

        finish_reason = "tool_calls" if tool_calls_result else "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls_result,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def close(self) -> None:
        """Close the client."""
        self._client_instance = None
        logger.info("azure_openai_client_closed")
