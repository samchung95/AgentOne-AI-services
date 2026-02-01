"""Google Vertex AI LLM client implementation using LangChain.

Supports GenAI Platform gateway mode with Azure AD authentication via SPCredentials.
Uses LangChain's ChatGoogleGenerativeAI (from langchain-google-genai>=4.0.0) with
vertexai=True for Vertex AI API calls.
"""

import asyncio
import base64
import json
import threading
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from services.llm_service.core.config.constants import ProviderID
from services.llm_service.core.config.settings import Settings, get_settings
from services.llm_service.core.llm.azure_token import SPCredentials
from services.llm_service.core.llm.base import (
    BaseLLMClient,
    LLMChunk,
    LLMMessage,
    LLMResponse,
    LLMToolDefinition,
    to_langchain_content,
)
from services.llm_service.core.llm.credentials import CredentialProvider, GCPCredentialProvider
from services.llm_service.core.llm.genai_platform import (
    build_genai_headers,
    resolve_genai_endpoint,
)
from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall
from shared.validators.id_generators import generate_tool_call_id, normalize_tool_call_id

logger = structlog.get_logger()

# Default Vertex AI configuration
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_VERTEX_MODEL = "gemini-2.5-flash-lite"
_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"


class VertexAIClient(BaseLLMClient):
    """Google Vertex AI LLM client using LangChain with GenAI Platform support.

    Uses ChatGoogleGenerativeAI with vertexai=True for Vertex AI API calls.
    Supports SPCredentials for Azure AD tokens when using GenAI Platform gateway.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize Vertex AI client."""
        self._settings = settings or get_settings()
        self._client_instance: ChatGoogleGenerativeAI | None = None
        self._model_name = self._settings.vertex_ai.model
        self._force_sync: bool = False
        # Direct config attributes (set by from_model_config)
        self._direct_project_id: str | None = None
        self._direct_location: str = DEFAULT_VERTEX_LOCATION
        self._direct_api_key: str | None = None
        self._genai_platform_enabled: bool = False
        self._genai_platform_base_url: str | None = None
        self._genai_platform_path: str | None = None
        self._genai_platform_user_id: str | None = None
        self._genai_platform_project_name: str | None = None

    @classmethod
    def from_model_config(
        cls,
        model: str,
        project_id: str | None = None,
        location: str = DEFAULT_VERTEX_LOCATION,
        api_key: str | None = None,
        genai_platform_enabled: bool = False,
        genai_platform_base_url: str | None = None,
        genai_platform_path: str | None = None,
        genai_platform_user_id: str | None = None,
        genai_platform_project_name: str | None = None,
    ) -> "VertexAIClient":
        """Create client from model configuration."""
        instance = cls.__new__(cls)
        instance._settings = None
        instance._client_instance = None
        instance._model_name = model
        instance._direct_project_id = project_id
        instance._direct_location = location
        instance._direct_api_key = api_key
        instance._force_sync = False
        instance._genai_platform_enabled = genai_platform_enabled
        instance._genai_platform_base_url = genai_platform_base_url
        instance._genai_platform_path = genai_platform_path
        instance._genai_platform_user_id = genai_platform_user_id
        instance._genai_platform_project_name = genai_platform_project_name
        return instance

    def _get_endpoint(self) -> str:
        """Return the Vertex AI endpoint URL.

        Handles two modes:
        - GenAI Platform: Constructs endpoint from base URL and path using resolve_genai_endpoint()
        - Direct Vertex AI: Returns empty string (uses Google Cloud defaults)

        Returns:
            The Vertex AI endpoint URL, or empty string for direct mode.
        """
        if self._settings:
            genai_enabled = self._settings.genai_platform.enabled
            genai_base_url = self._settings.genai_platform.base_url
            genai_path = self._settings.genai_platform.path
        else:
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url
            genai_path = self._genai_platform_path

        if genai_enabled and genai_base_url:
            return resolve_genai_endpoint(
                base_url=genai_base_url,
                path=genai_path,
                provider=ProviderID.VERTEX_AI,
            )

        return ""

    def _get_credential_provider(self) -> CredentialProvider | None:
        """Return the credential provider for Vertex AI.

        Handles two modes:
        - GenAI Platform: Returns None (uses SPCredentials directly in _create_client_instance)
        - Direct Vertex AI: Returns GCPCredentialProvider for ADC

        Note: For GenAI Platform mode, SPCredentials is used directly in _create_client_instance
        because it implements the Google Credentials interface required by ChatGoogleGenerativeAI.

        Returns:
            GCPCredentialProvider for direct mode, or None for GenAI Platform mode.
        """
        if self._settings:
            genai_enabled = self._settings.genai_platform.enabled
            genai_base_url = self._settings.genai_platform.base_url
        else:
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url

        # GenAI Platform mode uses SPCredentials directly
        if genai_enabled and genai_base_url:
            return None

        # Direct Vertex AI mode uses GCP ADC
        return GCPCredentialProvider()

    def _create_client_instance(
        self, credentials: dict[str, Any], endpoint: str
    ) -> ChatGoogleGenerativeAI:
        """Create the ChatGoogleGenerativeAI client instance.

        Args:
            credentials: Dictionary with 'credentials' key for GCP credentials (direct mode),
                        or empty dict for GenAI Platform mode (uses SPCredentials).
            endpoint: The Vertex AI endpoint URL (GenAI Platform) or empty string (direct).

        Returns:
            Configured ChatGoogleGenerativeAI client.
        """
        if self._settings:
            genai_enabled = self._settings.genai_platform.enabled
            genai_base_url = self._settings.genai_platform.base_url
            genai_user_id = self._settings.genai_platform.user_id
            genai_project_name = self._settings.genai_platform.project_name
            project_id = self._settings.vertex_ai.project_id
            location = self._settings.vertex_ai.location
        else:
            genai_enabled = self._genai_platform_enabled
            genai_base_url = self._genai_platform_base_url
            genai_user_id = self._genai_platform_user_id
            genai_project_name = self._genai_platform_project_name
            project_id = self._direct_project_id
            location = self._direct_location

        # Build headers for GenAI Platform using shared module
        headers = build_genai_headers(
            user_id=genai_user_id,
            project_name=genai_project_name,
        )

        if genai_enabled and genai_base_url:
            # GenAI Platform mode - use SPCredentials (Azure AD token adapter)
            client = ChatGoogleGenerativeAI(
                model=self._model_name,
                vertexai=True,
                base_url=endpoint,
                credentials=SPCredentials(),
                project="genai-platform",
                location=location,
                additional_headers=headers if headers else None,
            )
            # Keep sync fallback for GenAI Platform mode until native async is verified
            # with custom endpoints. The langchain-google-genai SDK may have issues
            # with custom base_url in async mode.
            self._force_sync = True

            logger.info(
                "vertex_client_initialized_genai_platform",
                endpoint=endpoint,
                model=self._model_name,
            )
            return client

        elif project_id:
            # Direct Vertex AI mode - uses GCP ADC from credentials dict
            client = ChatGoogleGenerativeAI(
                model=self._model_name,
                vertexai=True,
                project=project_id,
                location=location,
            )
            # Direct Vertex AI mode supports native async streaming
            self._force_sync = False

            logger.info(
                "vertex_client_initialized_vertex_ai",
                project_id=project_id,
                location=location,
                model=self._model_name,
            )
            return client

        else:
            raise RuntimeError(
                "Vertex AI requires either: GenAI Platform config (GENAI_PLATFORM_ENABLED=true) "
                "or VERTEX_PROJECT_ID for direct Vertex AI access"
            )

    def _initialize_client(self) -> ChatGoogleGenerativeAI:
        """Initialize the Vertex AI client.

        Overrides the base template method because Vertex AI has two distinct
        credential modes that require different handling:
        - GenAI Platform: Uses SPCredentials (no CredentialProvider)
        - Direct Vertex AI: Uses GCPCredentialProvider

        Returns:
            The initialized ChatGoogleGenerativeAI client.
        """
        credential_provider = self._get_credential_provider()
        if credential_provider is not None:
            credentials = credential_provider.get_credentials()
        else:
            # GenAI Platform mode - SPCredentials used directly in _create_client_instance
            credentials = {}

        endpoint = self._get_endpoint()
        return self._create_client_instance(credentials, endpoint)

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
                if msg.is_multimodal():
                    lc_content = to_langchain_content(msg.content)
                    lc_messages.append(HumanMessage(content=lc_content))
                else:
                    content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                    lc_messages.append(HumanMessage(content=content))
            elif msg.role == "assistant":
                content = msg.get_text_content() if hasattr(msg, "get_text_content") else (msg.content or "")
                if msg.tool_calls:
                    tool_calls = []
                    gemini_tool_call_signatures: dict[str, str] = {}
                    for tc in msg.tool_calls:
                        args: Any = tc.get("args")
                        if args is None:
                            args = tc.get("function", {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        raw_tc_id = tc.get("id") or tc.get("tool_call_id") or generate_tool_call_id()
                        tool_call_id, _ = normalize_tool_call_id(
                            raw_tc_id if isinstance(raw_tc_id, str) else None
                        )

                        thought_signature = tc.get("thought_signature")
                        if not thought_signature:
                            thought_signature = tc.get("function", {}).get("thought_signature")
                        if isinstance(thought_signature, str) and thought_signature:
                            gemini_tool_call_signatures[tool_call_id] = thought_signature

                        tool_calls.append(
                            {
                                "id": tool_call_id,
                                "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                                "args": args,
                            }
                        )

                    additional_kwargs: dict[str, Any] = {}
                    if gemini_tool_call_signatures:
                        additional_kwargs[_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = gemini_tool_call_signatures
                    lc_messages.append(
                        AIMessage(
                            content=content,
                            tool_calls=tool_calls,
                            additional_kwargs=additional_kwargs,
                        )
                    )
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

    def _langchain_content_to_text(self, content: Any) -> str:
        """Extract displayable text from LangChain chunk content."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, (list, tuple)):
            return "".join(self._langchain_content_to_text(part) for part in content)

        if isinstance(content, dict):
            text = content.get("text")
            if text is None:
                return ""
            return text if isinstance(text, str) else str(text)

        text_attr = getattr(content, "text", None)
        if text_attr is None:
            return ""
        return text_attr if isinstance(text_attr, str) else str(text_attr)

    async def _iter_sync_stream_as_async(
        self,
        chat: Any,
        lc_messages: list[Any],
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """Run LangChain's sync .stream() in a thread and yield chunks asynchronously."""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def safe_put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass

        def producer() -> None:
            try:
                for item in chat.stream(lc_messages, **kwargs):
                    safe_put(item)
            except BaseException as exc:
                safe_put(exc)
            finally:
                safe_put(sentinel)

        threading.Thread(target=producer, name="vertex-sync-stream", daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response using LangChain ChatGoogleGenerativeAI."""
        chat = self.client

        lc_messages = self._convert_to_langchain_messages(messages, system_prompt)

        if tools:
            tool_schemas = self._convert_tools_for_binding(tools)
            chat = chat.bind_tools(tool_schemas)

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_call_id_to_index: dict[str, int] = {}
        gemini_tool_call_signatures: dict[str, str] = {}
        input_tokens = 0
        output_tokens = 0

        stream_iter = (
            self._iter_sync_stream_as_async(chat, lc_messages, kwargs)
            if self._force_sync
            else chat.astream(lc_messages, **kwargs)
        )

        async for chunk in stream_iter:
            additional_kwargs = getattr(chunk, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                sig_map = additional_kwargs.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
                if isinstance(sig_map, dict):
                    for raw_id, sig in sig_map.items():
                        if not isinstance(raw_id, str) or not raw_id:
                            continue
                        if isinstance(sig, bytes):
                            sig_str = base64.b64encode(sig).decode("utf-8")
                        elif isinstance(sig, str):
                            sig_str = sig
                        else:
                            continue
                        normalized_id, _ = normalize_tool_call_id(raw_id)
                        gemini_tool_call_signatures[normalized_id] = sig_str

            if chunk.content:
                text = self._langchain_content_to_text(chunk.content)
                if text:
                    content_parts.append(text)
                    yield LLMChunk(delta=text)

            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc_chunk in chunk.tool_call_chunks:
                    idx_raw = tc_chunk.get("index")
                    idx: int | None
                    if isinstance(idx_raw, int):
                        idx = idx_raw
                    elif isinstance(idx_raw, str) and idx_raw.isdigit():
                        idx = int(idx_raw)
                    else:
                        idx = None

                    if idx is None or idx < 0:
                        tc_id = tc_chunk.get("id")
                        if isinstance(tc_id, str) and tc_id:
                            idx = tool_call_id_to_index.get(tc_id)
                            if idx is None:
                                idx = len(tool_calls)
                                tool_call_id_to_index[tc_id] = idx
                        else:
                            idx = 0

                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                    if tc_chunk.get("id"):
                        raw_id = tc_chunk["id"]
                        normalized_id, _ = normalize_tool_call_id(raw_id if isinstance(raw_id, str) else None)
                        tool_calls[idx]["id"] = normalized_id
                    if tc_chunk.get("name"):
                        tool_calls[idx]["function"]["name"] = tc_chunk["name"]
                    if tc_chunk.get("args"):
                        tool_calls[idx]["function"]["arguments"] += tc_chunk["args"]

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.get("input_tokens", 0)
                output_tokens = chunk.usage_metadata.get("output_tokens", 0)

        if gemini_tool_call_signatures and tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id in gemini_tool_call_signatures:
                    tc["thought_signature"] = gemini_tool_call_signatures[tc_id]

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
        """Generate a complete response using LangChain ChatGoogleGenerativeAI."""
        chat = self.client

        lc_messages = self._convert_to_langchain_messages(messages, system_prompt)

        if tools:
            tool_schemas = self._convert_tools_for_binding(tools)
            chat = chat.bind_tools(tool_schemas)

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        if self._force_sync:
            response = await asyncio.to_thread(chat.invoke, lc_messages, **kwargs)
        else:
            response = await chat.ainvoke(lc_messages, **kwargs)

        content = self._langchain_content_to_text(response.content) if response.content else ""

        gemini_tool_call_signatures: dict[str, str] = {}
        additional_kwargs = getattr(response, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            sig_map = additional_kwargs.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
            if isinstance(sig_map, dict):
                for raw_id, sig in sig_map.items():
                    if not isinstance(raw_id, str) or not raw_id:
                        continue
                    if isinstance(sig, bytes):
                        sig_str = base64.b64encode(sig).decode("utf-8")
                    elif isinstance(sig, str):
                        sig_str = sig
                    else:
                        continue
                    normalized_sig_id, _ = normalize_tool_call_id(raw_id)
                    gemini_tool_call_signatures[normalized_sig_id] = sig_str

        tool_calls_result: list[ToolCall] = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_defs = {t.name: t for t in (tools or [])}
            for tc in response.tool_calls:
                name = tc.get("name", "")
                tool_def = tool_defs.get(name)
                raw_tc_id = tc.get("id") if isinstance(tc.get("id"), str) else None
                tool_call_id, original_id = normalize_tool_call_id(raw_tc_id)
                extensions: dict[str, Any] | None = None
                if tool_call_id in gemini_tool_call_signatures:
                    extensions = {"thought_signature": gemini_tool_call_signatures[tool_call_id]}
                tool_calls_result.append(
                    ToolCall(
                        tool_call_id=tool_call_id,
                        provider_id=original_id,
                        name=name,
                        args=tc.get("args", {}),
                        audience=tool_def.audience if tool_def else "internal",
                        scopes=tool_def.scopes if tool_def else [],
                        extensions=extensions,
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
        logger.info("vertex_client_closed")
