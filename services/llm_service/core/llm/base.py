"""Base LLM client interface.

Defines the abstract interface for LLM providers and multimodal content types.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall

if TYPE_CHECKING:
    from services.llm_service.core.llm.credentials import CredentialProvider

# =============================================================================
# Multimodal Content Types
# =============================================================================


class TextContent(BaseModel):
    """Text content block for multimodal messages."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block for multimodal messages.

    Supports two source types:
    - base64: Image data encoded as base64 string
    - url: Direct URL to the image (for providers that support it)
    """

    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"]
    media_type: str  # MIME type: "image/png", "image/jpeg", "image/gif", "image/webp"
    data: str  # base64 encoded data or URL depending on source_type


# Union type for all content block types
ContentBlock = TextContent | ImageContent


def to_langchain_content(content: str | list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert LLMMessage content to LangChain multimodal format.

    Args:
        content: Either a string or list of ContentBlock objects.

    Returns:
        List of content dictionaries in LangChain format:
        - {"type": "text", "text": "..."} for text
        - {"type": "image_url", "image_url": {"url": "..."}} for images
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            if block.source_type == "base64":
                # Format: data:{media_type};base64,{data}
                image_url = f"data:{block.media_type};base64,{block.data}"
            else:
                # Direct URL
                image_url = block.data
            result.append({"type": "image_url", "image_url": {"url": image_url}})
    return result


def content_to_string(content: str | list[ContentBlock] | None) -> str:
    """Extract text content as a string.

    Useful for cases where only text content is needed (e.g., logging, display).

    Args:
        content: String, list of ContentBlock, or None.

    Returns:
        Concatenated text from all TextContent blocks, or empty string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
    return "\n".join(text_parts)


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class LLMMessage:
    """A message in the LLM conversation.

    Supports both simple string content and multimodal content (text + images).
    Also supports reasoning_content for models with thinking/reasoning capabilities.
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[ContentBlock] | None = None  # String or multimodal blocks
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Tool name for tool responses
    reasoning_content: str | None = None  # For models with thinking/reasoning (e.g., DeepSeek, Claude)

    def get_text_content(self) -> str:
        """Get the text content as a string.

        For multimodal content, concatenates all TextContent blocks.
        """
        return content_to_string(self.content)

    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content (images)."""
        if not isinstance(self.content, list):
            return False
        return any(isinstance(block, ImageContent) for block in self.content)


@dataclass
class LLMToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    audience: str  # API/service identifier for OBO token
    scopes: list[str]  # OAuth scopes required


@dataclass
class LLMChunk:
    """A streaming chunk from the LLM."""

    delta: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None  # "stop", "tool_calls", "length", "content_filter"
    usage: Usage | None = None
    reasoning_content: str | None = None  # For models with thinking/reasoning


@dataclass
class LLMResponse:
    """Complete response from the LLM."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage | None = None
    reasoning_content: str | None = None  # For models with thinking/reasoning


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    Provides a template method pattern for client initialization. Subclasses
    can override `_create_client_instance()` and `_get_endpoint()` to customize
    client creation, then use the `_client` property for lazy initialization.

    The initialization template method ensures consistent ordering:
    1. Get endpoint URL via `_get_endpoint()`
    2. Create client instance via `_create_client_instance(credentials, endpoint)`
    """

    # Subclasses should set this to hold the actual client instance
    _client_instance: Any = None

    def _get_endpoint(self) -> str:
        """Return the API endpoint URL for this provider.

        Subclasses should override this method to provide the appropriate
        endpoint URL based on their configuration.

        Returns:
            The API endpoint URL string.
        """
        return ""

    def _get_credential_provider(self) -> "CredentialProvider | None":
        """Return the credential provider for this client.

        Subclasses should override this method to provide the appropriate
        credential provider (APIKeyCredentialProvider, AzureADCredentialProvider,
        GCPCredentialProvider, etc.) based on their configuration.

        Returns:
            A CredentialProvider instance, or None if not using the template.
        """
        return None

    def _create_client_instance(
        self, credentials: dict[str, Any], endpoint: str
    ) -> Any:
        """Create the actual client instance.

        Subclasses should override this method to instantiate their specific
        client type (AsyncOpenAI, AzureChatOpenAI, ChatGoogleGenerativeAI, etc.)
        using the provided credentials and endpoint.

        Args:
            credentials: Dictionary of credentials from the credential provider.
            endpoint: The API endpoint URL from `_get_endpoint()`.

        Returns:
            The instantiated client object.
        """
        raise NotImplementedError(
            "Subclasses must implement _create_client_instance() to use the template"
        )

    def _initialize_client(self) -> Any:
        """Template method for initializing the client.

        This method orchestrates client initialization by:
        1. Getting the credential provider via `_get_credential_provider()`
        2. Getting credentials from the provider
        3. Getting the endpoint via `_get_endpoint()`
        4. Creating the client instance via `_create_client_instance()`

        Returns:
            The initialized client instance.

        Raises:
            NotImplementedError: If credential provider is not configured.
        """
        credential_provider = self._get_credential_provider()
        if credential_provider is None:
            raise NotImplementedError(
                "Subclasses must implement _get_credential_provider() to use "
                "the initialization template, or override _initialize_client()"
            )

        credentials = credential_provider.get_credentials()
        endpoint = self._get_endpoint()
        return self._create_client_instance(credentials, endpoint)

    @property
    def client(self) -> Any:
        """Lazily initialize and return the client instance.

        This property provides lazy initialization of the underlying client.
        On first access, it calls `_initialize_client()` to create the client.
        Subsequent accesses return the cached instance.

        Returns:
            The initialized client instance.
        """
        if self._client_instance is None:
            self._client_instance = self._initialize_client()
        return self._client_instance

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            system_prompt: System prompt to prepend.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Yields:
            LLMChunk objects with deltas and metadata.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            system_prompt: System prompt to prepend.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Returns:
            Complete LLMResponse.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        ...
