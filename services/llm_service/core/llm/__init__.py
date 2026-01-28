"""LLM clients and factory."""

from services.llm_service.core.llm.base import BaseLLMClient, LLMChunk, LLMResponse
from services.llm_service.core.llm.dispatcher import (
    DispatcherConfig,
    LLMDispatcher,
    ProviderHealth,
    get_dispatcher,
    reset_dispatcher,
)
from services.llm_service.core.llm.factory import LLMFactory
from services.llm_service.core.llm.remote_client import RemoteLLMClient
from services.llm_service.core.llm.retry import (
    RetryConfig,
    RetryStrategy,
    calculate_delay,
    is_retryable_error,
    with_retry,
)
from services.llm_service.core.llm.types import LLMToolCallChunk, LLMToolCallFunction, ToolCallExtensions

__all__ = [
    # Base
    "BaseLLMClient",
    "LLMChunk",
    "LLMResponse",
    # Types
    "LLMToolCallChunk",
    "LLMToolCallFunction",
    "ToolCallExtensions",
    # Factory
    "LLMFactory",
    "RemoteLLMClient",
    # Dispatcher
    "LLMDispatcher",
    "DispatcherConfig",
    "ProviderHealth",
    "get_dispatcher",
    "reset_dispatcher",
    # Retry
    "RetryConfig",
    "RetryStrategy",
    "calculate_delay",
    "is_retryable_error",
    "with_retry",
]
