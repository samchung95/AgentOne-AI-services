"""LLM Client Registry with caching and model resolution."""

import structlog

from services.llm_service.core.config.models import (
    ModelInfo,
    get_model_for_use_case,
    load_model_config,
)
from services.llm_service.core.llm.base import BaseLLMClient
from services.llm_service.core.llm.factory import LLMFactory

logger = structlog.get_logger(__name__)


class LLMClientRegistry:
    """Registry for LLM clients with caching.

    Clients are cached by (provider, model_name) to avoid recreating
    connections for repeated requests.
    """

    def __init__(self):
        """Initialize the registry."""
        self._clients: dict[str, BaseLLMClient] = {}

    def _cache_key(self, provider: str, model_name: str) -> str:
        """Generate cache key for a provider/model combination."""
        return f"{provider}:{model_name}"

    def _resolve_model(self, use_case: str, model: str | None) -> ModelInfo:
        """Resolve model info from use case and optional model string.

        Args:
            use_case: Use case (chat, vision, embedding).
            model: Optional model string (can be full ID or just model name).

        Returns:
            Resolved ModelInfo.
        """
        if model:
            # Try to parse as full model ID (provider/model)
            config = load_model_config()
            resolved = config.get_model(model)
            if resolved:
                return resolved

            # Try to find in any provider
            for provider_info in config.providers.values():
                if model in provider_info.models:
                    return provider_info.models[model]

        # Fall back to use case default
        return get_model_for_use_case(use_case)

    async def get_client(
        self,
        use_case: str = "chat",
        model: str | None = None,
    ) -> BaseLLMClient:
        """Get or create an LLM client.

        Args:
            use_case: Use case for default model selection.
            model: Optional specific model to use.

        Returns:
            Cached or newly created BaseLLMClient.
        """
        model_info = self._resolve_model(use_case, model)
        cache_key = self._cache_key(model_info.provider, model_info.model_name)

        if cache_key not in self._clients:
            logger.info(
                "creating_cached_llm_client",
                provider=model_info.provider,
                model=model_info.model_name,
                cache_key=cache_key,
            )
            self._clients[cache_key] = LLMFactory.create_for_model(model_info)

        return self._clients[cache_key]

    async def close_all(self) -> None:
        """Close all cached clients."""
        for cache_key, client in self._clients.items():
            try:
                await client.close()
                logger.info("closed_cached_client", cache_key=cache_key)
            except Exception as e:
                logger.warning("error_closing_client", cache_key=cache_key, error=str(e))

        self._clients.clear()

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "cached_clients": len(self._clients),
            "client_keys": list(self._clients.keys()),
        }


# Global registry instance
_registry: LLMClientRegistry | None = None


def get_registry() -> LLMClientRegistry:
    """Get the global LLM client registry."""
    global _registry
    if _registry is None:
        _registry = LLMClientRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
