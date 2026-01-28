"""LLM Factory for creating provider-specific clients.

Provides a unified interface for creating LLM clients based on:
- Model configuration (config/models.yaml)
- Provider credentials (environment variables)
- Use case defaults (chat, vision, embedding)
"""

import structlog

from services.llm_service.core.config.models import (
    ModelInfo,
    get_model_for_use_case,
    get_provider_credentials,
    load_model_config,
)
from services.llm_service.core.llm.base import BaseLLMClient
from services.llm_service.core.llm.exceptions import ConfigurationError, MissingAPIKeyError

logger = structlog.get_logger()


class LLMFactory:
    """Factory for creating LLM clients based on configuration."""

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        model: str | None = None,
    ) -> BaseLLMClient:
        """Create an LLM client for the default chat use case.

        Args:
            provider: Optional provider override. If not specified, uses default.
            model: Optional model override. If not specified, uses default.

        Returns:
            Configured BaseLLMClient instance.

        Raises:
            ConfigurationError: If configuration is invalid.
            MissingAPIKeyError: If required API keys are missing.
        """
        return cls.create_for_use_case("chat", provider=provider, model=model)

    @classmethod
    def create_for_use_case(
        cls,
        use_case: str = "chat",
        provider: str | None = None,
        model: str | None = None,
    ) -> BaseLLMClient:
        """Create an LLM client for a specific use case.

        Args:
            use_case: One of "chat", "vision", "embedding".
            provider: Optional provider override.
            model: Optional model override.

        Returns:
            Configured BaseLLMClient instance.

        Raises:
            ConfigurationError: If configuration is invalid.
            MissingAPIKeyError: If required API keys are missing.
        """
        model_info = get_model_for_use_case(use_case)

        # Apply overrides
        if provider:
            config = load_model_config()
            provider_info = config.get_provider(provider)
            if provider_info and provider_info.models:
                # Use first model from provider or the specified one
                if model and model in provider_info.models:
                    model_info = provider_info.models[model]
                else:
                    model_info = next(iter(provider_info.models.values()))

        if model:
            # If model specified as full ID (provider/model), parse it
            config = load_model_config()
            resolved = config.get_model(model)
            if resolved:
                model_info = resolved

        return cls.create_for_model(model_info)

    @classmethod
    def create_for_model(cls, model_info: ModelInfo) -> BaseLLMClient:
        """Create an LLM client for a specific model.

        Args:
            model_info: Model information from configuration.

        Returns:
            Configured BaseLLMClient instance.

        Raises:
            ConfigurationError: If provider is not supported.
            MissingAPIKeyError: If required API keys are missing.
        """
        provider = model_info.provider
        model_name = model_info.model_name
        creds = get_provider_credentials(provider)

        logger.info(
            "creating_llm_client",
            provider=provider,
            model=model_name,
            full_id=model_info.id,
        )

        if provider == "openrouter":
            return cls._create_openrouter_client(model_name, creds)
        elif provider == "openai":
            return cls._create_openai_client(model_name, creds)
        elif provider == "azure_openai":
            return cls._create_azure_openai_client(model_info, creds)
        elif provider == "vertex_ai":
            return cls._create_vertex_client(model_info, creds)
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def _create_openrouter_client(cls, model_name: str, creds: dict) -> BaseLLMClient:
        """Create an OpenRouter client."""
        from services.llm_service.core.llm.openrouter_client import OpenRouterClient

        api_key = creds.get("api_key")
        if not api_key:
            raise MissingAPIKeyError("openrouter", "OPENROUTER_API_KEY")

        return OpenRouterClient.from_model_config(
            model=model_name,
            api_key=api_key,
            site_url=creds.get("site_url"),
            app_name=creds.get("app_name", "AgentOne"),
        )

    @classmethod
    def _create_openai_client(cls, model_name: str, creds: dict) -> BaseLLMClient:
        """Create an OpenAI client."""
        from services.llm_service.core.llm.openai_client import OpenAIClient

        api_key = creds.get("api_key")
        if not api_key:
            raise MissingAPIKeyError("openai", "OPENAI_API_KEY")

        return OpenAIClient.from_model_config(
            model=model_name,
            api_key=api_key,
        )

    @classmethod
    def _create_azure_openai_client(cls, model_info: ModelInfo, creds: dict) -> BaseLLMClient:
        """Create an Azure OpenAI client."""
        from services.llm_service.core.llm.azure_openai import AzureOpenAIClient

        deployment = model_info.deployment or model_info.model_name
        endpoint = creds.get("endpoint")
        api_key = creds.get("api_key")
        api_version = creds.get("api_version", "2024-08-01-preview")

        # Check GenAI Platform mode
        genai_enabled = creds.get("genai_platform_enabled", False)
        if genai_enabled:
            return AzureOpenAIClient.from_model_config(
                deployment=deployment,
                endpoint=endpoint or "",
                api_version=api_version,
                genai_platform_enabled=True,
                genai_platform_base_url=creds.get("genai_platform_base_url"),
                genai_platform_path=creds.get("genai_platform_path"),
                genai_platform_user_id=creds.get("genai_platform_user_id"),
                genai_platform_project_name=creds.get("genai_platform_project_name"),
            )

        # Direct Azure OpenAI mode
        if not endpoint:
            raise ConfigurationError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT")

        return AzureOpenAIClient.from_model_config(
            deployment=deployment,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    @classmethod
    def _create_vertex_client(cls, model_info: ModelInfo, creds: dict) -> BaseLLMClient:
        """Create a Vertex AI client."""
        from services.llm_service.core.llm.vertex import VertexAIClient

        model_name = model_info.model_name
        project_id = creds.get("project_id")
        location = creds.get("location", "us-central1")

        # Check GenAI Platform mode
        genai_enabled = creds.get("genai_platform_enabled", False)
        if genai_enabled:
            return VertexAIClient.from_model_config(
                model=model_name,
                genai_platform_enabled=True,
                genai_platform_base_url=creds.get("genai_platform_base_url"),
                genai_platform_path=creds.get("genai_platform_path"),
                genai_platform_user_id=creds.get("genai_platform_user_id"),
                genai_platform_project_name=creds.get("genai_platform_project_name"),
            )

        # Direct Vertex AI mode requires project_id
        if not project_id:
            raise ConfigurationError(
                "Vertex AI requires either GENAI_PLATFORM_ENABLED=true "
                "or VERTEX_PROJECT_ID for direct access"
            )

        return VertexAIClient.from_model_config(
            model=model_name,
            project_id=project_id,
            location=location,
        )
