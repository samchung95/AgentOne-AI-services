"""Model configuration loader.

Loads model definitions from config/models.yaml and provides
utilities for selecting models by capability.

Supports profile-based configuration with layered YAML files:
- config/models.yaml               # Base configuration (all providers defined)
- config/models.{profile}.yaml     # Profile-specific overrides (openrouter, azure, etc.)
- config/models.local.yaml         # Local overrides (gitignored)

Selection via CONFIG_PROFILE environment variable (default: "openrouter")
"""

import copy
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()


class ModelCapability(str, Enum):
    """Model capabilities."""

    CHAT = "chat"
    VISION = "vision"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    """Information about a specific model."""

    id: str  # Full model ID (e.g., "openrouter/openai/gpt-4o")
    provider: str  # Provider ID (e.g., "openrouter")
    model_name: str  # Model name within provider (e.g., "openai/gpt-4o")
    display_name: str
    capabilities: list[ModelCapability]
    context_window: int = 128000
    max_output: int = 4096
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    deployment: str | None = None  # For Azure OpenAI
    extra: dict = field(default_factory=dict)

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities


@dataclass
class ProviderInfo:
    """Information about a provider."""

    id: str
    name: str
    base_url: str | None = None
    api_version: str | None = None
    models: dict[str, ModelInfo] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Complete model configuration."""

    providers: dict[str, ProviderInfo]
    defaults: dict[str, str]  # use_case -> model_id

    def get_default_model(self, use_case: str) -> ModelInfo | None:
        """Get the default model for a use case (chat, vision, embedding)."""
        model_id = self.defaults.get(use_case)
        if not model_id:
            return None
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get a model by its full ID (provider/model_name)."""
        parts = model_id.split("/", 1)
        if len(parts) < 2:
            return None
        provider_id = parts[0]
        model_name = parts[1]

        provider = self.providers.get(provider_id)
        if not provider:
            return None
        return provider.models.get(model_name)

    def get_provider(self, provider_id: str) -> ProviderInfo | None:
        """Get provider info by ID."""
        return self.providers.get(provider_id)

    def find_models_with_capability(
        self, capability: ModelCapability, provider_id: str | None = None
    ) -> list[ModelInfo]:
        """Find all models with a specific capability."""
        models = []
        for pid, provider in self.providers.items():
            if provider_id and pid != provider_id:
                continue
            for model in provider.models.values():
                if model.has_capability(capability):
                    models.append(model)
        return models


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with overlay taking precedence."""
    result = copy.deepcopy(base)

    for key, value in overlay.items():
        if value is None:
            continue

        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _find_config_dir() -> Path | None:
    """Find the config directory."""
    search_paths = [
        Path("/app/config"),  # Docker container
        Path(__file__).parent.parent.parent.parent.parent / "config",  # Local dev
        Path.cwd() / "config",  # Current directory
    ]

    for config_dir in search_paths:
        if config_dir.exists() and config_dir.is_dir():
            return config_dir

    return None


def _load_layered_config(config_profile: str) -> dict[str, Any]:
    """Load configuration with profile-based layering."""
    config_dir = _find_config_dir()
    if not config_dir:
        logger.warning("config_dir_not_found", searched_paths=["Docker", "local dev", "cwd"])
        return {}

    merged_config: dict[str, Any] = {}

    # Layer 1: Base config
    base_path = config_dir / "models.yaml"
    if base_path.exists():
        merged_config = _load_yaml_config(base_path)
        logger.debug("loaded_base_config", path=str(base_path))

    # Layer 2: Profile-specific config
    profile_path = config_dir / f"models.{config_profile}.yaml"
    if profile_path.exists():
        profile_config = _load_yaml_config(profile_path)
        merged_config = _deep_merge(merged_config, profile_config)
        logger.info("loaded_profile_config", profile=config_profile, path=str(profile_path))
    else:
        logger.debug("profile_config_not_found", profile=config_profile, path=str(profile_path))

    # Layer 3: Local overrides (gitignored)
    local_path = config_dir / "models.local.yaml"
    if local_path.exists():
        local_config = _load_yaml_config(local_path)
        merged_config = _deep_merge(merged_config, local_config)
        logger.info("loaded_local_config", path=str(local_path))

    return merged_config


def _parse_model_config(raw: dict[str, Any]) -> ModelConfig:
    """Parse raw YAML config into ModelConfig."""
    providers: dict[str, ProviderInfo] = {}

    for provider_id, provider_data in raw.get("providers", {}).items():
        models: dict[str, ModelInfo] = {}

        for model_name, model_data in provider_data.get("models", {}).items():
            capabilities = [ModelCapability(c) for c in model_data.get("capabilities", [])]

            full_id = f"{provider_id}/{model_name}"
            models[model_name] = ModelInfo(
                id=full_id,
                provider=provider_id,
                model_name=model_name,
                display_name=model_data.get("name", model_name),
                capabilities=capabilities,
                context_window=model_data.get("context_window", 128000),
                max_output=model_data.get("max_output", 4096),
                cost_per_1m_input=model_data.get("cost_per_1m_input", 0.0),
                cost_per_1m_output=model_data.get("cost_per_1m_output", 0.0),
                deployment=model_data.get("deployment"),
            )

        providers[provider_id] = ProviderInfo(
            id=provider_id,
            name=provider_data.get("name", provider_id),
            base_url=provider_data.get("base_url"),
            api_version=provider_data.get("api_version"),
            models=models,
        )

    return ModelConfig(
        providers=providers,
        defaults=raw.get("defaults", {}),
    )


def _get_config_profile() -> str:
    """Get the configuration profile from settings or environment variable."""
    profile_value = os.getenv("CONFIG_PROFILE")
    if profile_value:
        return profile_value

    try:
        from services.llm_service.core.config.settings import get_settings

        return get_settings().config_profile
    except Exception:
        return "openrouter"


@lru_cache(maxsize=1)
def load_model_config() -> ModelConfig:
    """Load and cache model configuration with profile-based layering."""
    config_profile = _get_config_profile()
    logger.info("loading_model_config", config_profile=config_profile)

    raw = _load_layered_config(config_profile)

    if raw:
        return _parse_model_config(raw)

    # Return minimal default config if no files found
    logger.warning("using_default_model_config", reason="no config files found")
    return ModelConfig(
        providers={
            "openrouter": ProviderInfo(
                id="openrouter",
                name="OpenRouter",
                base_url="https://openrouter.ai/api/v1",
                models={
                    "openai/gpt-4o": ModelInfo(
                        id="openrouter/openai/gpt-4o",
                        provider="openrouter",
                        model_name="openai/gpt-4o",
                        display_name="GPT-4o",
                        capabilities=[
                            ModelCapability.CHAT,
                            ModelCapability.VISION,
                            ModelCapability.STREAMING,
                            ModelCapability.TOOL_CALLING,
                        ],
                    ),
                },
            ),
        },
        defaults={
            "chat": "openrouter/openai/gpt-4o",
            "vision": "openrouter/openai/gpt-4o",
        },
    )


def reload_model_config() -> ModelConfig:
    """Force reload of model configuration (clears cache)."""
    load_model_config.cache_clear()
    return load_model_config()


def get_model_for_use_case(use_case: str) -> ModelInfo:
    """Get the configured model for a specific use case.

    Args:
        use_case: One of "chat", "vision", "embedding"

    Returns:
        ModelInfo for the default model

    Raises:
        ValueError: If no model configured for use case
    """
    config = load_model_config()
    model = config.get_default_model(use_case)
    if not model:
        raise ValueError(f"No model configured for use case: {use_case}")
    return model


def get_provider_credentials(provider_id: str) -> dict[str, Any]:
    """Get credentials for a provider from environment variables."""
    creds: dict[str, Any] = {}

    settings = None
    try:
        from services.llm_service.core.config.settings import get_settings

        settings = get_settings()
    except Exception:
        settings = None

    # GenAI Platform settings (shared by azure_openai and vertex_ai)
    genai_env = os.getenv("GENAI_PLATFORM_ENABLED")
    if genai_env is None and settings is not None:
        genai_enabled = bool(getattr(settings, "genai_platform_enabled", False))
    else:
        genai_enabled = (genai_env or "false").lower() == "true"
    if genai_enabled:
        creds["genai_platform_enabled"] = True
        creds["genai_platform_base_url"] = os.getenv("GENAI_PLATFORM_BASE_URL") or getattr(
            settings, "genai_platform_base_url", ""
        )
        creds["genai_platform_path"] = os.getenv("GENAI_PLATFORM_PATH") or getattr(
            settings, "genai_platform_path", "stg/v1"
        )
        creds["genai_platform_user_id"] = os.getenv("GENAI_PLATFORM_USER_ID") or getattr(
            settings, "genai_platform_user_id", ""
        )
        creds["genai_platform_project_name"] = os.getenv("GENAI_PLATFORM_PROJECT_NAME") or getattr(
            settings, "genai_platform_project_name", ""
        )
    else:
        creds["genai_platform_enabled"] = False

    if provider_id == "openrouter":
        creds["api_key"] = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "openrouter_api_key", "")
        creds["site_url"] = os.getenv("OPENROUTER_SITE_URL") or getattr(settings, "openrouter_site_url", "")
        creds["app_name"] = os.getenv("OPENROUTER_APP_NAME") or getattr(settings, "openrouter_app_name", "AgentOne")
    elif provider_id == "openai":
        creds["api_key"] = os.getenv("OPENAI_API_KEY") or getattr(settings, "openai_api_key", "")
    elif provider_id == "azure_openai":
        creds["api_key"] = os.getenv("AZURE_OPENAI_API_KEY") or getattr(settings, "azure_openai_api_key", "")
        creds["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT") or getattr(settings, "azure_openai_endpoint", "")
        creds["deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT") or getattr(settings, "azure_openai_deployment", "")
        creds["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION") or getattr(
            settings, "azure_openai_api_version", "2024-08-01-preview"
        )
    elif provider_id == "vertex_ai":
        creds["api_key"] = os.getenv("VERTEX_API_KEY") or getattr(settings, "vertex_api_key", "")
        creds["project_id"] = os.getenv("VERTEX_PROJECT_ID") or getattr(settings, "vertex_project_id", "")
        creds["location"] = os.getenv("VERTEX_LOCATION") or getattr(settings, "vertex_location", "us-central1")
        creds["model"] = os.getenv("VERTEX_MODEL") or getattr(settings, "vertex_model", "gemini-2.5-flash-lite")
    elif provider_id == "anthropic":
        creds["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")

    return creds
