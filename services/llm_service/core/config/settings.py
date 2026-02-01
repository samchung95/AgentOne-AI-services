"""LLM Service configuration using Pydantic Settings.

Environment variables are loaded from .env file or system environment.

Configuration is driven by CONFIG_PROFILE environment variable which
selects a YAML configuration file (config/models.{profile}.yaml).
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"
_LOAD_ENV_FILE = os.getenv("LLM_SERVICE_LOAD_ENV_FILE", "true").lower() not in {"0", "false", "no", "off"}
_ENV_FILE = str(_DEFAULT_ENV_FILE) if _LOAD_ENV_FILE else None


class OpenAISettings(BaseSettings):
    """OpenAI provider settings.

    Environment variables are prefixed with OPENAI_ (e.g., OPENAI_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str | None = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4-turbo", description="Default OpenAI model name")
    base_url: str | None = Field(default=None, description="Custom OpenAI API base URL")


class OpenRouterSettings(BaseSettings):
    """OpenRouter provider settings.

    Environment variables are prefixed with OPENROUTER_ (e.g., OPENROUTER_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str | None = Field(default=None, description="OpenRouter API key")
    model: str = Field(default="openai/gpt-4-turbo", description="Default OpenRouter model name")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    site_url: str | None = Field(default=None, description="Site URL for OpenRouter headers")
    app_name: str | None = Field(default=None, description="Application name for OpenRouter headers")


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI provider settings.

    Environment variables are prefixed with AZURE_OPENAI_ (e.g., AZURE_OPENAI_ENDPOINT).
    """

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint URL")
    deployment: str = Field(default="gpt-4", description="Azure OpenAI deployment name")
    api_version: str = Field(default="2024-02-01", description="Azure OpenAI API version")
    api_key: str | None = Field(default=None, description="Azure OpenAI API key")


class VertexAISettings(BaseSettings):
    """Vertex AI / Gemini provider settings.

    Environment variables are prefixed with VERTEX_ (e.g., VERTEX_PROJECT_ID).
    """

    model_config = SettingsConfigDict(
        env_prefix="VERTEX_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_id: str | None = Field(default=None, description="Google Cloud project ID for Vertex AI")
    location: str = Field(default="us-central1", description="Vertex AI location/region")
    model: str = Field(default="gemini-2.5-flash-lite", description="Vertex AI model name")
    api_key: str | None = Field(default=None, description="API key for Gemini API")


class GenAIPlatformSettings(BaseSettings):
    """GenAI Platform gateway settings.

    Environment variables are prefixed with GENAI_PLATFORM_ (e.g., GENAI_PLATFORM_ENABLED).
    """

    model_config = SettingsConfigDict(
        env_prefix="GENAI_PLATFORM_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Use company GenAI Platform gateway for Azure OpenAI")
    base_url: str = Field(
        default="https://genai-platform-dev.pg.com", description="GenAI Platform base URL"
    )
    path: str = Field(default="stg/v1", description="GenAI Platform API path")
    user_id: str | None = Field(default=None, description="User ID for GenAI Platform")
    project_name: str | None = Field(default=None, description="Project name registered in GenAI Platform")


class TelemetrySettings(BaseSettings):
    """Telemetry and observability settings.

    Environment variables:
    - ENABLE_TELEMETRY: Enable/disable telemetry collection
    - APPINSIGHTS_CONNECTION_STRING: Azure Application Insights connection string
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        validation_alias="ENABLE_TELEMETRY",
        description="Enable telemetry collection",
    )
    appinsights_connection_string: str | None = Field(
        default=None,
        validation_alias="APPINSIGHTS_CONNECTION_STRING",
        description="Azure Application Insights connection string",
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "AgentOne LLM Service"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Configuration profile - single source of truth for LLM configuration
    # Determines which config/models.{profile}.yaml to load
    # Options: "openrouter" (default), "azure", or custom profile name
    config_profile: str = Field(
        default="openrouter",
        description="Configuration profile name (matches config/models.{profile}.yaml).",
    )

    # Azure OpenAI Configuration (nested settings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)

    # GenAI Platform Configuration (nested settings)
    genai_platform: GenAIPlatformSettings = Field(default_factory=GenAIPlatformSettings)

    # OpenAI Configuration (nested settings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)

    # OpenRouter Configuration (nested settings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)

    # Vertex AI / Gemini Configuration (nested settings)
    vertex_ai: VertexAISettings = Field(default_factory=VertexAISettings)

    # Telemetry (nested settings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    # Service Configuration
    llm_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
