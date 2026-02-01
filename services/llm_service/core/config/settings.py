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

    # Azure OpenAI Configuration
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str = "gpt-4"
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_api_key: str | None = None

    # GenAI Platform Configuration (Company Azure GenAI Gateway)
    genai_platform_enabled: bool = Field(
        default=False, description="Use company GenAI Platform gateway for Azure OpenAI"
    )
    genai_platform_base_url: str = Field(
        default="https://genai-platform-dev.pg.com", description="GenAI Platform base URL"
    )
    genai_platform_path: str = Field(default="stg/v1", description="GenAI Platform API path")
    genai_platform_user_id: str | None = Field(
        default=None, description="User ID for GenAI Platform"
    )
    genai_platform_project_name: str | None = Field(
        default=None, description="Project name registered in GenAI Platform"
    )

    # OpenAI Configuration (nested settings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)

    # OpenRouter Configuration
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4-turbo"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None

    # Vertex AI / Gemini Configuration
    vertex_project_id: str | None = Field(default=None, description="Google Cloud project ID for Vertex AI")
    vertex_location: str = Field(default="us-central1", description="Vertex AI location/region")
    vertex_model: str = Field(default="gemini-2.5-flash-lite", description="Vertex AI model name")
    vertex_api_key: str | None = Field(default=None, description="API key for Gemini API")

    # Telemetry
    appinsights_connection_string: str | None = None
    enable_telemetry: bool = True

    # Service Configuration
    llm_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
