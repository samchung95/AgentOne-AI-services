"""Tests for configuration modules."""

import os

import pytest


class TestModelConfig:
    """Tests for model configuration loading."""

    def test_load_model_config(self):
        """Test loading model configuration."""
        from services.llm_service.core.config.models import load_model_config

        config = load_model_config()
        assert config is not None
        assert config.providers is not None
        assert config.defaults is not None

    def test_model_config_has_defaults(self):
        """Test that config has default models."""
        from services.llm_service.core.config.models import load_model_config

        config = load_model_config()
        assert "chat" in config.defaults
        assert "vision" in config.defaults

    def test_get_default_model(self):
        """Test getting default model for use case."""
        from services.llm_service.core.config.models import load_model_config

        config = load_model_config()
        chat_model = config.get_default_model("chat")
        assert chat_model is not None
        assert chat_model.provider is not None

    def test_get_model_by_id(self):
        """Test getting model by full ID."""
        from services.llm_service.core.config.models import load_model_config

        config = load_model_config()
        # Get the chat default and look it up by ID
        chat_model = config.get_default_model("chat")
        if chat_model:
            looked_up = config.get_model(chat_model.id)
            assert looked_up is not None
            assert looked_up.id == chat_model.id

    def test_get_model_for_use_case(self):
        """Test get_model_for_use_case helper."""
        from services.llm_service.core.config.models import get_model_for_use_case

        model = get_model_for_use_case("chat")
        assert model is not None
        assert model.provider is not None

    def test_get_model_for_invalid_use_case(self):
        """Test that invalid use case raises ValueError."""
        from services.llm_service.core.config.models import get_model_for_use_case

        with pytest.raises(ValueError, match="No model configured"):
            get_model_for_use_case("invalid_use_case")


class TestSettings:
    """Tests for application settings."""

    def test_settings_loads(self):
        """Test that settings load without error."""
        from services.llm_service.core.config.settings import Settings

        settings = Settings()
        assert settings is not None

    def test_settings_defaults(self):
        """Test default setting values."""
        from services.llm_service.core.config.settings import Settings

        settings = Settings()
        assert settings.config_profile == "openrouter"
        assert settings.debug is False
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        from services.llm_service.core.config.settings import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestProviderCredentials:
    """Tests for provider credential retrieval."""

    def test_get_openrouter_credentials(self, monkeypatch):
        """Test getting OpenRouter credentials."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        from services.llm_service.core.config.models import get_provider_credentials

        creds = get_provider_credentials("openrouter")
        assert creds["api_key"] == "test-key"

    def test_get_openai_credentials(self, monkeypatch):
        """Test getting OpenAI credentials."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from services.llm_service.core.config.models import get_provider_credentials

        creds = get_provider_credentials("openai")
        assert creds["api_key"] == "sk-test"

    def test_get_azure_credentials(self, monkeypatch):
        """Test getting Azure OpenAI credentials."""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")

        from services.llm_service.core.config.models import get_provider_credentials

        creds = get_provider_credentials("azure_openai")
        assert creds["endpoint"] == "https://test.openai.azure.com"
        assert creds["api_key"] == "azure-key"

    def test_genai_platform_credentials(self, monkeypatch):
        """Test getting GenAI Platform credentials."""
        monkeypatch.setenv("GENAI_PLATFORM_ENABLED", "true")
        monkeypatch.setenv("GENAI_PLATFORM_BASE_URL", "https://genai.example.com")
        monkeypatch.setenv("GENAI_PLATFORM_USER_ID", "user123")

        from services.llm_service.core.config.models import get_provider_credentials

        creds = get_provider_credentials("azure_openai")
        assert creds["genai_platform_enabled"] is True
        assert creds["genai_platform_base_url"] == "https://genai.example.com"
        assert creds["genai_platform_user_id"] == "user123"
