"""Tests for credential providers."""

import pytest

from services.llm_service.core.llm.credentials import APIKeyCredentialProvider


class TestAPIKeyCredentialProvider:
    """Tests for APIKeyCredentialProvider class."""

    def test_get_credentials_returns_api_key(self):
        """Test that get_credentials returns the API key in correct format."""
        provider = APIKeyCredentialProvider(api_key="test-api-key-123")
        credentials = provider.get_credentials()

        assert credentials == {"api_key": "test-api-key-123"}

    def test_get_credentials_with_different_key(self):
        """Test that provider correctly stores and returns different keys."""
        provider = APIKeyCredentialProvider(api_key="sk-different-key")
        credentials = provider.get_credentials()

        assert credentials["api_key"] == "sk-different-key"

    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            APIKeyCredentialProvider(api_key="")

    def test_provider_implements_protocol(self):
        """Test that APIKeyCredentialProvider satisfies CredentialProvider protocol."""
        provider = APIKeyCredentialProvider(api_key="test-key")

        # Verify it has the get_credentials method
        assert hasattr(provider, "get_credentials")
        assert callable(provider.get_credentials)

        # Verify return type matches protocol expectation
        credentials = provider.get_credentials()
        assert isinstance(credentials, dict)

    def test_credentials_are_not_mutated(self):
        """Test that returned credentials dict doesn't affect internal state."""
        provider = APIKeyCredentialProvider(api_key="original-key")
        credentials = provider.get_credentials()

        # Attempt to mutate the returned dict
        credentials["api_key"] = "modified-key"

        # Verify original is unchanged
        fresh_credentials = provider.get_credentials()
        assert fresh_credentials["api_key"] == "original-key"
