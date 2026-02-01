"""Tests for credential providers."""

from unittest.mock import Mock, patch

import pytest

from services.llm_service.core.llm.credentials import (
    APIKeyCredentialProvider,
    AzureADCredentialProvider,
    GCPCredentialProvider,
)


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


class TestAzureADCredentialProvider:
    """Tests for AzureADCredentialProvider class."""

    def test_get_credentials_returns_token_and_type(self):
        """Test that get_credentials returns token and token_type."""
        mock_token_provider = Mock()
        mock_token_provider.token.return_value = "mock-azure-ad-token-123"

        provider = AzureADCredentialProvider(token_provider=mock_token_provider)
        credentials = provider.get_credentials()

        assert credentials == {
            "token": "mock-azure-ad-token-123",
            "token_type": "Bearer",
        }
        mock_token_provider.token.assert_called_once()

    def test_get_credentials_calls_token_provider_each_time(self):
        """Test that get_credentials calls token provider on each call."""
        mock_token_provider = Mock()
        mock_token_provider.token.side_effect = ["token-1", "token-2", "token-3"]

        provider = AzureADCredentialProvider(token_provider=mock_token_provider)

        creds1 = provider.get_credentials()
        creds2 = provider.get_credentials()
        creds3 = provider.get_credentials()

        assert creds1["token"] == "token-1"
        assert creds2["token"] == "token-2"
        assert creds3["token"] == "token-3"
        assert mock_token_provider.token.call_count == 3

    def test_provider_implements_protocol(self):
        """Test that AzureADCredentialProvider satisfies CredentialProvider protocol."""
        mock_token_provider = Mock()
        mock_token_provider.token.return_value = "test-token"

        provider = AzureADCredentialProvider(token_provider=mock_token_provider)

        # Verify it has the get_credentials method
        assert hasattr(provider, "get_credentials")
        assert callable(provider.get_credentials)

        # Verify return type matches protocol expectation
        credentials = provider.get_credentials()
        assert isinstance(credentials, dict)

    def test_credentials_returns_fresh_dict(self):
        """Test that each get_credentials call returns a new dict."""
        mock_token_provider = Mock()
        mock_token_provider.token.return_value = "same-token"

        provider = AzureADCredentialProvider(token_provider=mock_token_provider)
        credentials1 = provider.get_credentials()
        credentials2 = provider.get_credentials()

        # Should be equal but not the same object
        assert credentials1 == credentials2
        assert credentials1 is not credentials2


class TestGCPCredentialProvider:
    """Tests for GCPCredentialProvider class."""

    def test_get_credentials_returns_credentials_object(self):
        """Test that get_credentials returns credentials in correct format."""
        mock_credentials = Mock()

        provider = GCPCredentialProvider(credentials=mock_credentials)
        credentials = provider.get_credentials()

        assert "credentials" in credentials
        assert credentials["credentials"] is mock_credentials

    def test_custom_scopes_are_stored(self):
        """Test that custom scopes are stored correctly."""
        mock_credentials = Mock()
        custom_scopes = ["https://www.googleapis.com/auth/bigquery"]

        provider = GCPCredentialProvider(credentials=mock_credentials, scopes=custom_scopes)

        assert provider._scopes == custom_scopes

    def test_default_scopes_are_applied(self):
        """Test that default cloud-platform scope is applied when not specified."""
        mock_credentials = Mock()

        provider = GCPCredentialProvider(credentials=mock_credentials)

        assert provider._scopes == ["https://www.googleapis.com/auth/cloud-platform"]

    @patch("services.llm_service.core.llm.credentials.google.auth.default")
    def test_uses_adc_when_no_credentials_provided(self, mock_default):
        """Test that Application Default Credentials are used when credentials=None."""
        mock_adc_credentials = Mock()
        mock_default.return_value = (mock_adc_credentials, "test-project")

        provider = GCPCredentialProvider()
        credentials = provider.get_credentials()

        mock_default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        assert credentials["credentials"] is mock_adc_credentials

    @patch("services.llm_service.core.llm.credentials.google.auth.default")
    def test_uses_adc_with_custom_scopes(self, mock_default):
        """Test that ADC is called with custom scopes when provided."""
        mock_adc_credentials = Mock()
        mock_default.return_value = (mock_adc_credentials, "test-project")
        custom_scopes = ["https://www.googleapis.com/auth/compute"]

        provider = GCPCredentialProvider(scopes=custom_scopes)
        credentials = provider.get_credentials()

        mock_default.assert_called_once_with(scopes=custom_scopes)
        assert credentials["credentials"] is mock_adc_credentials

    def test_provider_implements_protocol(self):
        """Test that GCPCredentialProvider satisfies CredentialProvider protocol."""
        mock_credentials = Mock()

        provider = GCPCredentialProvider(credentials=mock_credentials)

        # Verify it has the get_credentials method
        assert hasattr(provider, "get_credentials")
        assert callable(provider.get_credentials)

        # Verify return type matches protocol expectation
        credentials = provider.get_credentials()
        assert isinstance(credentials, dict)
