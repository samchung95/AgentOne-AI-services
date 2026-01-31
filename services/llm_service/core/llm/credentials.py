"""Credential providers for LLM clients.

Defines the CredentialProvider protocol and implementations for different
authentication mechanisms (API keys, Azure AD tokens, GCP credentials).
"""

from __future__ import annotations

from typing import Any, Protocol


class CredentialProvider(Protocol):
    """Protocol for credential providers.

    Abstracts credential retrieval for LLM clients, allowing different
    authentication mechanisms to be used uniformly.
    """

    def get_credentials(self) -> dict[str, Any]:
        """Return credentials as a dictionary.

        Returns:
            Dictionary containing credential information.
            The structure depends on the credential type:
            - API key: {'api_key': key}
            - Azure AD: {'token': token, 'token_type': 'Bearer'}
            - GCP: {'credentials': credentials_object}
        """
        ...


class APIKeyCredentialProvider:
    """Credential provider for simple API key authentication.

    Wraps an API key in the CredentialProvider interface.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with an API key.

        Args:
            api_key: The API key for authentication.
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        self._api_key = api_key

    def get_credentials(self) -> dict[str, Any]:
        """Return credentials containing the API key.

        Returns:
            Dictionary with 'api_key' key containing the API key.
        """
        return {"api_key": self._api_key}
