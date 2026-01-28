"""Azure GenAI authentication helpers.

Provides Azure AD token management for:
1. Azure OpenAI via GenAI Platform gateway
2. Vertex AI via GenAI Platform gateway (using SPCredentials adapter)

SPCredentials adapts Azure AD tokens to the google.auth.credentials.Credentials
interface required by ChatVertexAI.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from google.auth.transport.requests import Request as GoogleAuthRequest

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    CredentialUnavailableError,
    DefaultAzureCredential,
    UsernamePasswordCredential,
)

from services.llm_service.core.config.constants import TOKEN_REFRESH_THRESHOLD_SECONDS

# Import google.auth.credentials for proper inheritance
try:
    from google.auth.credentials import Credentials as GoogleCredentials

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    # Fallback for environments without google-auth
    GoogleCredentials = object  # type: ignore[assignment,misc]
    GOOGLE_AUTH_AVAILABLE = False

logger = logging.getLogger("ai_chat.genai")

# Azure Cognitive Services scope for GenAI Platform gateway
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class _Token(Protocol):
    token: str
    expires_on: int


class _Credential(Protocol):
    def get_token(self, *scopes: str) -> _Token:  # pragma: no cover - protocol stub
        """Return an Azure access token for the provided scopes."""


class TokenProvider(Protocol):
    """Protocol describing the minimal interface for GenAI token providers."""

    def token(self) -> str:  # pragma: no cover - protocol stub
        """Return an access token for Azure AD secured resources."""


def _build_default_credentials() -> ChainedTokenCredential:
    """Construct the default Azure credential chain used for GenAI tokens."""

    username = os.getenv("AZURE_USERNAME", "").strip()
    password = os.getenv("AZURE_PASSWORD", "")
    tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
    client_id = os.getenv("AZURE_CLIENT_ID", "").strip()

    credential_chain: list[_Credential] = []

    if username and password and client_id:
        logger.info("Configuring Azure username/password credential for GenAI token acquisition")
        credential_chain.append(
            UsernamePasswordCredential(
                client_id=client_id,
                username=username,
                password=password,
                tenant_id=tenant_id or None,
            )
        )

    credential_chain.append(DefaultAzureCredential(exclude_interactive_browser_credential=False))
    credential_chain.append(AzureCliCredential())

    return ChainedTokenCredential(*credential_chain)


@dataclass
class GenAIToken(TokenProvider):
    """Provide Azure AD token management for GenAI service access.

    Supports configurable OAuth scope for different Azure services:
    - Default: Cognitive Services scope for Azure OpenAI/GenAI Platform
    - Can be customized for other Azure AD protected resources
    """

    refresh_threshold: int = TOKEN_REFRESH_THRESHOLD_SECONDS
    scope: str = COGNITIVE_SERVICES_SCOPE  # Configurable scope
    credential: _Credential | None = None

    _token: str = field(init=False, default="")
    _expires_on: int = field(init=False, default=0)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._refresh()

    @cached_property
    def _credentials(self) -> _Credential:
        """Return the credential chain used to request Azure AD tokens."""

        if self.credential is not None:
            return self.credential
        return _build_default_credentials()

    def _refresh(self) -> None:
        token = self._get_token()
        with self._lock:
            self._token, self._expires_on = token.token, token.expires_on
        logger.info("Refreshed Azure GenAI token expiring at %s", self._expires_on)

    def _get_token(self) -> _Token:
        return self._credentials.get_token(self.scope)

    def token(self) -> str:
        """Return a valid Azure AD token, refreshing if it is near expiry."""

        with self._lock:
            expires_on = self._expires_on
            token = self._token

        if expires_on < time.time() + self.refresh_threshold:
            self._refresh()
            with self._lock:
                return self._token
        return token

    def refresh_if_needed(self) -> None:
        """Refresh the token if it's near expiry. Used by SPCredentials."""
        with self._lock:
            expires_on = self._expires_on

        if expires_on < time.time() + self.refresh_threshold:
            self._refresh()

    def is_expired(self) -> bool:
        """Check if the current token is expired or near expiry."""
        with self._lock:
            expires_on = self._expires_on
        return expires_on < time.time() + self.refresh_threshold

    @property
    def expiry(self) -> datetime:
        """Return token expiry as datetime (UTC)."""
        with self._lock:
            return datetime.fromtimestamp(self._expires_on, tz=UTC)

    @property
    def current_token(self) -> str:
        """Return the current token value without refresh check."""
        with self._lock:
            return self._token


class _UnavailableTokenProvider(TokenProvider):
    """Token provider used when Azure credentials are unavailable."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def token(self) -> str:
        raise RuntimeError(
            f"Azure credentials are not configured; unable to obtain GenAI access token. Details: {self._reason}"
        )


_GLOBAL_TOKEN: TokenProvider | None = None
_GLOBAL_LOCK = threading.Lock()


def get_genai_token_provider() -> TokenProvider:
    """Return the shared GenAI token provider, initialising if required."""

    global _GLOBAL_TOKEN

    if _GLOBAL_TOKEN is not None:
        return _GLOBAL_TOKEN

    with _GLOBAL_LOCK:
        if _GLOBAL_TOKEN is None:
            _GLOBAL_TOKEN = GenAIToken()
        return _GLOBAL_TOKEN


def set_genai_token_provider(provider: TokenProvider | None) -> None:
    """Override the shared GenAI token provider (primarily for testing)."""

    global _GLOBAL_TOKEN
    with _GLOBAL_LOCK:
        _GLOBAL_TOKEN = provider


def prime_genai_token_provider() -> None:
    """Ensure the global GenAI token provider has an active access token."""
    try:
        provider = get_genai_token_provider()
    except (ClientAuthenticationError, CredentialUnavailableError) as exc:
        logger.warning(
            "Azure credentials unavailable; continuing with placeholder token provider.",
            exc_info=exc,
        )
        set_genai_token_provider(_UnavailableTokenProvider(str(exc)))
        return

    try:
        provider.token()
    except (ClientAuthenticationError, CredentialUnavailableError) as exc:
        logger.warning(
            "Failed to obtain Azure GenAI token; using placeholder provider until configured.",
            exc_info=exc,
        )
        set_genai_token_provider(_UnavailableTokenProvider(str(exc)))
    except Exception:  # pragma: no cover - defensive logging path
        logger.exception("Failed to prime the GenAI token provider")
        raise


class SPCredentials(GoogleCredentials):
    """Service Principal credentials for Vertex AI using GenAI tokens.

    This class properly inherits from google.auth.credentials.Credentials,
    adapting Azure AD tokens (via GenAIToken) to the interface required by
    ChatVertexAI and other Google Cloud clients.

    Use Case:
    - GenAI Platform mode: Routes Vertex AI requests through company's Azure
      GenAI Platform gateway, using Azure AD authentication instead of
      Google Cloud ADC.

    NOTE: This is ONLY for GenAI Platform mode (company gateway).
    Standard Vertex AI uses Google Cloud ADC via ChatVertexAI defaults.
    """

    def __init__(self, scope: str = COGNITIVE_SERVICES_SCOPE) -> None:
        """Initialize credentials using the global GenAI token provider.

        Args:
            scope: Azure AD scope for token acquisition.
                   Default: Cognitive Services scope for GenAI Platform.
        """
        super().__init__()
        self._scope = scope
        self._token_provider = get_genai_token_provider()
        self.token: str | None = None
        self.expiry: datetime | None = None
        # Initial token fetch
        self.refresh(None)

    def refresh(self, request: GoogleAuthRequest | None) -> None:
        """Refresh credentials using GenAIToken.

        Implements google.auth.credentials.Credentials.refresh().

        Args:
            request: Unused, but required by Google Auth interface.
                    The actual HTTP transport is handled by GenAIToken.
        """
        # Use the token provider to refresh and get current token
        self._token_provider.refresh_if_needed()
        self.token = self._token_provider.current_token
        self.expiry = self._token_provider.expiry

    @property
    def valid(self) -> bool:
        """Check if credentials are valid.

        Implements google.auth.credentials.Credentials.valid.
        """
        if not self.token:
            return False
        if self.expiry is None:
            return True
        # Compare with timezone-aware datetime
        now = datetime.now(tz=UTC)
        return self.expiry > now

    @property
    def expired(self) -> bool:
        """Check if credentials are expired.

        Implements google.auth.credentials.Credentials.expired.
        """
        return not self.valid

    def apply(self, headers: dict, token: str | None = None) -> None:
        """Apply the token to the authentication header.

        Implements google.auth.credentials.Credentials.apply().

        Args:
            headers: The HTTP request headers dictionary.
            token: Optional token override (usually not used).
        """
        token_to_use = token or self.token
        headers["authorization"] = f"Bearer {token_to_use}"

    def before_request(
        self,
        request: GoogleAuthRequest | None,
        method: str,
        url: str,
        headers: dict,
    ) -> None:
        """Perform credential refresh and apply headers before request.

        Implements google.auth.credentials.Credentials.before_request().

        Args:
            request: The HTTP transport request object (unused).
            method: The HTTP method (unused).
            url: The request URL (unused).
            headers: The HTTP request headers dictionary to modify.
        """
        if not self.valid:
            self.refresh(request)
        self.apply(headers)
