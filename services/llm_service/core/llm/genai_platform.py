"""GenAI Platform integration module.

Provides shared functionality for Azure and Vertex AI clients when using
the GenAI Platform gateway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.llm_service.core.config.constants import ProviderID
from services.llm_service.core.llm.exceptions import ConfigurationError

if TYPE_CHECKING:
    from services.llm_service.core.config.settings import Settings

# =============================================================================
# Default Configuration
# =============================================================================

#: Default path for GenAI Platform endpoints when none is configured
DEFAULT_GENAI_PATH = "stg/v1"

# =============================================================================
# Endpoint Resolution
# =============================================================================


def resolve_genai_endpoint(
    base_url: str,
    path: str | None = None,
    provider: ProviderID | None = None,
) -> str:
    """Resolve the full GenAI Platform endpoint URL.

    Constructs the complete endpoint URL by joining the base URL and path,
    handling edge cases like missing slashes and empty paths. For Vertex AI,
    appends the /vertexai suffix as required by the GenAI Platform gateway.

    Args:
        base_url: The GenAI Platform base URL (e.g., "https://genai.example.com").
        path: The API path (e.g., "stg/v1" or "/stg/v1"). Defaults to "stg/v1" if None or empty.
        provider: The LLM provider for provider-specific path suffixes.
                  Vertex AI gets "/vertexai" appended.

    Returns:
        The fully resolved endpoint URL.

    Example:
        >>> resolve_genai_endpoint("https://genai.example.com", "stg/v1", ProviderID.AZURE_OPENAI)
        'https://genai.example.com/stg/v1'
        >>> resolve_genai_endpoint("https://genai.example.com/", "/stg/v1/", ProviderID.VERTEX_AI)
        'https://genai.example.com/stg/v1/vertexai'
    """
    # Normalize base URL - remove trailing slashes
    normalized_base = base_url.rstrip("/") if base_url else ""

    # Normalize path - use default if empty, strip leading/trailing slashes
    normalized_path = (path.strip("/") if path else "") or DEFAULT_GENAI_PATH

    # Build the endpoint URL
    endpoint = f"{normalized_base}/{normalized_path}"

    # Add provider-specific suffix for Vertex AI
    if provider == ProviderID.VERTEX_AI:
        endpoint = f"{endpoint}/vertexai"

    return endpoint


# =============================================================================
# Header Building
# =============================================================================


def build_genai_headers(
    user_id: str | None = None,
    project_name: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Build headers for GenAI Platform requests.

    Args:
        user_id: The user ID for the request (X-User-ID / userid header).
        project_name: The project name for the request (X-Project-Name / project-name header).
        token: Optional Bearer token for Authorization header.

    Returns:
        Dictionary of headers for the GenAI Platform request.
        Empty dict if no values are provided.

    Example:
        >>> headers = build_genai_headers("user123", "my-project", "bearer-token")
        >>> headers["Authorization"]
        'Bearer bearer-token'
        >>> headers["userid"]
        'user123'
        >>> headers["project-name"]
        'my-project'
    """
    headers: dict[str, str] = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if user_id:
        headers["userid"] = user_id

    if project_name:
        headers["project-name"] = project_name

    return headers


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_genai_config(settings: Settings) -> None:
    """Validate GenAI Platform configuration when enabled.

    Checks that all required fields are set when GenAI Platform is enabled.
    This should be called at application startup to fail fast on
    misconfiguration rather than at first request time.

    Args:
        settings: Application settings object to validate.

    Raises:
        ConfigurationError: If GenAI Platform is enabled but required
            fields are missing. The error includes a list of missing fields.

    Example:
        >>> from services.llm_service.core.config.settings import get_settings
        >>> validate_genai_config(get_settings())  # raises if misconfigured
    """
    # Skip validation if GenAI Platform is not enabled
    if not settings.genai_platform.enabled:
        return

    missing_fields: list[str] = []

    # Check required fields when GenAI Platform is enabled
    if not settings.genai_platform.base_url:
        missing_fields.append("GENAI_PLATFORM_BASE_URL")

    if not settings.genai_platform.user_id:
        missing_fields.append("GENAI_PLATFORM_USER_ID")

    if not settings.genai_platform.project_name:
        missing_fields.append("GENAI_PLATFORM_PROJECT_NAME")

    if missing_fields:
        raise ConfigurationError(
            f"GenAI Platform is enabled but missing required configuration: "
            f"{', '.join(missing_fields)}",
            details={"missing_fields": missing_fields},
        )
