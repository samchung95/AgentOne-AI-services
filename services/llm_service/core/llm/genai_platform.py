"""GenAI Platform integration module.

Provides shared functionality for Azure and Vertex AI clients when using
the GenAI Platform gateway.
"""

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
