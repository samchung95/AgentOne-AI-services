"""Tests for GenAI Platform module."""

from services.llm_service.core.config.constants import ProviderID
from services.llm_service.core.llm.genai_platform import (
    DEFAULT_GENAI_PATH,
    build_genai_headers,
    resolve_genai_endpoint,
)


class TestBuildGenaiHeaders:
    """Tests for build_genai_headers function."""

    def test_returns_empty_dict_when_no_values_provided(self):
        """Test that empty dict is returned when no values are provided."""
        headers = build_genai_headers()

        assert headers == {}

    def test_includes_authorization_header_with_bearer_prefix(self):
        """Test that token is included with Bearer prefix in Authorization header."""
        headers = build_genai_headers(token="my-token-123")

        assert headers["Authorization"] == "Bearer my-token-123"

    def test_includes_userid_header(self):
        """Test that user_id is included in userid header."""
        headers = build_genai_headers(user_id="user123")

        assert headers["userid"] == "user123"

    def test_includes_project_name_header(self):
        """Test that project_name is included in project-name header."""
        headers = build_genai_headers(project_name="my-project")

        assert headers["project-name"] == "my-project"

    def test_includes_all_headers_when_all_values_provided(self):
        """Test that all headers are included when all values are provided."""
        headers = build_genai_headers(
            user_id="user123",
            project_name="my-project",
            token="bearer-token",
        )

        assert headers["Authorization"] == "Bearer bearer-token"
        assert headers["userid"] == "user123"
        assert headers["project-name"] == "my-project"
        assert len(headers) == 3

    def test_omits_authorization_when_token_is_none(self):
        """Test that Authorization header is omitted when token is None."""
        headers = build_genai_headers(user_id="user123", project_name="my-project")

        assert "Authorization" not in headers
        assert "userid" in headers
        assert "project-name" in headers

    def test_omits_userid_when_user_id_is_none(self):
        """Test that userid header is omitted when user_id is None."""
        headers = build_genai_headers(project_name="my-project", token="token")

        assert "userid" not in headers
        assert "project-name" in headers
        assert "Authorization" in headers

    def test_omits_project_name_when_project_name_is_none(self):
        """Test that project-name header is omitted when project_name is None."""
        headers = build_genai_headers(user_id="user123", token="token")

        assert "project-name" not in headers
        assert "userid" in headers
        assert "Authorization" in headers

    def test_handles_empty_string_values(self):
        """Test that empty strings are treated as falsy (headers omitted)."""
        headers = build_genai_headers(user_id="", project_name="", token="")

        assert headers == {}

    def test_returns_new_dict_each_call(self):
        """Test that each call returns a new dict instance."""
        headers1 = build_genai_headers(user_id="user1")
        headers2 = build_genai_headers(user_id="user2")

        assert headers1 is not headers2
        assert headers1["userid"] == "user1"
        assert headers2["userid"] == "user2"


class TestResolveGenaiEndpoint:
    """Tests for resolve_genai_endpoint function."""

    def test_basic_url_joining(self):
        """Test basic URL joining without trailing/leading slashes."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "stg/v1")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_removes_trailing_slash_from_base_url(self):
        """Test that trailing slash is removed from base URL."""
        endpoint = resolve_genai_endpoint("https://genai.example.com/", "stg/v1")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_removes_leading_slash_from_path(self):
        """Test that leading slash is removed from path."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "/stg/v1")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_handles_both_slashes(self):
        """Test URL joining when both base has trailing and path has leading slash."""
        endpoint = resolve_genai_endpoint("https://genai.example.com/", "/stg/v1")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_removes_trailing_slash_from_path(self):
        """Test that trailing slash is removed from path."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "stg/v1/")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_handles_all_slashes(self):
        """Test URL joining when both have trailing and leading slashes."""
        endpoint = resolve_genai_endpoint("https://genai.example.com/", "/stg/v1/")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_uses_default_path_when_path_is_none(self):
        """Test that default path is used when path is None."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", None)

        assert endpoint == f"https://genai.example.com/{DEFAULT_GENAI_PATH}"

    def test_uses_default_path_when_path_is_empty_string(self):
        """Test that default path is used when path is empty string."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "")

        assert endpoint == f"https://genai.example.com/{DEFAULT_GENAI_PATH}"

    def test_uses_default_path_when_path_is_only_slashes(self):
        """Test that default path is used when path is only slashes."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "///")

        assert endpoint == f"https://genai.example.com/{DEFAULT_GENAI_PATH}"

    def test_vertex_ai_appends_vertexai_suffix(self):
        """Test that Vertex AI provider gets /vertexai suffix appended."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com", "stg/v1", ProviderID.VERTEX_AI
        )

        assert endpoint == "https://genai.example.com/stg/v1/vertexai"

    def test_azure_openai_no_suffix(self):
        """Test that Azure OpenAI does not get any suffix appended."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com", "stg/v1", ProviderID.AZURE_OPENAI
        )

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_openai_no_suffix(self):
        """Test that OpenAI does not get any suffix appended."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com", "stg/v1", ProviderID.OPENAI
        )

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_none_provider_no_suffix(self):
        """Test that None provider does not get any suffix appended."""
        endpoint = resolve_genai_endpoint("https://genai.example.com", "stg/v1", None)

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_vertex_ai_with_slashes(self):
        """Test Vertex AI with trailing/leading slashes are handled correctly."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com/", "/stg/v1/", ProviderID.VERTEX_AI
        )

        assert endpoint == "https://genai.example.com/stg/v1/vertexai"

    def test_complex_path(self):
        """Test with multi-segment path."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com", "api/v2/llm/openai"
        )

        assert endpoint == "https://genai.example.com/api/v2/llm/openai"

    def test_handles_empty_base_url(self):
        """Test that empty base URL still produces valid path."""
        endpoint = resolve_genai_endpoint("", "stg/v1")

        assert endpoint == "/stg/v1"

    def test_multiple_trailing_slashes_on_base(self):
        """Test that multiple trailing slashes are removed from base URL."""
        endpoint = resolve_genai_endpoint("https://genai.example.com///", "stg/v1")

        assert endpoint == "https://genai.example.com/stg/v1"

    def test_vertex_ai_with_default_path(self):
        """Test that Vertex AI uses default path and appends suffix correctly."""
        endpoint = resolve_genai_endpoint(
            "https://genai.example.com", None, ProviderID.VERTEX_AI
        )

        assert endpoint == f"https://genai.example.com/{DEFAULT_GENAI_PATH}/vertexai"
