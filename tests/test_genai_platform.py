"""Tests for GenAI Platform module."""

from services.llm_service.core.llm.genai_platform import build_genai_headers


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
