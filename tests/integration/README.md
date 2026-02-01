# Integration Tests

This directory contains integration tests that make real API calls to LLM providers.

## Pytest Markers

The following markers are available for selectively running integration tests:

| Marker | Description |
|--------|-------------|
| `integration` | All integration tests (requires real API calls) |
| `provider_openai` | Tests requiring OpenAI API key |
| `provider_openrouter` | Tests requiring OpenRouter API key |
| `provider_azure` | Tests requiring Azure OpenAI credentials |
| `provider_vertex` | Tests requiring Vertex AI/GCP credentials |

## Running Tests

### Run all integration tests

```bash
CONFIG_PROFILE=test uv run pytest -m integration -v
```

### Run tests by provider

```bash
# OpenRouter tests
CONFIG_PROFILE=test uv run pytest -m provider_openrouter -v

# OpenAI tests
CONFIG_PROFILE=test uv run pytest -m provider_openai -v

# Azure OpenAI tests
CONFIG_PROFILE=test uv run pytest -m provider_azure -v

# Vertex AI tests
CONFIG_PROFILE=test uv run pytest -m provider_vertex -v
```

### Run specific test files

```bash
# All LLM provider tests
CONFIG_PROFILE=test uv run pytest tests/integration/test_llm_providers.py -v

# Vertex AI specific tests
CONFIG_PROFILE=test uv run pytest tests/integration/test_vertex_integration.py -v
```

### Run with verbose output

```bash
CONFIG_PROFILE=test uv run pytest tests/integration/ -v -s
```

## Required Environment Variables

### OpenRouter
- `OPENROUTER_API_KEY` - Your OpenRouter API key

### OpenAI Direct
- `OPENAI_API_KEY` - Your OpenAI API key

### Azure OpenAI
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY` - (optional if using DefaultAzureCredential)

### Vertex AI (Direct)
- `VERTEX_PROJECT_ID` - GCP project ID
- `VERTEX_LOCATION` - Region (default: us-central1)
- Uses Application Default Credentials (`gcloud auth application-default login`)

### GenAI Platform (for Azure OpenAI or Vertex AI)
- `GENAI_PLATFORM_ENABLED=true` - Enable GenAI Platform mode
- `GENAI_PLATFORM_BASE_URL` - Base URL of the GenAI gateway
- `GENAI_PLATFORM_PATH` - API path (default: stg/v1)
- `GENAI_PLATFORM_USER_ID` - User ID header (optional)
- `GENAI_PLATFORM_PROJECT_NAME` - Project name header (optional)

## Test Structure

- `test_llm_providers.py` - Comprehensive tests for all LLM providers
- `test_vertex_integration.py` - Detailed tests for Vertex AI client migration

## Skip Behavior

Tests are automatically skipped if required environment variables are not set.
This allows running a subset of tests based on available credentials.

## Adding New Integration Tests

1. Add the `@pytest.mark.integration` marker to mark tests as requiring real API calls
2. Add the appropriate provider marker (e.g., `@pytest.mark.provider_openai`)
3. Use `pytest.mark.skipif` decorators to skip tests when credentials are unavailable
4. Follow the existing patterns for fixture setup and teardown
