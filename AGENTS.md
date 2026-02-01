# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn services.llm_service.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_mixin.py

# Run a single test
uv run pytest tests/test_mixin.py::TestConvertMessages::test_simple_message -v

# Lint and format
uv run ruff format .
uv run ruff check --fix .

# Type checking (if mypy is added)
uv run mypy services/
```

## Architecture Overview

This is a **multi-provider LLM gateway** that abstracts different LLM providers behind a unified API.

### Core Abstraction Pattern

```
BaseLLMClient (base.py)
    ↓ implements
OpenAICompatibleMixin (mixin.py) ──── shared message/tool conversion
    ↓ uses
├── OpenAIClient
├── OpenRouterClient
├── AzureOpenAIClient
└── VertexAIClient
    ↑ creates
LLMFactory (factory.py) ──── provider selection based on config
    ↑ uses
ModelConfig (models.py) ──── loads config/models.yaml + profile overrides
```

**Key types** (all in `base.py`):
- `LLMMessage` - Conversation message (supports multimodal: text + images)
- `LLMChunk` - Streaming response chunk
- `LLMResponse` - Complete response
- `LLMToolDefinition` - Tool/function definition

### Configuration System

Configuration is **profile-driven** via `CONFIG_PROFILE` env var:
1. Base config loaded from `config/models.yaml`
2. Profile overrides merged from `config/models.{profile}.yaml`
3. Local overrides from `config/models.local.yaml` (gitignored)

Provider selection: `defaults.chat` in YAML determines the default provider/model.

### Provider Credentials

Each provider uses different env vars:
- **OpenRouter**: `OPENROUTER_API_KEY`
- **OpenAI**: `OPENAI_API_KEY`
- **Azure OpenAI**: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
- **Vertex AI**: `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`
- **GenAI Platform** (enterprise gateway): `GENAI_PLATFORM_ENABLED=true`, `GENAI_PLATFORM_BASE_URL`

### Request Flow

```
POST /v1/generate
    → main.py (FastAPI routes)
    → LLMClientRegistry.get_client() (caches clients)
    → LLMFactory.create_for_model() (creates new or returns cached)
    → BaseLLMClient.generate() or generate_stream()
    → Provider-specific API call
```

### Shared Protocol

`shared/protocol/` contains models used across services:
- `common.py` - `Usage`, `ErrorInfo`, typed IDs
- `tool_models.py` - `ToolCall`, `ToolResult`, `DelegationResult`

### Tool Call ID Handling

Different providers use different ID formats. `normalize_tool_call_id()` in `shared/validators/id_generators.py` normalizes them to a consistent format prefixed with `tc_`.

## Testing

Tests use `asyncio_mode = "auto"` - async test functions work without decorators.

Set test environment variables in `conftest.py`:
```python
os.environ.setdefault("CONFIG_PROFILE", "openrouter")
os.environ.setdefault("LLM_SERVICE_LOAD_ENV_FILE", "false")
```

Integration tests in `tests/integration/` require real API keys.

## Key Patterns

**Mixin for code reuse**: `OpenAICompatibleMixin` provides `_convert_messages()`, `_convert_tools()`, and streaming aggregation shared by OpenAI-compatible clients.

**Lazy client initialization**: Clients use `_ensure_client()` pattern - actual SDK client created on first use.

**Streaming-first**: `generate()` is typically implemented by aggregating `generate_stream()` chunks via `_aggregate_stream_to_response()`.
