# AgentOne AI Services

Multi-provider LLM gateway and AI services for the AgentOne platform.

## Services

### LLM Service

A FastAPI-based LLM gateway that provides:
- **Multi-provider support**: OpenAI, OpenRouter, Azure OpenAI, Vertex AI (Gemini)
- **Streaming responses**: NDJSON streaming for real-time text generation
- **Tool calling**: Function calling support across all providers
- **Rate limiting**: Built-in dispatcher with per-provider rate limit handling
- **Retry logic**: Exponential backoff with jitter for transient failures
- **Client caching**: Efficient connection reuse across requests

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
cd AgentOne-AI-services

# Install dependencies with uv
uv sync

# Copy environment template
cp .env.example .env

# Edit .env to set your API keys and configuration
```

### Configuration

Configuration is driven by the `CONFIG_PROFILE` environment variable, which selects a YAML config file.

```bash
# Option 1: OpenRouter (default - recommended for development)
CONFIG_PROFILE=openrouter
OPENROUTER_API_KEY=sk-or-...

# Option 2: OpenAI Direct
CONFIG_PROFILE=openai
OPENAI_API_KEY=sk-...

# Option 3: Azure OpenAI
CONFIG_PROFILE=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...

# Option 4: Azure GenAI Platform Gateway
CONFIG_PROFILE=azure
GENAI_PLATFORM_ENABLED=true
GENAI_PLATFORM_BASE_URL=https://genai.yourcompany.com
```

### Running the Service

```bash
# Run with uv
uv run uvicorn services.llm_service.main:app --host 0.0.0.0 --port 8003 --reload

# Or run directly
uv run python -m services.llm_service.main
```

### API Endpoints

- `GET /health` - Health check
- `POST /v1/generate` - Non-streaming generation
- `POST /v1/generate-stream` - Streaming generation (NDJSON)

### Example Request

```bash
# Non-streaming
curl -X POST http://localhost:8003/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# Streaming
curl -X POST http://localhost:8003/v1/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "temperature": 0.7
  }'
```

## Architecture

```
AgentOne-AI-services/
├── config/                     # Model configuration files
│   ├── models.yaml            # Base configuration (all providers)
│   ├── models.openrouter.yaml # OpenRouter profile
│   └── models.azure.yaml      # Azure profile
├── services/
│   └── llm_service/           # LLM Service
│       ├── main.py            # FastAPI application entry point
│       ├── api/               # HTTP API layer
│       │   ├── protocol.py    # Request/response models
│       │   └── registry.py    # Client caching
│       └── core/              # Core business logic
│           ├── config/        # Configuration management
│           │   ├── settings.py
│           │   ├── models.py
│           │   └── constants.py
│           └── llm/           # LLM client implementations
│               ├── base.py    # Abstract base class
│               ├── factory.py # Client factory
│               ├── openai_client.py
│               ├── openrouter_client.py
│               ├── azure_openai.py
│               ├── vertex.py
│               ├── dispatcher.py
│               └── retry.py
└── shared/                    # Shared protocol definitions
    ├── protocol/              # Pydantic models
    └── validators/            # ID generators
```

## Provider Support

| Provider | Streaming | Tool Calling | Vision | Notes |
|----------|-----------|--------------|--------|-------|
| OpenRouter | ✅ | ✅ | ✅ | Access to 100+ models |
| OpenAI | ✅ | ✅ | ✅ | Direct API access |
| Azure OpenAI | ✅ | ✅ | ✅ | Enterprise deployments |
| Vertex AI | ✅ | ✅ | ✅ | Gemini models |

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format .
uv run ruff check --fix .
```

## License

MIT
