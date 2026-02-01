# PRD: Comprehensive Tech Debt Review

## Introduction

This PRD defines a comprehensive tech debt remediation effort for the AgentOne LLM Service. The codebase has strong fundamentals—clean architecture, excellent test coverage (4:1 test-to-code ratio), and proper design patterns—but has accumulated technical debt in three key areas: code duplication, configuration sprawl, and incomplete production features. This effort will reduce maintenance burden, improve extensibility, and ensure production readiness.

## Goals

- Eliminate repeated `_ensure_client()` patterns across 6 client implementations
- Consolidate GenAI Platform boilerplate into a reusable module
- Restructure 97 scattered environment variables into organized, discoverable configuration
- Integrate the existing (but unused) dispatcher for rate limiting and backpressure
- Improve streaming reliability with partial recovery support
- Achieve consistent behavior across all LLM providers through unified abstractions
- Maintain backward compatibility with existing API contracts

## User Stories

### Phase 1: Code Duplication - Client Initialization

#### US-001: Extract base client initialization pattern
**Description:** As a developer, I want client initialization logic centralized so that adding new providers doesn't require copy-pasting boilerplate.

**Acceptance Criteria:**
- [ ] Create `_initialize_client()` template method in `BaseLLMClient`
- [ ] Define abstract hooks: `_create_client_instance()`, `_get_credentials()`, `_get_endpoint()`
- [ ] Migrate `openai_client.py` to use new pattern
- [ ] Migrate `openrouter_client.py` to use new pattern
- [ ] Migrate `azure_openai.py` to use new pattern
- [ ] Migrate `vertex.py` to use new pattern
- [ ] Migrate `remote_client.py` to use new pattern
- [ ] Remove duplicated `_ensure_client()` from all 6 files
- [ ] All existing unit tests pass
- [ ] Typecheck passes

#### US-002: Create credential provider abstraction
**Description:** As a developer, I want credential retrieval abstracted so that different auth mechanisms (API key, Azure AD, GCP) are handled uniformly.

**Acceptance Criteria:**
- [ ] Create `CredentialProvider` protocol/interface in `services/llm_service/core/llm/credentials.py`
- [ ] Implement `APIKeyCredentialProvider` for OpenAI/OpenRouter
- [ ] Implement `AzureADCredentialProvider` wrapping existing `azure_token.py` logic
- [ ] Implement `GCPCredentialProvider` for Vertex AI
- [ ] Each client uses appropriate credential provider via dependency injection
- [ ] Existing authentication flows unchanged
- [ ] All existing tests pass
- [ ] Typecheck passes

### Phase 2: Code Duplication - GenAI Platform

#### US-003: Extract GenAI Platform integration module
**Description:** As a developer, I want GenAI Platform logic in one place so that Azure and Vertex clients share the same gateway integration code.

**Acceptance Criteria:**
- [ ] Create `services/llm_service/core/llm/genai_platform.py`
- [ ] Extract header building logic: `build_genai_headers(user_id, project_name, token)`
- [ ] Extract endpoint resolution logic: `resolve_genai_endpoint(base_url, path, provider)`
- [ ] Extract token retrieval: `get_genai_token(credentials)`
- [ ] Refactor `azure_openai.py` to use new module
- [ ] Refactor `vertex.py` to use new module
- [ ] Remove duplicate GenAI Platform checks from both clients
- [ ] All existing tests pass
- [ ] Typecheck passes

#### US-004: Add GenAI Platform configuration validation
**Description:** As a developer, I want GenAI Platform config validated at startup so that missing credentials fail fast with clear error messages.

**Acceptance Criteria:**
- [ ] Add `validate_genai_config()` function to `genai_platform.py`
- [ ] Validate required fields when `GENAI_PLATFORM_ENABLED=true`
- [ ] Raise `ConfigurationError` with specific missing field names
- [ ] Call validation during settings initialization
- [ ] Add unit tests for validation logic
- [ ] Typecheck passes

### Phase 3: Configuration Restructuring

#### US-005: Organize settings into nested dataclasses
**Description:** As a developer, I want settings grouped by provider/concern so that configuration is discoverable and IDE-friendly.

**Acceptance Criteria:**
- [ ] Create `OpenAISettings` dataclass with all OpenAI-specific fields
- [ ] Create `OpenRouterSettings` dataclass with all OpenRouter-specific fields
- [ ] Create `AzureOpenAISettings` dataclass with all Azure-specific fields
- [ ] Create `VertexAISettings` dataclass with all Vertex-specific fields
- [ ] Create `GenAIPlatformSettings` dataclass with all gateway fields
- [ ] Create `TelemetrySettings` dataclass with observability fields
- [ ] Nest all dataclasses under main `Settings` class
- [ ] Maintain backward compatibility with existing env var names
- [ ] Update all client code to use new nested structure
- [ ] All existing tests pass
- [ ] Typecheck passes

#### US-006: Create settings documentation generator
**Description:** As a developer, I want auto-generated docs for all env vars so that configuration is always up-to-date and discoverable.

**Acceptance Criteria:**
- [ ] Create `scripts/generate_env_docs.py`
- [ ] Script introspects `Settings` class and nested dataclasses
- [ ] Generates markdown table with: variable name, type, default, description
- [ ] Groups variables by provider/concern
- [ ] Output written to `docs/configuration.md`
- [ ] Add script to Makefile/pyproject.toml commands
- [ ] Typecheck passes

#### US-007: Add constants module for magic strings
**Description:** As a developer, I want all magic strings in one place so that provider names and status codes are consistent and discoverable.

**Acceptance Criteria:**
- [ ] Create/extend `services/llm_service/core/config/constants.py`
- [ ] Add `ProviderID` enum: `OPENAI`, `OPENROUTER`, `AZURE_OPENAI`, `VERTEX_AI`, `REMOTE`
- [ ] Add `RetryableStatusCodes` frozenset: `{429, 500, 502, 503, 504}`
- [ ] Add `DefaultTimeouts` dataclass with connection, read, total timeouts
- [ ] Replace all magic strings in factory.py with `ProviderID` enum
- [ ] Replace status code literals in retry.py with `RetryableStatusCodes`
- [ ] All existing tests pass
- [ ] Typecheck passes

### Phase 4: Performance & Rate Limiting

#### US-008: Integrate dispatcher into main API
**Description:** As an operator, I want rate limiting active so that the service handles load spikes gracefully without overwhelming upstream providers.

**Acceptance Criteria:**
- [ ] Wire `LLMDispatcher` into `main.py` FastAPI app
- [ ] Configure dispatcher with settings from `Settings` class
- [ ] Route `/v1/chat/completions` through dispatcher
- [ ] Route `/v1/chat/completions/stream` through dispatcher
- [ ] Add health check endpoint that reports dispatcher status
- [ ] Add metrics endpoint for rate limit statistics
- [ ] Add integration test for rate limiting behavior
- [ ] Typecheck passes

#### US-009: Implement request queuing with backpressure
**Description:** As an operator, I want requests queued during rate limits so that clients receive retryable errors instead of immediate failures.

**Acceptance Criteria:**
- [ ] Add `max_queue_size` setting to dispatcher configuration
- [ ] Add `queue_timeout_seconds` setting
- [ ] Implement async queue in dispatcher for pending requests
- [ ] Return 429 with `Retry-After` header when queue is full
- [ ] Add queue depth to metrics endpoint
- [ ] Add unit tests for queue behavior
- [ ] Add integration test for backpressure scenario
- [ ] Typecheck passes

#### US-010: Add provider health tracking
**Description:** As an operator, I want automatic health tracking so that unhealthy providers are temporarily bypassed.

**Acceptance Criteria:**
- [ ] Extend `LLMDispatcher` with per-provider health state
- [ ] Track success/failure ratio over sliding window
- [ ] Implement circuit breaker pattern: closed → open → half-open
- [ ] Add `health_check_interval_seconds` setting
- [ ] Add health status to `/health` endpoint response
- [ ] Log provider state transitions with structured logging
- [ ] Add unit tests for circuit breaker state machine
- [ ] Typecheck passes

### Phase 5: Streaming Reliability

#### US-011: Add streaming checkpoint support
**Description:** As a developer, I want streaming to support checkpoints so that partial responses can be recovered on failure.

**Acceptance Criteria:**
- [ ] Create `StreamCheckpoint` dataclass: `chunks: list[LLMChunk]`, `last_token_index: int`
- [ ] Add `checkpoint_callback` parameter to `generate_stream()` signature
- [ ] Call checkpoint callback every N chunks (configurable)
- [ ] Store checkpoint in memory (not persistent)
- [ ] Add `resume_from_checkpoint()` method to base client
- [ ] Document limitation: checkpoint lost if process restarts
- [ ] Add unit tests for checkpoint creation
- [ ] Typecheck passes

#### US-012: Implement streaming retry with replay
**Description:** As a developer, I want failed streams to replay cached chunks so that clients receive complete responses even after transient failures.

**Acceptance Criteria:**
- [ ] Extend `retry_async_generator()` in `retry.py`
- [ ] Buffer yielded chunks during iteration
- [ ] On retryable failure, yield buffered chunks first, then continue from provider
- [ ] Add `max_buffer_chunks` setting to prevent memory issues
- [ ] Add `enable_stream_replay` setting (default: true)
- [ ] Add unit tests for replay behavior
- [ ] Add integration test for mid-stream failure recovery
- [ ] Typecheck passes

### Phase 6: Cross-Provider Consistency

#### US-013: Normalize tool call IDs without data loss
**Description:** As a developer, I want tool call IDs normalized while preserving original IDs so that debugging traces back to provider logs.

**Acceptance Criteria:**
- [ ] Extend `ToolCall` model with `provider_id: str | None` field
- [ ] Modify `normalize_tool_call_id()` to return tuple: `(normalized_id, original_id)`
- [ ] Store original ID in `provider_id` field
- [ ] Include `provider_id` in structured logs
- [ ] Update all tool call creation sites
- [ ] Add unit tests for ID preservation
- [ ] Typecheck passes

#### US-014: Create cross-provider test suite
**Description:** As a developer, I want parameterized tests across all providers so that behavior differences are caught automatically.

**Acceptance Criteria:**
- [ ] Create `tests/test_cross_provider.py`
- [ ] Use pytest parameterization for all provider types
- [ ] Test: basic completion returns valid response structure
- [ ] Test: streaming yields chunks then final response
- [ ] Test: tool calling produces valid tool call objects
- [ ] Test: error responses have consistent structure
- [ ] Test: multimodal content handled uniformly (where supported)
- [ ] Tests can run with mocked responses (unit) or real APIs (integration)
- [ ] Typecheck passes

### Phase 7: Testing Infrastructure

#### US-015: Add CI-compatible test configuration
**Description:** As a developer, I want tests to run in CI without real API keys so that PRs are validated automatically.

**Acceptance Criteria:**
- [ ] Create `tests/fixtures/` directory with mock response files
- [ ] Add `--mock-providers` pytest flag
- [ ] When flag set, all provider clients return mock responses
- [ ] Mock responses cover: success, streaming, tool calls, errors
- [ ] Update `conftest.py` with mock injection fixtures
- [ ] Document test modes in README
- [ ] Typecheck passes

#### US-016: Add integration test documentation
**Description:** As a developer, I want clear docs for running integration tests so that real provider testing is straightforward.

**Acceptance Criteria:**
- [ ] Create `tests/integration/README.md`
- [ ] Document required environment variables per provider
- [ ] Document which tests require which providers
- [ ] Add `pytest.mark.integration` marker to all integration tests
- [ ] Add `pytest.mark.provider_<name>` markers for selective runs
- [ ] Update pyproject.toml with marker definitions
- [ ] Add example commands for running specific provider tests

## Functional Requirements

- FR-1: All client implementations must use the base class initialization template
- FR-2: Credential retrieval must go through `CredentialProvider` abstraction
- FR-3: GenAI Platform integration must be consolidated in single module
- FR-4: Settings must be organized into nested, provider-specific dataclasses
- FR-5: All magic strings must be replaced with constants/enums
- FR-6: Dispatcher must be integrated into FastAPI request handling
- FR-7: Rate limiting must support configurable queue with backpressure
- FR-8: Provider health must be tracked with circuit breaker pattern
- FR-9: Streaming must support checkpoint-based partial recovery
- FR-10: Tool call IDs must preserve original provider IDs
- FR-11: Cross-provider behavior must be validated by parameterized tests
- FR-12: Tests must support both mocked (CI) and real (integration) modes

## Non-Goals

- No changes to the external API contract (`/v1/chat/completions` endpoints)
- No new LLM provider implementations in this effort
- No persistent storage for streaming checkpoints (memory only)
- No UI or dashboard for monitoring (metrics endpoint only)
- No changes to the YAML configuration file format
- No migration tooling for existing deployments (manual update acceptable)
- No performance benchmarking beyond basic sanity checks

## Design Considerations

### Code Organization
```
services/llm_service/core/llm/
├── base.py              # Extended with initialization template
├── credentials.py       # NEW: Credential provider abstraction
├── genai_platform.py    # NEW: Consolidated gateway logic
├── constants.py         # Extended with enums and constants
├── factory.py           # Uses ProviderID enum
├── dispatcher.py        # Integrated into main.py
├── retry.py             # Extended with stream replay
└── [client files]       # Simplified, use base patterns
```

### Settings Structure
```python
class Settings(BaseSettings):
    config_profile: str
    debug: bool
    log_level: str

    openai: OpenAISettings
    openrouter: OpenRouterSettings
    azure_openai: AzureOpenAISettings
    vertex_ai: VertexAISettings
    genai_platform: GenAIPlatformSettings
    telemetry: TelemetrySettings
```

### Backward Compatibility
- Environment variables keep same names (nested classes use `env_prefix`)
- API endpoints unchanged
- Response schemas unchanged
- Existing configuration files work without modification

## Technical Considerations

- **Python version**: Requires 3.12+ (already enforced)
- **Breaking changes**: Internal refactoring only; external API unchanged
- **Dependencies**: No new dependencies required
- **Migration**: Developers update imports; no runtime migration needed
- **Rollback**: Git revert possible at any phase boundary

## Success Metrics

- Code duplication reduced: `_ensure_client()` exists in 1 file (down from 6)
- GenAI Platform code: consolidated into 1 module (down from 2)
- Settings discoverability: all 97 env vars documented with types and defaults
- Rate limiting: dispatcher active on all endpoints with configurable limits
- Test reliability: all tests pass in CI without real API keys
- Provider consistency: cross-provider test suite covers all 5 providers

## Open Questions

1. Should streaming checkpoints be persisted to Redis for cross-process recovery?
2. What queue size and timeout defaults are appropriate for production?
3. Should circuit breaker thresholds be configurable per-provider?
4. Is there a preference for the credential provider pattern (protocol vs ABC)?
5. Should the settings documentation generator also produce `.env.example`?

## Implementation Order

Recommended implementation sequence to minimize conflicts:

1. **US-007** (constants) - Foundation for other changes
2. **US-001, US-002** (client init, credentials) - Core refactoring
3. **US-003, US-004** (GenAI Platform) - Depends on credentials
4. **US-005, US-006** (settings restructure) - Can parallelize
5. **US-008, US-009, US-010** (dispatcher integration) - Performance phase
6. **US-011, US-012** (streaming reliability) - Can parallelize with dispatcher
7. **US-013** (tool call IDs) - Small, independent
8. **US-014, US-015, US-016** (testing) - Final validation
