"""Application constants and configuration defaults for LLM Service.

This module centralizes all hardcoded values for the LLM service.
"""

from enum import Enum
from typing import Final

# =============================================================================
# Provider Identification
# =============================================================================


class ProviderID(str, Enum):
    """Enumeration of supported LLM providers.

    Using str as base allows direct comparison with strings and serialization.
    """

    OPENAI = "openai"
    OPENROUTER = "openrouter"
    AZURE_OPENAI = "azure_openai"
    VERTEX_AI = "vertex_ai"
    REMOTE = "remote"


# =============================================================================
# Retry Configuration
# =============================================================================

#: HTTP status codes that should trigger a retry
#: 429 = Rate Limited, 5xx = Server Errors
RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

#: Default maximum number of retry attempts
DEFAULT_MAX_RETRIES: Final[int] = 3

#: Default timeout for API calls (seconds)
#: Alias for DEFAULT_LLM_TIMEOUT_SECONDS for consistency with PRD naming
DEFAULT_TIMEOUT_SECONDS: Final[int] = 60

# =============================================================================
# LLM Dispatcher Configuration
# =============================================================================

#: Maximum concurrent requests per LLM provider
#: Prevents overwhelming provider rate limits
#: Valid range: 1-20
DEFAULT_MAX_CONCURRENT_LLM_REQUESTS: Final[int] = 5

#: Maximum requests that can be queued per provider when rate limited
#: Requests beyond this limit receive a 429 response
#: Valid range: 10-1000
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 100

#: Maximum rate limits before marking provider unhealthy
#: Used by LLM dispatcher for health tracking
#: Valid range: 3-50
MAX_RATE_LIMITS_PER_WINDOW: Final[int] = 10

#: Timeout for queued LLM requests (seconds)
#: Requests waiting longer are rejected
#: Valid range: 60-600
QUEUE_TIMEOUT_SECONDS: Final[float] = 300.0  # 5 minutes

#: Interval between provider health checks (seconds)
#: Valid range: 10-300
DISPATCHER_HEALTH_CHECK_INTERVAL_SECONDS: Final[float] = 60.0

#: Rate limit window size (seconds)
#: Valid range: 60-3600
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

#: Number of requests to track in circuit breaker sliding window
#: Used to calculate failure rate for circuit breaker decisions
#: Valid range: 10-1000
CIRCUIT_BREAKER_WINDOW_SIZE: Final[int] = 100

#: Failure rate threshold (0.0-1.0) to open the circuit
#: Circuit opens when failure rate exceeds this threshold
#: Valid range: 0.1-1.0
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[float] = 0.5  # 50%

#: Seconds to wait before transitioning from open to half-open
#: Allows testing if provider has recovered
#: Valid range: 5-300
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: Final[float] = 30.0

# =============================================================================
# Token Management
# =============================================================================

#: Refresh Azure tokens this many seconds before expiry
#: Ensures continuous authentication without interruptions
#: Valid range: 60-900 seconds
TOKEN_REFRESH_THRESHOLD_SECONDS: Final[int] = 600  # 10 minutes

# =============================================================================
# Streaming Retry Configuration
# =============================================================================

#: Maximum number of chunks to buffer for streaming retry/replay
#: Allows replaying buffered chunks on mid-stream failures
#: Valid range: 100-10000
DEFAULT_MAX_BUFFER_CHUNKS: Final[int] = 1000

# =============================================================================
# Timeout Configuration
# =============================================================================

#: Timeout for LLM API calls (seconds)
#: Should be long enough for complex reasoning
#: Valid range: 30-300 seconds
DEFAULT_LLM_TIMEOUT_SECONDS: Final[int] = 60
