"""Application constants and configuration defaults for LLM Service.

This module centralizes all hardcoded values for the LLM service.
"""

from typing import Final

# =============================================================================
# LLM Dispatcher Configuration
# =============================================================================

#: Maximum concurrent requests per LLM provider
#: Prevents overwhelming provider rate limits
#: Valid range: 1-20
DEFAULT_MAX_CONCURRENT_LLM_REQUESTS: Final[int] = 5

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
# Token Management
# =============================================================================

#: Refresh Azure tokens this many seconds before expiry
#: Ensures continuous authentication without interruptions
#: Valid range: 60-900 seconds
TOKEN_REFRESH_THRESHOLD_SECONDS: Final[int] = 600  # 10 minutes

# =============================================================================
# Timeout Configuration
# =============================================================================

#: Timeout for LLM API calls (seconds)
#: Should be long enough for complex reasoning
#: Valid range: 30-300 seconds
DEFAULT_LLM_TIMEOUT_SECONDS: Final[int] = 60
