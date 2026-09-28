"""
In-memory sliding-window rate limiter for sensitive authentication endpoints.
Protects against brute-force password guessing, credential stuffing, and registration spam.
"""

import os
import time
import logging
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status

logger = logging.getLogger("nutrisense.rate_limiter")

# Rate limit configuration (per IP address per window)
RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
LOGIN_RATE_LIMIT: int = int(os.getenv("LOGIN_RATE_LIMIT", "60"))  # Max 60 attempts per minute
REGISTER_RATE_LIMIT: int = int(os.getenv("REGISTER_RATE_LIMIT", "60"))  # Max 60 attempts per minute
WINDOW_SECONDS: int = 60

# In-memory storage: action:ip -> list of timestamps
_request_history: Dict[str, List[float]] = defaultdict(list)


def reset_rate_limiter() -> None:
    """Clears all stored rate limit timestamps (used for test teardowns)."""
    global _request_history
    _request_history.clear()


def set_rate_limit_enabled(enabled: bool) -> None:
    """Toggles rate limiting on or off dynamically."""
    global RATE_LIMIT_ENABLED
    RATE_LIMIT_ENABLED = enabled


def get_client_ip(request: Request) -> str:
    """Extracts client IP address safely, checking X-Forwarded-For first if behind a proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the leftmost untrusted client IP
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def check_rate_limit(
    request: Request,
    action: str,
    max_requests: int = 10,
    window_seconds: int = 60
) -> None:
    """
    Checks and enforces the sliding-window rate limit for a given action and client IP.
    Raises HTTP 429 Too Many Requests if limit is breached.
    """
    if not RATE_LIMIT_ENABLED:
        return

    client_ip = get_client_ip(request)
    key = f"{action}:{client_ip}"
    now = time.time()
    window_start = now - window_seconds

    # Filter timestamps to within the current sliding window
    timestamps = [ts for ts in _request_history[key] if ts > window_start]

    if len(timestamps) >= max_requests:
        oldest_in_window = timestamps[0]
        retry_after = max(1, int(window_seconds - (now - oldest_in_window)))
        logger.warning(
            f"Rate limit exceeded for action='{action}' from ip='{client_ip}'. "
            f"Attempts: {len(timestamps)}/{max_requests} in {window_seconds}s. Retry-After: {retry_after}s."
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "TOO_MANY_REQUESTS",
                "message": f"Too many requests for {action}. Please try again in {retry_after} seconds.",
                "details": [f"Maximum {max_requests} requests allowed per {window_seconds} seconds."]
            },
            headers={"Retry-After": str(retry_after)}
        )

    # Append current timestamp and update history
    timestamps.append(now)
    _request_history[key] = timestamps
