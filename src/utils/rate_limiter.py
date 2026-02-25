"""
Centralized rate-limiter / retry wrapper for all LLM API calls.
Handles Groq free-tier limits (30 RPM per model) with exponential backoff.
"""

from __future__ import annotations

import time
import logging
from functools import wraps
from threading import Lock
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# ── Per-model rate state ─────────────────────────────────────────────────────

_lock = Lock()
_last_call: dict[str, float] = {}  # model → last call timestamp

# Minimum seconds between calls per model (Groq free tier: 30 RPM → 2s)
MIN_INTERVAL_S = 2.5  # slightly over 2s for safety margin


def _wait_for_rate_limit(model: str) -> None:
    """Block until enough time has passed since the last call to this model."""
    with _lock:
        now = time.monotonic()
        last = _last_call.get(model, 0.0)
        elapsed = now - last
        if elapsed < MIN_INTERVAL_S:
            wait = MIN_INTERVAL_S - elapsed
            logger.debug("Rate limiter: waiting %.1fs for %s", wait, model)
            time.sleep(wait)
        _last_call[model] = time.monotonic()


def rate_limited_completion(
    *,
    model: str,
    messages: list[dict],
    max_retries: int = 5,
    **kwargs: Any,
) -> Any:
    """
    Wrapper around litellm.completion with:
    1. Pre-call rate limiting (respects MIN_INTERVAL_S per model)
    2. Exponential backoff on rate-limit (429) or server errors (5xx)
    
    Returns the litellm response object.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        _wait_for_rate_limit(model)

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                **kwargs,
            )
            return response

        except Exception as e:
            last_exc = e
            err_str = str(e).lower()

            # Detect rate-limit or transient server errors
            is_rate_limit = any(k in err_str for k in [
                "rate_limit", "rate limit", "429", "too many requests",
                "resource_exhausted", "quota",
            ])
            is_server_error = any(k in err_str for k in [
                "500", "502", "503", "504", "service unavailable",
                "internal server error", "overloaded",
            ])

            if is_rate_limit or is_server_error:
                backoff = min(60, (2 ** attempt) * 5)  # 5, 10, 20, 40, 60
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, str(e)[:120], backoff,
                )
                time.sleep(backoff)
            else:
                # Non-retriable error — raise immediately
                raise

    # All retries exhausted
    raise last_exc  # type: ignore[misc]
