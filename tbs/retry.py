"""Retrying chat-completion call (HTTP 429 / 5xx / timeouts back off exponentially)."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "unavailable",
    "temporarily",
    "timeout",
    "timed out",
    "connection",
    "econnreset",
    "jsondecodeerror",
    "expecting ',' delimiter",
)


def _is_transient(err: Exception) -> bool:
    if isinstance(err, json.JSONDecodeError):
        return True
    s = str(err).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


def robust_chat_completion(
    client: Any,
    model: str,
    system_message: str,
    prompt: str,
    *,
    verbose: bool = False,
    max_retries: int = 6,
    backoff_base: float = 5.0,
    max_backoff: float = 60.0,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.0,
) -> Any:
    """Plain-text chat completion at temperature 0, retrying transient failures (incl. HTTP 429)
    with capped exponential backoff. Replies are cached by tbs.call_cache (TBS_CALL_CACHE)."""

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - we re-raise non-transient below
            last_error = e
            if attempt < max_retries and _is_transient(e):
                delay = min(backoff_base * (2 ** attempt), max_backoff)
                if verbose:
                    print(f"    Transient error (attempt {attempt + 1}/{max_retries}); "
                          f"backing off {delay:.0f}s: {e}")
                time.sleep(delay)
                continue
            raise
        else:
            return response

    if last_error is not None:
        raise last_error
    raise RuntimeError("robust_chat_completion exhausted retries without a response")
