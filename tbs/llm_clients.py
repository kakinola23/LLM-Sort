"""OpenAI clients, model table and list prices used in the paper.

Every client is wrapped in :class:`tbs.call_cache.CachedOpenAI`: when the ``TBS_CALL_CACHE``
environment variable names a ``.jsonl`` file, every reply is stored there and an interrupted run
resumes (replays) from it at no API cost.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .call_cache import CachedOpenAI

# name -> API model id (all OpenAI)
MODELS: List[Dict[str, str]] = [
    {"name": "GPT-4o", "provider": "openai", "model": "gpt-4o"},
    {"name": "GPT-4.1", "provider": "openai", "model": "gpt-4.1"},
    {"name": "GPT-4o-mini", "provider": "openai", "model": "gpt-4o-mini"},
    {"name": "GPT-5-nano", "provider": "openai", "model": "gpt-5-nano"},
]

# USD per 1M (input, output) tokens, list prices used for every cost column in the paper.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
}


def load_env(path: str | os.PathLike = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_model(name: str) -> Dict[str, str]:
    for m in MODELS:
        if m["name"].lower() == name.lower():
            return m
    raise ValueError(f"Unknown model {name!r}. Choose from: {[m['name'] for m in MODELS]}")


def make_client(name: str) -> Tuple[Any, str]:
    """Return (cached OpenAI client, model id) for a model name such as 'GPT-4o'."""
    from openai import OpenAI

    load_env()
    cfg = resolve_model(name)
    return CachedOpenAI(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))), cfg["model"]


def make_client_for_id(model_id: str) -> Tuple[Any, str]:
    """(cached OpenAI client, model_id) for an exact model id, e.g. 'gpt-5-nano-2025-08-07'."""
    from openai import OpenAI

    load_env()
    return CachedOpenAI(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))), model_id


def make_async_client() -> Any:
    """Async OpenAI client (used by the vendored OrderByLLM baselines)."""
    from openai import AsyncOpenAI

    load_env()
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES[model_id.split("-20")[0] if model_id not in PRICES else model_id]
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout
