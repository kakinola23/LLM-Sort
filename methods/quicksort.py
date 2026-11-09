"""Baseline: OrderByLLM pairwise quicksort, QS1 (one vote per comparison) and QS3 (majority of 3).

Calls Zhao's ``order_by.sorting.quick_sort`` unchanged. A query whose comparisons fell back to a
default answer while the API was rate-limiting is reported as failed, so it can be rerun.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
from order_by.sorting import quick_sort  # noqa: E402

from methods.merge_sort import normalize_output  # noqa: E402

UNIT: contextvars.ContextVar[str] = contextvars.ContextVar("unit", default="")
FALLBACKS: Dict[str, int] = {}
RATELIMITS: Dict[str, int] = {}


class _FallbackCounter(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        u = UNIT.get()
        if "retries exhausted" in msg:
            FALLBACKS[u] = FALLBACKS.get(u, 0) + 1
        elif "429" in msg:
            RATELIMITS[u] = RATELIMITS.get(u, 0) + 1


def install_fallback_counter() -> None:
    logging.getLogger("order_by.pair_comparison").addHandler(_FallbackCounter())


def limit_concurrency(aclient: Any, limit: int) -> Any:
    sem = asyncio.Semaphore(limit)
    completions = aclient.beta.chat.completions
    orig = completions.parse

    async def parse(*a, **kw):
        async with sem:
            return await orig(*a, **kw)

    completions.parse = parse
    return aclient


async def quicksort_passages(
    unit_id: str, keyed_texts: Sequence[Tuple[str, str]], prompt: str, client: Any, model_id: str, *,
    vote: int = 1, limit_k: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Quicksort (id, text) candidates; returns (best-first ids, counts incl. fallbacks)."""
    UNIT.set(unit_id)
    t0 = time.time()
    worst_to_best, calls, in_t, out_t = await quick_sort(
        list(keyed_texts), client, prompt, model_id, isPassage=True, vote=vote, limit_k=limit_k or None)
    best_first = list(reversed(normalize_output(worst_to_best)))
    return best_first, {"calls": calls, "prompt_tokens": in_t, "completion_tokens": out_t,
                        "fallbacks": FALLBACKS.get(unit_id, 0),
                        "rate_limited": bool(RATELIMITS.get(unit_id)),
                        "time_s": round(time.time() - t0, 1)}


async def quicksort_with_comparator(items: Sequence[str], compare, vote: int = 1) -> List[str]:
    """Plain recursive quicksort (first element as pivot) over an async ``compare(a, b) -> -1|1``.
    vote=3: each element-vs-pivot decision is the majority of three independent comparisons.
    Used for MovieLens, whose pairwise prompt is a plain-text preference question."""

    async def qsort(arr: List[str]) -> List[str]:
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left: List[str] = []
        right: List[str] = []
        for x in arr[1:]:
            if vote == 1:
                res = await compare(x, pivot)
            else:
                v = [await compare(x, pivot) for _ in range(vote)]
                res = -1 if sum(v) < 0 else 1
            (left if res == -1 else right).append(x)
        return (await qsort(left)) + [pivot] + (await qsort(right))

    return await qsort(list(items))
