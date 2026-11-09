"""Baseline: OrderByLLM listwise external merge sort (Zhao et al.)."""

from __future__ import annotations

import asyncio
import contextvars
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
import order_by.pair_comparison as zhao_pc  # noqa: E402
from order_by.pair_comparison import external_comparisons  # noqa: E402
from order_by.sorting import external_merge_sort  # noqa: E402

MERGE_BATCH_SIZE = 4
MAX_EXHAUSTED_ROUNDS = 5

# Per-unit counters (raw API requests, tokens, re-asks); set inside each query/user task.
UNIT_STATS: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar("unit_stats", default=None)


def new_unit_stats() -> Dict[str, int]:
    stats = {"api_attempts": 0, "api_requests": 0, "raw_prompt_tokens": 0,
             "raw_completion_tokens": 0, "exhausted_reasks": 0}
    UNIT_STATS.set(stats)
    return stats


class _CountingCompletions:
    """Wraps client.beta.chat.completions: caps in-flight requests and counts every request."""

    def __init__(self, inner: Any, max_inflight: int) -> None:
        self._inner = inner
        self._max_inflight = max_inflight
        self._sem: Optional[asyncio.Semaphore] = None

    async def parse(self, **kwargs: Any) -> Any:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_inflight)
        stats = UNIT_STATS.get()
        if stats is not None:
            stats["api_attempts"] += 1
        async with self._sem:
            response = await self._inner.parse(**kwargs)
        if stats is not None:
            stats["api_requests"] += 1
            usage = getattr(response, "usage", None)
            stats["raw_prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            stats["raw_completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return response


def counting_client(client: Any, max_inflight: int = 16) -> Any:
    completions = _CountingCompletions(client.beta.chat.completions, max_inflight)
    return SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)))


async def guarded_external_comparisons(data, *args, **kwargs):
    """Zhao's external_comparisons, never accepting the all-retries-failed input-order fallback."""
    key = lambda x: str(x[0]) if isinstance(x, tuple) else str(x)  # noqa: E731
    want = sorted(key(x) for x in data)
    for round_i in range(1, MAX_EXHAUSTED_ROUNDS + 1):
        out, num, in_t, out_t = await external_comparisons(data, *args, **kwargs)
        if sorted(key(x) for x in out) == want:
            return out, num, in_t, out_t
        stats = UNIT_STATS.get()
        if stats is not None:
            stats["exhausted_reasks"] += 1
        print(f"  batch exhausted Zhao's retries (round {round_i}); re-asking", flush=True)
        await asyncio.sleep(30.0 * round_i)
    raise RuntimeError(f"batch {want} failed {MAX_EXHAUSTED_ROUNDS} rounds of Zhao retries; halting")


def use_recall_schema() -> None:
    """Reply schema {heights: [{id, height_m}], sorted_list} instead of {Reasoning_steps, sorted_list}
    (used for GPT-4o-mini on NBA, as in TBS's 'recall' schema)."""
    from pydantic import BaseModel

    class RecalledHeight(BaseModel):
        id: int
        height_m: float

    class RecallComparison(BaseModel):
        heights: List[RecalledHeight]
        sorted_list: List[int]

    zhao_pc.ExternalComparisonReasoning = RecallComparison


def use_movie_id_labels() -> None:
    """Render keys as ``movie_id:<i>`` instead of Zhao's hard-coded ``passage_id:<i>`` (MovieLens)."""

    def create_numbered_movies(passages, usePID=False):
        if usePID:
            return "\n".join([f"movie_id:{pid}\n{p}\n\n" for pid, p in passages])
        return "\n".join([f"movie_id:{i+1}\n{p}\n\n" for i, p in enumerate(passages)])

    zhao_pc.create_numbered_passages = create_numbered_movies


def safe_prompt(template: str, **kwargs: str) -> str:
    """str.format that leaves unknown {placeholders} (e.g. {keys}) for Zhao's code to fill."""

    class _PartialFmt(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_PartialFmt(**kwargs))


def normalize_output(sorted_data: List[Any]) -> List[str]:
    if not sorted_data:
        return []
    if isinstance(sorted_data[0], tuple):
        return [str(x[0]) for x in sorted_data]
    return [str(x) for x in sorted_data]


async def merge_sort_items(
    items: Sequence[str], prompt: str, client: Any, model_id: str, *,
    batch_size: int = MERGE_BATCH_SIZE, guarded: bool = True, use_cache: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """Merge-sort plain items (NBA names, countries); returns (ascending order, counts)."""
    stats = new_unit_stats()
    t0 = time.time()
    sorted_data, calls, in_t, out_t = await external_merge_sort(
        list(items), guarded_external_comparisons if guarded else external_comparisons,
        batch_size, client, prompt, model_id, isPassage=False, useCache=use_cache)
    order = [str(x) for x in sorted_data]
    return order, {"calls": int(calls), "prompt_tokens": int(in_t), "completion_tokens": int(out_t),
                   **stats, "time_s": round(time.time() - t0, 1)}


async def merge_sort_passages(
    keyed_texts: Sequence[Tuple[str, str]], prompt: str, client: Any, model_id: str, *,
    batch_size: int = MERGE_BATCH_SIZE,
) -> Tuple[List[str], Dict[str, Any]]:
    """Merge-sort (id, text) candidates for a query; returns (best-first ids, counts)."""
    stats = new_unit_stats()
    t0 = time.time()
    worst_to_best, calls, in_t, out_t = await external_merge_sort(
        list(keyed_texts), guarded_external_comparisons, batch_size, client, prompt, model_id,
        isPassage=True)
    best_first = list(reversed(normalize_output(worst_to_best)))
    if sorted(best_first) != sorted(k for k, _ in keyed_texts):
        raise RuntimeError(f"merge output does not cover all {len(keyed_texts)} candidates")
    return best_first, {"calls": int(calls), "prompt_tokens": int(in_t), "completion_tokens": int(out_t),
                        **stats, "time_s": round(time.time() - t0, 1)}
