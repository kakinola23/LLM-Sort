"""Baseline: OrderByLLM pointwise scoring (one relevance prompt per candidate), then sort by score.

Calls Zhao's ``order_by.sorting.pointwise_sort`` unchanged, except that a value whose retries all
failed (returned as 0 with no tokens) is re-asked instead of entering the ranking as score 0.
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
import order_by.pointwise as zhao_pointwise  # noqa: E402
from order_by.sorting import pointwise_sort  # noqa: E402

from methods.merge_sort import UNIT_STATS, new_unit_stats  # noqa: E402

MAX_POINTWISE_ROUNDS = 5


def fresh_cache() -> None:
    """Zhao's pointwise module keeps a disk/memory cache; clear it so every value is a fresh call."""
    zhao_pointwise.cache = {}


async def pointwise_passages(
    unit_id: str, keyed_texts: Sequence[Tuple[str, str]], prompt: str, client: Any, model_id: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """Score each (id, text) candidate; returns (best-first ids, counts and per-id scores)."""

    class GuardedRelevanceKey(zhao_pointwise.PointwiseRelevanceKey):
        async def value(self, client, modelname, output_type):
            for round_i in range(1, MAX_POINTWISE_ROUNDS + 1):
                v, n, in_t, out_t = await super().value(client, modelname, output_type)
                if in_t:
                    return v, n, in_t, out_t
                UNIT_STATS.get()["exhausted_reasks"] += 1
                print(f"  {unit_id}: pointwise value exhausted Zhao's retries (round {round_i}); re-asking", flush=True)
                await asyncio.sleep(30.0 * round_i)
            raise RuntimeError(f"pointwise value failed {MAX_POINTWISE_ROUNDS} rounds of Zhao retries")

    stats = new_unit_stats()
    t0 = time.time()
    doc_ids, scores, calls, in_tok, out_tok = await pointwise_sort(
        list(keyed_texts), client, prompt, model_id, float, key_class=GuardedRelevanceKey, isPassage=True)
    if sorted(map(str, doc_ids)) != sorted(k for k, _ in keyed_texts):
        raise RuntimeError(f"pointwise output does not cover all {len(keyed_texts)} candidates")
    # Zhao scores the run with trec_eval, which orders ties by docno descending; reproduce that order.
    score = {str(d): float(s) for d, s in zip(doc_ids, scores)}
    best_first = sorted(score, key=lambda d: (score[d], d), reverse=True)
    return best_first, {"calls": calls, "prompt_tokens": in_tok, "completion_tokens": out_tok,
                        **stats, "scores": score, "time_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Item datasets (NBA, Population): one OrderByLLM pointwise value prompt per item, then sort.


ZHAO_SYSTEM = "You are a helpful agent. Think step by step. Output a JSON object."
SCHEMA_MAX_ATTEMPTS = 5


class PointwiseReasoning(BaseModel):
    """OrderByLLM order_by/pointwise.py schema for item values."""
    key: str
    explanation: str
    value: float


def _score_item(client: Any, model_id: str, prompt: str) -> Tuple[float, int, int, int]:
    """Structured parse, up to 5 attempts; value 0 if all fail (OrderByLLM). Returns
    (value, attempts, prompt_tokens, completion_tokens)."""
    pt = ct = 0
    for attempt in range(1, SCHEMA_MAX_ATTEMPTS + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=model_id,
                messages=[{"role": "system", "content": ZHAO_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0, response_format=PointwiseReasoning, max_completion_tokens=8192)
            usage = getattr(response, "usage", None)
            pt += int(getattr(usage, "prompt_tokens", 0) or 0)
            ct += int(getattr(usage, "completion_tokens", 0) or 0)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("no parsed output")
            return float(parsed.value), attempt, pt, ct
        except Exception as e:  # noqa: BLE001
            print(f"  pointwise parse attempt {attempt}/{SCHEMA_MAX_ATTEMPTS} failed: {e}", flush=True)
            if "429" in str(e) or "rate limit" in str(e).lower():
                time.sleep(10.0)
    return 0.0, SCHEMA_MAX_ATTEMPTS, pt, ct


def pointwise_items(items: Sequence[str], prompt_template: str, client: Any, model_id: str, *,
                    workers: int = 16) -> Tuple[List[str], Dict[str, Any]]:
    """Ascending order by the model's own value for each item; also returns the values."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(lambda k: _score_item(client, model_id, prompt_template.format(key=k)), items))
    values = {k: r[0] for k, r in zip(items, res)}
    order = [k for _, k in sorted(((v, k) for k, v in values.items()))]
    return order, {"calls": len(items), "api_requests": sum(r[1] for r in res),
                   "prompt_tokens": sum(r[2] for r in res), "completion_tokens": sum(r[3] for r in res),
                   "values": values, "time_s": round(time.time() - t0, 1)}
