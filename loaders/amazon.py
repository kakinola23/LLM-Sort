"""Amazon Reviews 2023 products ranked by price (up to n = 1M).

Ground truth ``data/amazon/amazon_1m_price_gt.csv`` (price ascending, ties by title then ASIN) is
built from the McAuley Lab metadata dump by ``scripts/fetch_amazon_priced.py`` and
``scripts/build_amazon_gt.py`` (see data/amazon/README.md). The paper's scaling runs show the price
next to each title ("explicit" prompt); the "plain"/"rescued" prompts show titles only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from scipy.stats import kendalltau

from tbs.sorter import BatchedLLMSorter, _supports_temperature_zero

ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "amazon" / "amazon_1m_price_gt.csv"

AMAZON_PRICE_PROMPT = """Given a list of Amazon products with id at the front:
{keys}
Based on your own knowledge, sort the products in ascending order based on their price on Amazon (in US dollars).
Break ties based on the product title in alphabetical order.
Output a JSON list of integer ids, representing the products from cheapest to most expensive.
"""

# Amazon analogue of NBA_KNOWLEDGE_COMPARISON_PROMPT_RESCUED: anchor each product to a stated
# price estimate before sorting, spell out the tie rule, and ban omissions/duplicates.
AMAZON_PRICE_PROMPT_RESCUED = """You will sort Amazon products by price, cheapest to most expensive, using your own knowledge of what each product typically sells for on Amazon US.

Products (id: title):
{keys}
Do this internally, then output only the final JSON:
1. For each id, estimate that product's typical Amazon US price in dollars, rounded to exactly two decimal places. Use the same value you would give if asked about that product alone — do not nudge it up or down just to create or avoid a tie with a neighboring product.
2. Sort all ids in ascending order by that price (cheapest first).
3. If two or more products land on the exact same price, break the tie by comparing their titles alphabetically, A to Z, case-insensitive, comparing the whole string exactly as given (including punctuation and spaces).
4. Every id from the list above must appear in "order" exactly once — no omissions, no duplicates, no ids that were not given to you.

Respond with ONLY this JSON object and nothing else — no markdown code fences, no commentary before or after it:
{{"prices": {{"<id>": <price_usd_2dp>, ...}}, "order": [<id>, <id>, ...]}}
"""

# v2: adds a dataset-wide price-range anchor, a price-signal checklist (pack/size/material/
# brand), and a self-consistency check that "order" actually matches sorted "prices".
AMAZON_PRICE_PROMPT_RESCUED_V2 = """You will sort Amazon products by price, cheapest to most expensive, using your own knowledge of what each product typically sells for on Amazon US.

Typical prices across this catalog range from about $0.50 to $10,000, so calibrate estimates to that full range rather than clustering everything near the middle.

Products (id: title):
{keys}
Do this internally, then output only the final JSON:
1. For each id, read the title for price-driving signals - pack/quantity count, size or volume, material/quality words, brand positioning, product category - then estimate that product's typical Amazon US price in dollars, rounded to exactly two decimal places. Use the same value you would give if asked about that product alone - do not nudge it up or down just to create or avoid a tie with a neighboring product.
2. Sort all ids in ascending order by that price (cheapest first).
3. If two or more products land on the exact same price, break the tie by comparing their titles alphabetically, A to Z, case-insensitive, comparing the whole string exactly as given (including punctuation and spaces).
4. Every id from the list above must appear in "order" exactly once - no omissions, no duplicates, no ids that were not given to you.
5. Before responding, check that "order" is exactly the ids from "prices" sorted ascending by price with the tie rule applied - fix "order" if it doesn't match.

Respond with ONLY this JSON object and nothing else - no markdown code fences, no commentary before or after it:
{{"prices": {{"<id>": <price_usd_2dp>, ...}}, "order": [<id>, <id>, ...]}}
"""

# Explicit: each product line carries its actual listed price ("<title> — $12.99",
# see explicit_display), so the LLM orders given values instead of recalling them.
AMAZON_PRICE_PROMPT_EXPLICIT = """Given a list of Amazon products with id and price:
{keys}
Sort the products in ascending order by the price shown.
Break ties based on the product title in alphabetical order.
Output a JSON list of integer ids, representing the products from cheapest to most expensive.
"""


# Explicit, but the model must copy every shown price into its answer, so it
# writes one entry per product (the order-only format lets it skip items).
AMAZON_PRICE_PROMPT_EXPLICIT_ECHO = """Given a list of Amazon products with id and price:
{keys}
Sort the products in ascending order by the price shown.
Break ties based on the product title in alphabetical order.
Every id from the list above must appear in "order" exactly once - no omissions, no duplicates.

Respond with ONLY this JSON object and nothing else:
{{"prices": {{"<id>": <price exactly as shown>, ...}}, "order": [<id>, <id>, ...]}}
"""


def explicit_display(title: str, price: float) -> str:
    return f"{title} — ${price:.2f}"


PROMPTS = {
    "explicit": AMAZON_PRICE_PROMPT_EXPLICIT,
    "explicit_echo": AMAZON_PRICE_PROMPT_EXPLICIT_ECHO,
    "plain": AMAZON_PRICE_PROMPT,
    "rescued": AMAZON_PRICE_PROMPT_RESCUED,
    "rescued_v2": AMAZON_PRICE_PROMPT_RESCUED_V2,
}


class AmazonPriceSorter(BatchedLLMSorter):
    """BatchedLLMSorter with a title-only price prompt; items are parent_asin keys."""

    prompt_template = AMAZON_PRICE_PROMPT

    def _sort_with_llm_once(
        self,
        presentation: Sequence[str],
        round_id: int,
        payload: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        keys = "\n".join(f"{i}: {self.text_map[a]}" for i, a in enumerate(presentation, 1))
        create_kw: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
                {"role": "user", "content": self.prompt_template.format(keys=keys)},
            ],
        }
        if _supports_temperature_zero(self.model):
            create_kw["temperature"] = 0
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(**create_kw)
                break
            except Exception as e:  # transient API errors
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        text = response.choices[0].message.content or ""
        self.stats.update(list(presentation), response)
        return self._parse_ids(text, presentation)

    @staticmethod
    def _parse_ids(text: str, batch: Sequence[str]) -> List[str]:
        return AmazonPriceSorter._parse_ids_detail(text, batch)[0]

    @staticmethod
    def _parse_ids_detail(text: str, batch: Sequence[str]):
        """(order, n_missing): ids the model omitted are appended in input order."""
        ordered: List[str] = []
        ids: Any = None
        obj = re.search(r"\{.*\}", text, re.S)
        if obj:
            try:
                ids = json.loads(obj.group()).get("order")
            except (json.JSONDecodeError, AttributeError):
                ids = None
        if ids is None:
            m = re.search(r"\[[\s\d,]*\]", text)
            if m:
                try:
                    ids = json.loads(m.group())
                except json.JSONDecodeError:
                    ids = None
        if isinstance(ids, list):
            for v in ids:
                if isinstance(v, str) and v.strip().isdigit():
                    v = int(v)
                if isinstance(v, int) and 1 <= v <= len(batch) and batch[v - 1] not in ordered:
                    ordered.append(batch[v - 1])
        missing = [a for a in batch if a not in ordered]
        return ordered + missing, len(missing)


def load_sample(n: int, seed: int) -> pd.DataFrame:
    gt = pd.read_csv(GT_CSV, dtype={"parent_asin": str})
    return gt.sample(n=n, random_state=seed).reset_index(drop=True)
