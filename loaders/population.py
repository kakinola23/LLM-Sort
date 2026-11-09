"""World population (2020) by country: loader and TBS prompt adapters.

Data: OrderByLLM's ``population_by_country_2020.csv`` (n=200 after their sample); ground truth is
ascending 2020 population, ties broken by country name.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import population_external_comparison_prompt_template  # noqa: E402

from tbs.partition import apply_prompt_order  # noqa: E402
from tbs.sorter import BatchedLLMSorter, _supports_temperature_zero  # noqa: E402

DATA_CSV = ROOT / "data" / "population" / "population_by_country_2020.csv"


def load(seed: int = 0, shuffle: bool = True, csv_path: Path = DATA_CSV) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Return (input items, true order ascending, population map)."""
    df = pd.read_csv(csv_path)
    countries = df["Country"].astype(str).tolist()
    pop_map = dict(zip(df["Country"].astype(str), df["Population (2020)"].astype(float)))
    true_order = (
        df.sort_values(by=["Population (2020)", "Country"], ascending=[True, True])["Country"]
        .astype(str)
        .tolist()
    )
    items = countries[:]
    if shuffle:
        random.Random(seed).shuffle(items)
    return items, true_order, pop_map


class PopulationTBSSorter(BatchedLLMSorter):
    """TBS with OrderByLLM population listwise prompt (knowledge-based)."""

    def _build_prompt(
        self, batch: Sequence[str], round_id: int, payload: Dict[str, object]
    ) -> str:
        keys_text = "\n".join(f"id:{i + 1}\n{name}\n" for i, name in enumerate(batch))
        return population_external_comparison_prompt_template.format(keys=keys_text)

    def _sort_with_llm(
        self, batch: Sequence[str], round_id: int, payload: Optional[Dict[str, object]] = None
    ) -> List[str]:

        presentation = apply_prompt_order(batch, self.prompt_order, self.partition_seed)
        prompt = self._build_prompt(presentation, round_id, payload or {})
        if self.structured_output:
            return self._structured_listwise(prompt, presentation)
        system_message = "You are a helpful assistant. Respond with valid JSON only."

        create_kw: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        }
        if _supports_temperature_zero(self.model):
            create_kw["temperature"] = 0

        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**create_kw)
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if attempt < 2 and (
                    "503" in err_str or "500" in err_str or "overloaded" in err_str or "unavailable" in err_str
                ):
                    if self.verbose:
                        print(f"    Retry {attempt + 1}/2 in 5s after: {e}")
                    time.sleep(5)
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error

        ordered = self._parse_response(response.choices[0].message.content, presentation)
        self.stats.update(list(batch), response)
        return ordered


class PopulationExplicitTBSSorter(BatchedLLMSorter):
    """TBS with 2020 population (millions) shown in each batch — explicit value path."""

    def _build_prompt(
        self, batch: Sequence[str], round_id: int, payload: Dict[str, object]
    ) -> str:
        lines = []
        for i, name in enumerate(batch):
            pop_m = self.height_map[name] / 1e6
            lines.append(f"{i}: {name} ({pop_m:.2f}M)")
        keys_text = "\n".join(lines)
        return (
            f"Given a list of countries with id at the front:\n\n"
            f"{keys_text}\n\n"
            f"Based on the provided 2020 population values (in millions), sort the countries in "
            f"ascending order by population.\n"
            f"Break ties alphabetically by country name.\n\n"
            f"Output a JSON list of integer ids, representing countries from least populated to "
            f"most populated."
        )

    def _sort_with_llm(
        self, batch: Sequence[str], round_id: int, payload: Optional[Dict[str, object]] = None
    ) -> List[str]:

        presentation = apply_prompt_order(batch, self.prompt_order, self.partition_seed)
        prompt = self._build_prompt(presentation, round_id, payload or {})
        system_message = (
            "You are a precise sorting oracle. Sort countries by the provided 2020 population "
            "values in millions (ascending). Break ties alphabetically. "
            "Return a JSON list of integer IDs only."
        )

        create_kw: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        }
        if _supports_temperature_zero(self.model):
            create_kw["temperature"] = 0

        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**create_kw)
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if attempt < 2 and (
                    "503" in err_str or "500" in err_str or "overloaded" in err_str or "unavailable" in err_str
                ):
                    if self.verbose:
                        print(f"    Retry {attempt + 1}/2 in 5s after: {e}")
                    time.sleep(5)
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error

        ordered = self._parse_response(response.choices[0].message.content, presentation)
        self.stats.update(list(batch), response)
        return ordered
