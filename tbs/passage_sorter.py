"""TBS adapter for query-conditioned ranking (DL19, SciFact, MovieLens): the listwise call is
OrderByLLM's passage external-comparison prompt with structured output, "worst to best"."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import passage_external_comparison_prompt_template  # noqa: E402

from .prompts import ZHAO_SYSTEM_PROMPT  # noqa: E402
from .sorter import BatchedLLMSorter  # noqa: E402


class PassageExternalComparisonReasoning(BaseModel):
    Reasoning_steps: str
    worstToBestPassageKeys: List[int]



class DL19ZhaoSorter(BatchedLLMSorter):
    """TBS whose listwise call is OrderByLLM's external_comparisons on passages."""

    def __init__(self, *args, query: str, passages: Dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.query = query
        self.passages = passages

    def _build_prompt(self, batch: Sequence[str], round_id: int, payload: Dict[str, object]) -> str:
        keys = "\n".join(f"passage_id:{i + 1}\n{self.passages[d]}\n\n" for i, d in enumerate(batch))
        return passage_external_comparison_prompt_template.format(question=self.query, keys=keys)

    def _sort_with_llm(
        self, batch: Sequence[str], round_id: int, payload: Optional[Dict[str, object]] = None
    ) -> List[str]:
        presentation = list(batch)
        prompt = self._build_prompt(presentation, round_id, payload or {})
        n = len(presentation)
        best: List[str] = []
        for attempt in range(1, 6):
            suffix = "" if attempt == 1 else (
                f"\nRespond in JSON. Make sure the JSON worstToBest list has length {n} "
                f"and contains all ids {list(range(1, n + 1))}.\n"
            )
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ZHAO_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + suffix},
                    ],
                    temperature=0.0,
                    response_format=PassageExternalComparisonReasoning,
                    max_completion_tokens=12 * 1024,
                )
            except Exception as e:  # noqa: BLE001
                err = str(e).lower()
                time.sleep(15 if ("429" in err or "rate limit" in err) else 3)
                continue
            self.stats.update(presentation, response)
            parsed = response.choices[0].message.parsed
            ids = list(parsed.worstToBestPassageKeys) if parsed is not None else []
            ordered: List[str] = []
            for i in ids:
                if isinstance(i, int) and 1 <= i <= n and presentation[i - 1] not in ordered:
                    ordered.append(presentation[i - 1])
            if len(ordered) == n:
                return ordered
            if len(ordered) > len(best):
                best = ordered
        return best + [x for x in presentation if x not in best]


class StrictZhaoSorter(DL19ZhaoSorter):
    """API errors back off and retry without consuming a parse attempt; a batch is never
    silently replaced by its input order because of rate limiting."""

    def _sort_with_llm(self, batch, round_id, payload=None):
        presentation = list(batch)
        prompt = self._build_prompt(presentation, round_id, payload or {})
        n = len(presentation)
        best: List[str] = []
        attempt, api_errors = 0, 0
        while attempt < 5:
            suffix = "" if attempt == 0 else (
                f"\nRespond in JSON. Make sure the JSON worstToBest list has length {n} "
                f"and contains all ids {list(range(1, n + 1))}.\n"
            )
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ZHAO_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + suffix},
                    ],
                    temperature=0.0,
                    response_format=PassageExternalComparisonReasoning,
                    max_completion_tokens=12 * 1024,
                    timeout=90.0,
                )
            except Exception as e:  # noqa: BLE001
                api_errors += 1
                print(f"    API error {api_errors}: {str(e)[:200]}", flush=True)
                if api_errors > 30:
                    raise RuntimeError(f"giving up after {api_errors} API errors: {e}") from e
                time.sleep(min(60.0, 5.0 * 2 ** min(api_errors, 4)) * (0.5 + random.random()))
                continue
            attempt += 1
            self.stats.update(presentation, response)
            parsed = response.choices[0].message.parsed
            ids = list(parsed.worstToBestPassageKeys) if parsed is not None else []
            ordered: List[str] = []
            for i in ids:
                if isinstance(i, int) and 1 <= i <= n and presentation[i - 1] not in ordered:
                    ordered.append(presentation[i - 1])
            if len(ordered) == n:
                return ordered
            if len(ordered) > len(best):
                best = ordered
        return best + [x for x in presentation if x not in best]
