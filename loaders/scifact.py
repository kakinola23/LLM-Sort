"""SciFact (BEIR) fact-checking retrieval: fixed 50-claim test slice x E5 top-100 abstracts.

Corpus/queries/qrels come from BEIR's SciFact (``python scripts/prepare_scifact.py``). The top-100
candidates per claim were retrieved once with intfloat/e5-base-v2 (LOTUS setup) and are shipped in
``data/scifact_slice/e5_top100_test50.json`` so every run sees identical candidates. Candidates are
shuffled with one Random(seed) shared across claims in slice order.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import (  # noqa: E402
    passage_external_comparison_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
)

from tbs.passage_sorter import StrictZhaoSorter  # noqa: E402,F401  (TBS listwise adapter)

BEIR_DIR = ROOT / "data" / "scifact"
SLICE_FILE = ROOT / "data" / "scifact_slice" / "e5_top100_test50.json"

PROMPTS = {
    "external": passage_external_comparison_prompt_template,
    "pairwise": passage_pairwise_comparison_prompt_template,
    "pointwise": passage_pointwise_prompt_template,
}


def load_corpus(data_dir: Path = BEIR_DIR) -> Dict[str, str]:
    """doc_id -> abstract text (no title, no truncation; LOTUS setup)."""
    docs: Dict[str, str] = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                docs[str(o["_id"])] = (o.get("text") or "").strip()
    return docs


def load_qrels(data_dir: Path = BEIR_DIR, split: str = "test") -> Dict[str, Dict[str, int]]:
    rel: Dict[str, Dict[str, int]] = {}
    with open(data_dir / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rel.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return rel


def load(seed: int = 0) -> Tuple[List[Tuple[str, str, List[str]]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Return (jobs [(claim id, claim, shuffled top-100)], corpus, qrels)."""
    sl = json.loads(SLICE_FILE.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    jobs = []
    for c in sl["claims"]:
        cands = list(c["top100"])
        rng.shuffle(cands)
        jobs.append((c["qid"], c["claim"], cands))
    return jobs, load_corpus(), load_qrels()
