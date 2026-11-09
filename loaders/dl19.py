"""TREC DL19 passage ranking under OrderByLLM's protocol: 43 queries x BM25 top-100.

Files (``data/dl19/``): BM25 run, TREC qrels, query text, and ``dl19_passages_43x100.csv``
(passage text; build it with ``python scripts/prepare_dl19.py``, MS MARCO is not redistributed).
Each query's candidates are shuffled with one Random(seed) shared across queries in sorted-qid order.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import (  # noqa: E402
    passage_external_comparison_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
)

from tbs.passage_sorter import DL19ZhaoSorter  # noqa: E402,F401  (TBS listwise adapter)

PROMPTS = {
    "external": passage_external_comparison_prompt_template,
    "pairwise": passage_pairwise_comparison_prompt_template,
    "pointwise": passage_pointwise_prompt_template,
}

DATA = ROOT / "data" / "dl19"
RUN_FILE = DATA / "run.msmarco-v1-passage.bm25-default.dl19.txt"
PASSAGES_CSV = DATA / "dl19_passages_43x100.csv"
QRELS_FILE = DATA / "2019qrels-pass.txt"


def load_dl19(hit_depth: int = 100):
    text: Dict[tuple, str] = {}
    query_text: Dict[str, str] = {}
    with PASSAGES_CSV.open(encoding="utf-8", newline="") as f:
        for qid, qtext, _rank, docid, passage, _rel in csv.reader(f):
            text[(qid, docid)] = passage
            query_text[qid] = qtext

    by_qid: Dict[str, List[tuple]] = defaultdict(list)
    with RUN_FILE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, _, _ = parts
            if int(rank) <= hit_depth and qid in query_text:
                by_qid[qid].append((int(rank), docid))

    first_stage = []
    for qid, entries in sorted(by_qid.items(), key=lambda x: x[0]):
        entries.sort()
        first_stage.append((qid, query_text[qid], [d for _, d in entries]))

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with QRELS_FILE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return first_stage, text, dict(qrels)


def shuffled_jobs(first_stage, seed: int = 0):
    """[(qid, query, shuffled docids)] with one Random(seed) shared across queries (OrderByLLM)."""
    rng = random.Random(seed)
    jobs = []
    for qid, query, docids in first_stage:
        shuffled = docids[:]
        rng.shuffle(shuffled)
        jobs.append((qid, query, shuffled))
    return jobs

