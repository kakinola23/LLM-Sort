#!/usr/bin/env python3
"""TREC DL19 passage ranking (43 queries x BM25 top-100, OrderByLLM protocol).

  python run_dl19.py --method tbs --model GPT-4o -m 10
  python run_dl19.py --method tbs --model GPT-4o-mini -m 5 --no-transitivity
  python run_dl19.py --method merge --model GPT-4o          # Zhao external merge sort, m = 4
  python run_dl19.py --method qs1   --model GPT-4o          # pairwise quicksort, full list
  python run_dl19.py --method pointwise --model GPT-4o

One JSON line per query (full best-first order, nDCG_Q, nDCG@10, calls, tokens) goes to
``<out-dir>/<variant>.jsonl``; totals and USD cost to ``<variant>_summary.json``. Rerunning the
same command resumes from finished queries and from the LLM reply cache.
"""

from __future__ import annotations

import argparse

from loaders import dl19
from tbs import query_runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    query_runner.add_common_args(ap, default_out="results/dl19")
    args = ap.parse_args()

    first_stage, text, qrels = dl19.load_dl19()
    jobs = dl19.shuffled_jobs(first_stage, args.seed)
    query_runner.run(
        dataset="dl19", args=args, jobs=jobs,
        text_of=lambda qid, d: text[(qid, d)], qrels=qrels,
        tbs_sorter_cls=dl19.DL19ZhaoSorter, prompts=dl19.PROMPTS,
    )


if __name__ == "__main__":
    main()
