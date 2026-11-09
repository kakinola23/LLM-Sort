#!/usr/bin/env python3
"""SciFact claims (50-claim test slice x E5 top-100 abstracts, OrderByLLM passage protocol).

  python run_scifact.py --method tbs --model GPT-4o -m 10
  python run_scifact.py --method tbs --model GPT-4o-mini -m 4
  python run_scifact.py --method merge --model GPT-4o
  python run_scifact.py --method qs1 --model GPT-4o

One JSON line per claim (full best-first order, nDCG_C, nDCG@10, calls, tokens) goes to
``<out-dir>/<variant>.jsonl``; totals and USD cost to ``<variant>_summary.json``. Rerunning the
same command resumes from finished claims and from the LLM reply cache.
"""

from __future__ import annotations

import argparse

from loaders import scifact
from tbs import query_runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    query_runner.add_common_args(ap, default_out="results/scifact")
    args = ap.parse_args()

    jobs, corpus, qrels = scifact.load(args.seed)
    query_runner.run(
        dataset="scifact", args=args, jobs=jobs,
        text_of=lambda qid, d: corpus[d], qrels=qrels,
        tbs_sorter_cls=scifact.StrictZhaoSorter, prompts=scifact.PROMPTS,
    )


if __name__ == "__main__":
    main()
