#!/usr/bin/env python3
"""NBA heights (n=200): sort players by height (Kendall tau vs. true heights).

  python run_nba.py --method tbs --model GPT-4o -m 10 --out results/nba/gpt4o_m10.json
  python run_nba.py --method tbs --model GPT-4o-mini -m 4 --structured-schema recall --out ...
  python run_nba.py --method tbs --model GPT-4o -m 10 --no-transitivity --out ...      # TBS-NoTrans
  python run_nba.py --method merge --model GPT-4o --out ...                            # Zhao merge, m=4
  python run_nba.py --method pointwise --model GPT-4o --out results/nba/gpt4o_pw.json
  python run_nba.py --method tbs --model GPT-4o -m 10 --values-from results/nba/gpt4o_pw.json --out ...

Every LLM reply is cached next to --out (``*.calls.jsonl``); rerunning the same command resumes
without paying for finished calls.
"""

from __future__ import annotations

import argparse
import sys

from loaders import nba
from tbs import item_runner
from tbs.sorter import BatchedLLMSorter

sys.path.insert(0, str(nba.ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import nba_external_comparison_prompt_template, nba_pointwise_prompt_template  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    item_runner.add_args(ap)
    args = ap.parse_args()
    items, true_order, height_map = nba.load(seed=args.seed, shuffle=True)
    item_runner.run(
        dataset="nba", args=args, items=items, true_order=true_order, value_map=height_map,
        make_tbs=lambda mode: (BatchedLLMSorter, {"structured_output": True,
                                                  "structured_schema": args.structured_schema}),
        merge_prompt=nba_external_comparison_prompt_template,
        pointwise_prompt=nba_pointwise_prompt_template,
    )


if __name__ == "__main__":
    main()
