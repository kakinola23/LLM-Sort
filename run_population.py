#!/usr/bin/env python3
"""World population 2020 (n=200 countries): sort countries by population (Kendall tau).

  python run_population.py --method tbs --model GPT-4o -m 10 --out results/population/gpt4o_m10.json
  python run_population.py --method tbs --model GPT-4o -m 10 --prompt-mode explicit --out ...  # true values shown
  python run_population.py --method merge --model GPT-4o --out ...                             # Zhao merge, m=4
  python run_population.py --method pointwise --model GPT-4o --out results/population/gpt4o_pw.json
  python run_population.py --method tbs --model GPT-4o -m 10 --values-from results/population/gpt4o_pw.json --out ...

Every LLM reply is cached next to --out (``*.calls.jsonl``); rerunning the same command resumes
without paying for finished calls.
"""

from __future__ import annotations

import argparse
import sys

from loaders import population
from tbs import item_runner

sys.path.insert(0, str(population.ROOT / "third_party" / "orderbyllm"))
from prompts.all_prompts import (  # noqa: E402
    population_external_comparison_prompt_template,
    population_pointwise_prompt_template,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    item_runner.add_args(ap)
    args = ap.parse_args()
    items, true_order, pop_map = population.load(seed=args.seed, shuffle=True)
    item_runner.run(
        dataset="population", args=args, items=items, true_order=true_order, value_map=pop_map,
        make_tbs=lambda mode: (population.PopulationTBSSorter if mode == "knowledge"
                               else population.PopulationExplicitTBSSorter, {"structured_output": True}),
        merge_prompt=population_external_comparison_prompt_template,
        pointwise_prompt=population_pointwise_prompt_template,
        value_scale=1e6,  # pointwise answers are in millions; the explicit prompt divides by 1e6
    )


if __name__ == "__main__":
    main()
