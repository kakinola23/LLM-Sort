#!/usr/bin/env python3
"""Collect every result under results/ into one CSV: dataset, method, model, m, switches,
quality (Kendall tau or full-list nDCG), calls, tokens and USD cost.

  python scripts/summarize_results.py                 # -> results/summary.csv
  python scripts/summarize_results.py --root results/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tbs.llm_clients import cost_usd, resolve_model  # noqa: E402

COLS = ["file", "dataset", "method", "model", "batch_size_m", "theta", "round1_partition", "round2_mode",
        "round2_all_rows", "use_transitivity", "quality_metric", "quality", "total_calls",
        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"]


def row_from(path: Path, d: dict) -> dict | None:
    if "dataset" not in d or "total_calls" not in d:
        return None
    if "tau" in d:
        metric, q = "kendall_tau", d["tau"]
    elif d["dataset"] == "movielens":
        metric, q = "ndcg_U", d.get("ndcg")
    else:
        metric, q = f"ndcg_{'Q' if d['dataset'] == 'dl19' else 'C'}", d.get("ndcg_full")
    pt, ct = int(d.get("prompt_tokens", 0)), int(d.get("completion_tokens", 0))
    cost = d.get("cost_usd")
    if cost is None:
        try:
            cost = round(cost_usd(resolve_model(d["model"])["model"], pt, ct), 4)
        except (KeyError, ValueError):
            cost = ""
    r = {k: d.get(k, "") for k in COLS}
    r.update(file=str(path.relative_to(ROOT)), quality_metric=metric, quality=q,
             prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, cost_usd=cost)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()
    rows = []
    for p in sorted((ROOT / args.root).rglob("*.json")):
        if p.name in ("final_result.json", "run_config.json") or "code_snapshot" in p.parts:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and (r := row_from(p, d)):
            rows.append(r)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
