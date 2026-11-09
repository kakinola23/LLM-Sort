"""Shared driver for item datasets with a single global order (NBA heights, world population).

Methods: TBS (``--prompt-mode knowledge`` = names only; ``explicit`` = a value shown next to each
name, either the true value or, with ``--values-from``, the model's own pointwise estimates),
OrderByLLM merge sort, and OrderByLLM pointwise scoring. Quality is Kendall tau against the true
order; the result JSON also holds the full predicted order, calls, tokens and USD cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from methods.tbs import add_tbs_args, run_tbs, tbs_kwargs
from tbs.llm_clients import cost_usd, make_async_client, make_client
from tbs.metrics import kendalls_tau


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--method", choices=("tbs", "merge", "pointwise"), default="tbs")
    ap.add_argument("--model", default="GPT-4o")
    ap.add_argument("--seed", type=int, default=0, help="input shuffle seed")
    ap.add_argument("--prompt-mode", choices=("knowledge", "explicit"), default="knowledge",
                    help="tbs: knowledge = names only (OrderByLLM); explicit = value shown per item")
    ap.add_argument("--values-from", default="",
                    help="tbs explicit: a --method pointwise result JSON; show the model's own values")
    ap.add_argument("--structured-schema", choices=("reasoning", "recall"), default="reasoning",
                    help="reply schema: OrderByLLM {Reasoning_steps, sorted_list}, or 'recall' (values first)")
    ap.add_argument("--merge-batch-size", type=int, default=4)
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--verbose", action="store_true")
    add_tbs_args(ap, required=False)


def run(
    *,
    dataset: str,
    args,
    items: List[str],
    true_order: List[str],
    value_map: Dict[str, float],
    make_tbs: Callable[[str], Any],
    merge_prompt: str,
    pointwise_prompt: str,
    value_scale: float = 1.0,
) -> Dict[str, Any]:
    """``make_tbs(prompt_mode)`` returns (sorter class, extra sorter kwargs)."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TBS_CALL_CACHE", str(out.with_suffix("")) + ".calls.jsonl")
    client, model_id = make_client(args.model)
    meta: Dict[str, Any] = {"dataset": dataset, "method": args.method, "model": args.model,
                            "model_id": model_id, "seed": args.seed}

    if args.method == "tbs":
        if not args.batch_size:
            raise SystemExit("--batch-size/-m is required for --method tbs")
        shown = value_map
        if args.values_from:
            est = json.loads(Path(args.values_from).read_text(encoding="utf-8"))["values"]
            shown = {k: float(est.get(k, 0.0)) * value_scale for k in items}
            args.prompt_mode = "explicit"
        sorter_cls, extra = make_tbs(args.prompt_mode)
        res = run_tbs(sorter_cls, items, true_order, client=client, model=model_id, verbose=args.verbose,
                      sort_by="height", height_map=shown, height_prompt_mode=args.prompt_mode,
                      prompt_order="input", **extra, **tbs_kwargs(args))
        meta.update({"prompt_mode": args.prompt_mode, "values_source": "pointwise" if args.values_from else
                     ("true" if args.prompt_mode == "explicit" else "none"),
                     "batch_size_m": args.batch_size, "theta": args.theta,
                     "round1_partition": args.round1_partition, "round2_mode": args.round2_mode,
                     "round2_all_rows": args.round2_all_rows, "round3_mode": args.round3_mode,
                     "use_transitivity": not args.no_transitivity, "structured_schema": args.structured_schema})
    elif args.method == "merge":
        from methods import merge_sort

        if args.structured_schema == "recall":
            merge_sort.use_recall_schema()
        aclient = merge_sort.counting_client(make_async_client())
        order, counts = asyncio.run(merge_sort.merge_sort_items(
            items, merge_prompt, aclient, model_id, batch_size=args.merge_batch_size))
        res = {"n": len(items), "total_calls": counts.pop("calls"), **counts,
               "tau": round(kendalls_tau(order, true_order), 6), "predicted_order": order}
        meta.update({"merge_batch_size": args.merge_batch_size, "structured_schema": args.structured_schema})
    else:
        from methods import pointwise

        order, counts = pointwise.pointwise_items(items, pointwise_prompt, client, model_id)
        res = {"n": len(items), "total_calls": counts.pop("calls"), **counts,
               "tau": round(kendalls_tau(order, true_order), 6), "predicted_order": order}

    result = {**meta, **res}
    result["cost_usd"] = round(cost_usd(model_id, result["prompt_tokens"], result["completion_tokens"]), 4)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{dataset} {args.method}: tau={result['tau']:.4f} calls={result['total_calls']} "
          f"tokens={result['prompt_tokens'] + result['completion_tokens']} cost=${result['cost_usd']} -> {out}")
    return result
