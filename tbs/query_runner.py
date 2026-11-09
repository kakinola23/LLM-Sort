"""Shared driver for query-conditioned datasets (DL19, SciFact, MovieLens).

Each unit (query / claim / user) has a list of candidate ids. A method orders them best-first;
we record the full order, full-list nDCG (the paper's metric), nDCG@10, calls and tokens as one
JSON line per unit, skip finished units on rerun, and write a ``_summary.json`` with totals and
USD cost at list prices.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from methods.tbs import add_tbs_args, tbs_kwargs
from tbs.llm_clients import cost_usd, make_async_client, make_client
from tbs.metrics import ndcg_cut_10, ndcg_full

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "orderbyllm"))

METHODS = ("tbs", "merge", "qs1", "qs3", "pointwise")

# Job = (unit_id, query_text, shuffled candidate ids)
Job = Tuple[str, str, List[str]]


def add_common_args(ap, default_out: str) -> None:
    ap.add_argument("--method", choices=METHODS, default="tbs")
    ap.add_argument("--model", default="GPT-4o")
    ap.add_argument("--seed", type=int, default=0, help="candidate shuffle seed")
    ap.add_argument("--workers", type=int, default=4, help="units processed concurrently")
    ap.add_argument("--max-inflight", type=int, default=16, help="baselines: concurrent API requests")
    ap.add_argument("--max-units", type=int, default=0, help="debug: only the first N units")
    ap.add_argument("--merge-batch-size", type=int, default=4, help="merge: batch size")
    ap.add_argument("--limit-k", type=int, default=0, help="qs1/qs3: stop after the top-k (0 = full list)")
    ap.add_argument("--out-dir", default=default_out)
    add_tbs_args(ap, required=False)


def variant_name(args) -> str:
    tag = args.model.replace(".", "_").replace("-", "_")
    if args.method == "merge":
        return f"{tag}_merge_m{args.merge_batch_size}"
    if args.method in ("qs1", "qs3"):
        return f"{tag}_{args.method}_" + (f"limit{args.limit_k}" if args.limit_k else "full")
    if args.method == "pointwise":
        return f"{tag}_pointwise"
    v = f"{tag}_tbs_m{args.batch_size}"
    if args.round1_partition != "contiguous":
        v += f"_r1{args.round1_partition}"
    if args.round2_mode != "recursive":
        v += f"_r2{args.round2_mode}"
    if args.round2_all_rows:
        v += "_r2allrows"
    if args.round3_mode != "diagonals_then_unresolved":
        v += f"_r3{args.round3_mode}"
    if args.no_transitivity:
        v += "_notrans"
    if args.theta < 1.0:
        v += f"_theta{args.theta}"
    return v


def _row(unit_id: str, best_first: List[str], judged: Dict[str, int], counts: Dict[str, Any]) -> Dict[str, Any]:
    return {"qid": unit_id, "ndcg_full": ndcg_full(best_first, judged), "ndcg10": ndcg_cut_10(best_first, judged),
            **counts, "order_best_first": best_first}


def run(
    *,
    dataset: str,
    args,
    jobs: Sequence[Job],
    text_of: Callable[[str, str], str],
    qrels: Dict[str, Dict[str, int]],
    tbs_sorter_cls: Any,
    prompts: Dict[str, str],
    prepare_baselines: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run ``args.method`` over ``jobs``; ``prompts`` holds 'external', 'pairwise', 'pointwise' templates."""
    if args.method == "tbs" and not args.batch_size:
        raise SystemExit("--batch-size/-m is required for --method tbs")
    if args.max_units:
        jobs = list(jobs)[: args.max_units]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = variant_name(args)
    rows_path = out_dir / f"{variant}.jsonl"
    os.environ.setdefault("TBS_CALL_CACHE", str(out_dir / f"{variant}.calls.jsonl"))

    done: Dict[str, Dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["qid"]] = row
    todo = [j for j in jobs if j[0] not in done]
    print(f"{dataset} {variant}: units={len(jobs)} done={len(done)} todo={len(todo)}", flush=True)
    out = rows_path.open("a", encoding="utf-8")

    def record(row: Dict[str, Any]) -> None:
        out.write(json.dumps(row) + "\n")
        out.flush()
        done[row["qid"]] = row
        print(f"  {row['qid']}: nDCG={row['ndcg_full']:.3f} calls={row['calls']} ({len(done)}/{len(jobs)})", flush=True)

    if args.method == "tbs":
        client, model_id = make_client(args.model)

        def work(job: Job) -> Optional[Dict[str, Any]]:
            uid, query, cands = job
            passages = {d: text_of(uid, d) for d in cands}
            sorter = tbs_sorter_cls(
                items=cands, client=client, model=model_id,
                sort_by="height", height_map={d: 0.0 for d in cands}, height_prompt_mode="knowledge",
                prompt_order="input", text_map=passages, structured_output=True,
                query=query, passages=passages, **tbs_kwargs(args))
            t0 = time.time()
            try:
                worst_to_best, stats = sorter.sort()
            except RuntimeError as e:
                print(f"  {uid}: SKIPPED ({e})", flush=True)
                return None
            best_first = list(reversed(worst_to_best))
            return _row(uid, best_first, qrels.get(uid, {}), {
                "calls": stats.count, "round1_calls": sorter.round1_calls, "round2_calls": sorter.round2_calls,
                "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
                "complete": sorter.graph.is_complete(), "rho": round(sorter.known_pair_counts()[2], 6),
                "stop_reason": getattr(getattr(sorter, "round3_stats", None), "stop_reason", ""),
                "time_s": round(time.time() - t0, 1)})

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(work, j) for j in todo]):
                row = fut.result()
                if row is not None:
                    record(row)
    else:
        from methods import merge_sort, pointwise, quicksort

        model_id = make_client(args.model)[1]
        if prepare_baselines:
            prepare_baselines()
        if args.method == "pointwise":
            pointwise.fresh_cache()
        if args.method in ("qs1", "qs3"):
            quicksort.install_fallback_counter()
            aclient = quicksort.limit_concurrency(make_async_client(), args.max_inflight)
        else:
            aclient = merge_sort.counting_client(make_async_client(), args.max_inflight)
        sem = asyncio.Semaphore(args.workers)

        async def one(job: Job) -> None:
            uid, query, cands = job
            keyed = [(d, text_of(uid, d)) for d in cands]
            async with sem:
                try:
                    if args.method == "merge":
                        p = merge_sort.safe_prompt(prompts["external"], question=query)
                        best, counts = await merge_sort.merge_sort_passages(
                            keyed, p, aclient, model_id, batch_size=args.merge_batch_size)
                    elif args.method == "pointwise":
                        p = merge_sort.safe_prompt(prompts["pointwise"], question=query)
                        best, counts = await pointwise.pointwise_passages(uid, keyed, p, aclient, model_id)
                    else:
                        p = merge_sort.safe_prompt(prompts["pairwise"], question=query)
                        best, counts = await quicksort.quicksort_passages(
                            uid, keyed, p, aclient, model_id, vote=1 if args.method == "qs1" else 3,
                            limit_k=args.limit_k or None)
                        if counts["fallbacks"] and counts["rate_limited"]:
                            print(f"  {uid}: SKIPPED ({counts['fallbacks']} comparisons fell back under rate limiting)", flush=True)
                            return
                except RuntimeError as e:
                    print(f"  {uid}: SKIPPED ({e})", flush=True)
                    return
            if sorted(best) != sorted(cands):
                print(f"  {uid}: SKIPPED (order does not cover all {len(cands)} candidates)", flush=True)
                return
            record(_row(uid, best, qrels.get(uid, {}), counts))

        async def all_units() -> None:
            await asyncio.gather(*(one(j) for j in todo))

        asyncio.run(all_units())
    out.close()

    missing = [j[0] for j in jobs if j[0] not in done]
    if missing:
        print(f"Incomplete: {len(missing)} units skipped ({missing}); rerun the same command to fill them in.", flush=True)
        return {}
    rows = [done[j[0]] for j in jobs]
    pt, ct = sum(r["prompt_tokens"] for r in rows), sum(r["completion_tokens"] for r in rows)
    summary = {
        "dataset": dataset, "method": args.method, "model": args.model, "variant": variant, "seed": args.seed,
        **({"batch_size_m": args.batch_size, "theta": args.theta, "round1_partition": args.round1_partition,
            "round2_mode": args.round2_mode, "round2_all_rows": args.round2_all_rows,
            "round3_mode": args.round3_mode, "use_transitivity": not args.no_transitivity,
            "all_complete": all(r.get("complete", True) for r in rows)} if args.method == "tbs" else {}),
        "n_units": len(rows),
        "ndcg_full": round(sum(r["ndcg_full"] for r in rows) / len(rows), 4),
        "ndcg10": round(sum(r["ndcg10"] for r in rows) / len(rows), 4),
        "total_calls": sum(r["calls"] for r in rows),
        "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
        "cost_usd": round(cost_usd(model_id, pt, ct), 2),
    }
    (out_dir / f"{variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done: " + json.dumps(summary), flush=True)
    return summary
