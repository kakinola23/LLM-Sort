#!/usr/bin/env python3
"""MovieLens-1M preference ranking (Phase-1 slice: 63 users x 70 movies).

  python run_movielens.py --method tbs --model GPT-4o -m 10
  python run_movielens.py --method merge --model GPT-4o      # Zhao merge sort, movie-preference prompt
  python run_movielens.py --method qs1 --model GPT-4o        # pairwise quicksort (qs3 = 3 votes)

One JSON line per user (full predicted order, nDCG_U, nDCG@10, calls, tokens) goes to
``<out-dir>/<variant>.jsonl``; totals and USD cost to ``<variant>_summary.json``. Rerunning the
same command skips finished users and replays LLM replies from ``<variant>.calls.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from loaders import movielens
from methods.tbs import add_tbs_args
from tbs.llm_clients import cost_usd, make_async_client, make_client

# Zhao's passage_external_comparison_prompt_template reworded for movie preference (merge baseline).
MOVIELENS_EXTERNAL_COMPARISON_PROMPT = """You are given a user's movie viewing history and a list of candidate movies with movie id at the front.
Rank the movies by how much this user would prefer them, from least preferred to most preferred.
User history: {question}
Movies:
{keys}
Output a JSON list of integer ids, representing the ranking from least to most preferred.
"""


def variant_name(args) -> str:
    tag = args.model.replace(".", "_").replace("-", "_")
    if args.method != "tbs":
        return f"{tag}_{args.method}" + (f"_m{args.merge_batch_size}" if args.method == "merge" else "")
    v = f"{tag}_tbs_m{args.batch_size}"
    if args.round1_partition != "contiguous":
        v += f"_r1{args.round1_partition}"
    if args.round2_mode != "recursive":
        v += f"_r2{args.round2_mode}"
    if args.round2_all_rows:
        v += "_r2allrows"
    if args.no_transitivity:
        v += "_notrans"
    if args.theta < 1.0:
        v += f"_theta{args.theta}"
    return v


def run_tbs_user(task, client, model_id, args) -> Dict[str, Any]:
    user_id, profile, movie_ids, gt, rel, display = task
    sim = args.round1_partition in ("similarity", "similarity_embed")
    sorter = movielens.MovieLensPackingSorter(
        items=movie_ids, batch_size=args.batch_size, client=client, model=model_id, verbose=False,
        completion_threshold=args.theta, enable_completion_early_stop=args.theta < 1.0,
        enable_stall_early_stop=False, round1_partition=args.round1_partition, prompt_order="input",
        round2_mode=args.round2_mode, round2_all_rows=args.round2_all_rows,
        round3_mode="diagonals_then_unresolved", height_map={m: 0.0 for m in movie_ids},
        text_map=display if sim else None,  # similarity presets cluster on title + genres
        profile_text=profile, display_map=display, partition_seed=42 + int(user_id),
        use_transitivity=not args.no_transitivity, graph_backend=args.graph_backend,
    )
    t0 = time.time()
    predicted, stats = sorter.sort()
    return {"user_id": user_id, "calls": stats.count, "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens, "time_s": round(time.time() - t0, 2),
            "predicted_order": list(predicted), **movielens.evaluate(predicted, gt, rel)}


async def run_merge_user(task, aclient, model_id, args) -> Dict[str, Any]:
    from methods import merge_sort

    user_id, profile, movie_ids, gt, rel, display = task
    shuffled = movie_ids[:]
    random.Random(args.seed).shuffle(shuffled)
    keyed = [(mid, display.get(mid, mid)) for mid in shuffled]
    prompt = merge_sort.safe_prompt(MOVIELENS_EXTERNAL_COMPARISON_PROMPT, question=profile)
    best, counts = await merge_sort.merge_sort_passages(keyed, prompt, aclient, model_id,
                                                        batch_size=args.merge_batch_size)
    return {"user_id": user_id, **counts, "predicted_order": best, **movielens.evaluate(best, gt, rel)}


def run_qs_user(task, client, model_id, args) -> Dict[str, Any]:
    from methods.quicksort import quicksort_with_comparator

    user_id, profile, movie_ids, gt, rel, display = task
    comp = movielens.MovieLensPairwiseComparator(
        client, model_id, profile, display, random.Random(f"{args.seed}:{user_id}"))
    t0 = time.time()
    predicted = asyncio.run(quicksort_with_comparator(movie_ids, comp.compare, vote=1 if args.method == "qs1" else 3))
    return {"user_id": user_id, "calls": comp.calls, "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens, "unparsed_fallbacks": comp.fallbacks,
            "time_s": round(time.time() - t0, 2), "predicted_order": predicted,
            **movielens.evaluate(predicted, gt, rel)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("tbs", "merge", "qs1", "qs3"), default="tbs")
    ap.add_argument("--model", default="GPT-4o")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3, help="users processed concurrently")
    ap.add_argument("--max-inflight", type=int, default=16, help="merge: concurrent API requests")
    ap.add_argument("--merge-batch-size", type=int, default=4)
    ap.add_argument("--max-units", type=int, default=0, help="debug: only the first N users")
    ap.add_argument("--out-dir", default="results/movielens")
    add_tbs_args(ap, required=False)
    args = ap.parse_args()
    if args.method == "tbs" and not args.batch_size:
        raise SystemExit("--batch-size/-m is required for --method tbs")

    tasks = movielens.load_tasks()
    if args.max_units:
        tasks = tasks[: args.max_units]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = variant_name(args)
    rows_path = out_dir / f"{variant}.jsonl"
    os.environ.setdefault("TBS_CALL_CACHE", str(out_dir / f"{variant}.calls.jsonl"))
    client, model_id = make_client(args.model)

    done: Dict[str, Dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["user_id"]] = row
    todo = [t for t in tasks if t[0] not in done]
    print(f"MovieLens {variant}: users={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
    out = rows_path.open("a", encoding="utf-8")

    def record(row: Dict[str, Any]) -> None:
        out.write(json.dumps(row) + "\n")
        out.flush()
        done[row["user_id"]] = row
        print(f"  user {row['user_id']}: nDCG_U={row['ndcg']:.4f} calls={row['calls']} ({len(done)}/{len(tasks)})", flush=True)

    if args.method == "merge":
        from methods import merge_sort

        merge_sort.use_movie_id_labels()
        aclient = merge_sort.counting_client(make_async_client(), args.max_inflight)
        sem = asyncio.Semaphore(args.workers)

        async def one(task):
            async with sem:
                record(await run_merge_user(task, aclient, model_id, args))

        async def all_users():
            await asyncio.gather(*(one(t) for t in todo))

        asyncio.run(all_users())
    else:
        fn = run_tbs_user if args.method == "tbs" else run_qs_user
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for fut in as_completed([pool.submit(fn, t, client, model_id, args) for t in todo]):
                record(fut.result())
    out.close()

    rows: List[Dict[str, Any]] = [done[t[0]] for t in tasks]
    pt, ct = sum(r["prompt_tokens"] for r in rows), sum(r["completion_tokens"] for r in rows)
    summary = {
        "dataset": "movielens", "method": args.method, "model": args.model, "variant": variant,
        **({"batch_size_m": args.batch_size, "theta": args.theta, "round1_partition": args.round1_partition,
            "round2_mode": args.round2_mode, "round2_all_rows": args.round2_all_rows,
            "use_transitivity": not args.no_transitivity} if args.method == "tbs" else {}),
        "n_users": len(rows),
        "ndcg": round(sum(r["ndcg"] for r in rows) / len(rows), 6),
        "ndcg10": round(sum(r["ndcg10"] for r in rows) / len(rows), 6),
        "total_calls": sum(r["calls"] for r in rows),
        "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
        "cost_usd": round(cost_usd(model_id, pt, ct), 2),
    }
    (out_dir / f"{variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
