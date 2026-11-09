#!/usr/bin/env python3
"""Amazon price ranking at scale (resumable, parallel TBS; paper: n = 20K and 1M, GPT-5-nano).

Usage: python run_amazon.py <run_name> <N> [options]
  e.g. python run_amazon.py n1m 1000000            # paper setting: explicit prompt, m=40, theta=0.9
       python run_amazon.py n200 200 --m 20 --model-id gpt-4o-mini
       python run_amazon.py n1m 1000000 --max-live-calls 20000   # stop safely, resume later

Results go to results/amazon_1m_run_<run_name>/ (final_result.json, sorted orders, call cache,
code snapshot). Rerunning the same command resumes from the call cache.

Options:
  --backend dag|chain|sets  transitive-order backend (default dag; chain needs
                        O(n * n/m) memory and does not scale past ~50k items;
                        sets = explicit pairwise closure / NoReachIndex)
  --max-live-calls K    halt (cache intact) before the (K+1)-th live API call
  --sim                 no API calls: cache misses are answered from the
                        LLM's own Round-1 price estimates plus noise, written
                        to a separate amazon_1m_run_<run_name>_sim directory
                        seeded with the real run's cache. For timing / call-
                        count projections only.
  --block B             chains per exact-count pass (memory ~ N*B bytes)
  --tau-stop-threshold T  halt if Round-3 tau stays below T for 3 checks
                        (default 0.5; pass a negative value to disable)
"""
import argparse
import dataclasses
import json
import random
import shutil
import sys
import threading
import time
import tracemalloc
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import psutil
from loaders.amazon import PROMPTS, explicit_display, load_sample
from tbs.llm_clients import make_client_for_id
from tbs.reach_index import DagTransitiveOrder, fast_kendalls_tau
from tbs.resumable_sorter import (
    make_resumable_sorter_class, TauTooLowError, LiveCallBudgetError, IncompleteAnswerError, StopAfterRound1,
)
from tbs.rank_aggregation import order_from_scores, plackett_luce_scores, price_offset_scores
from scipy.stats import kendalltau

M = 40
THETA = 0.9
SEED = 0
PROMPT_KEY = "explicit"
R1_WORKERS = 40
TAU_CHECK_INTERVAL = 2000
TAU_STOP_THRESHOLD = 0.5
TAU_STOP_CONSECUTIVE = 3

# USD per 1M tokens (input, output) at list price; output includes hidden reasoning tokens.
MODEL_PRICES = {
    "gpt-5.1": (1.25, 10.00), "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00), "gpt-5-nano": (0.05, 0.40), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60),
}


def model_prices(model_id: str):
    """Price for a model id or dated snapshot (longest matching name wins)."""
    match = max((k for k in MODEL_PRICES if (model_id or "").startswith(k)), key=len, default="gpt-5.1")
    return MODEL_PRICES[match]

RESULTS_DIR = ROOT / "results"


class RssSampler:
    """Background thread sampling process RSS every few seconds, tracking peak."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / 2**20
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_rss_mb


class SimClient:
    """Stands in for the OpenAI client in --sim mode. Orders the current
    presentation by the model's own cached price estimate times per-call
    lognormal noise (deterministic per presentation), and reports token usage
    scaled from Round-1 averages."""

    def __init__(self, estimates, default_price, prompt_per_call, completion_per_call, noise=0.15):
        self.estimates = estimates
        self.default_price = default_price
        self.prompt_per_call = prompt_per_call
        self.completion_per_call = completion_per_call
        self.noise = noise
        self._local = threading.local()  # current presentation, per thread
        self.chat = self
        self.completions = self

    def create(self, **_kw):
        pres = self._local.presentation
        rng = random.Random(zlib.crc32("|".join(pres).encode()))
        est = {
            i: self.estimates.get(a, self.default_price) * rng.lognormvariate(0, self.noise)
            for i, a in enumerate(pres, 1)
        }
        order = sorted(est, key=lambda i: est[i])
        # Order only: simulated prices would be cached as "estimates" and make a
        # resumed sim answer differently from an uninterrupted one.
        text = json.dumps({"order": order})
        scale = len(pres) / M
        usage = type("U", (), {
            "prompt_tokens": int(self.prompt_per_call * scale),
            "completion_tokens": int(self.completion_per_call * scale),
        })()
        msg = type("Msg", (), {"content": text})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": usage})()


CODE_FILES = (
    "run_amazon.py", "loaders/amazon.py", "tbs/resumable_sorter.py", "tbs/sorter.py",
    "tbs/transitive.py", "tbs/partition.py", "tbs/reach_index.py",
    "tbs/scalable_round3.py", "tbs/rank_aggregation.py",
)


def check_run_config(run_dir: Path, cache_path: Path, config: dict) -> None:
    """Refuse to mix runs: the cache is keyed by item list only, so answers
    from another prompt / sample / batch size would be silently reused."""
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        old.setdefault("model", "gpt-5.1")
        old.setdefault("shuffle_presentation", False)
        diff = {k: (old.get(k), v) for k, v in config.items()
                if old.get(k) != v and not (k == "model" and {old.get(k), v} <= {"gpt-5.1", "gpt-5.1-2025-11-13"})}
        if diff:
            raise SystemExit(f"run_config mismatch in {run_dir.name} (stored, requested): {diff}. "
                             "Use a different run name.")
        return
    if cache_path.exists():
        # Older run directories have no config: infer the prompt from the cache.
        with cache_path.open("r", encoding="utf-8") as f:
            first = next((json.loads(l) for l in f if l.strip()), None)
        if first is not None:
            # Price estimates are only produced by the rescued prompts; a cache
            # without them may predate price capture, so it proves nothing.
            stored = first.get("prompt_key") or ("rescued" if first.get("prices") else None)
            if stored is not None and stored != config["prompt"]:
                raise SystemExit(f"{run_dir.name} cache was made with prompt '{stored}', "
                                 f"not '{config['prompt']}'. Use a different run name.")
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def snapshot_code(run_dir: Path) -> None:
    """Copy the exact code used into the run directory (no git here)."""
    dest = run_dir / "code_snapshot" / time.strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        if (ROOT / name).exists():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest / name)


def keep_system_awake() -> None:
    """Ask Windows not to sleep while this process runs (released on exit)."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    global M, PROMPT_KEY, SEED, THETA
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", nargs="?", default="n2000_validation")
    ap.add_argument("n", nargs="?", type=int, default=2000)
    ap.add_argument("--backend", choices=("dag", "chain", "sets"), default="dag")
    ap.add_argument(
        "--theta",
        type=float,
        default=THETA,
        help="Round-3 completion threshold (1.0 = full diagonals+packing, no early stop)",
    )
    ap.add_argument("--max-live-calls", type=int, default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--tau-stop-threshold", type=float, default=TAU_STOP_THRESHOLD)
    ap.add_argument("--prompt", choices=tuple(PROMPTS), default=PROMPT_KEY)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strict-complete", action="store_true",
                    help="re-ask any call whose answer omits items (up to --max-attempts); never drop or fill in")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--round2-all-rows", action="store_true",
                    help="Round 2 on every grid row (default: R0 only)")
    ap.add_argument("--r2-workers", type=int, default=1,
                    help="run independent Round-2 merges concurrently (same results as 1)")
    ap.add_argument("--round3-mode", choices=("diagonals_then_unresolved", "diagonals"),
                    default="diagonals_then_unresolved",
                    help="diagonals = stop Round 3 after diagonal stitching (no remaining packing)")
    ap.add_argument("--finalize-at-cache-end", action="store_true",
                    help="with --sim --max-live-calls 0: when greedy packing reaches a batch with no "
                         "cached answer, stop there and write final results for the state reached")
    ap.add_argument("--greedy-parallel", type=int, default=1,
                    help="greedy packing: key-disjoint batches selected per degree count and "
                         "sent concurrently (1 = one batch per count, the original schedule)")
    ap.add_argument("--r3-prefetch", type=int, default=0,
                    help="fetch up to this many upcoming Round-3 batches concurrently (same decisions as 0)")
    ap.add_argument("--stop-after-round1", action="store_true",
                    help="run (or replay) Round 1 only, then exit; rerun without it to continue")
    ap.add_argument("--shuffle-presentation", action="store_true",
                    help="show each batch's items in a deterministic shuffled order (removes position bias)")
    ap.add_argument("--model-id", default="gpt-5-nano-2025-08-07",
                    help="exact OpenAI model id (paper: gpt-5-nano-2025-08-07)")
    args = ap.parse_args()
    M, PROMPT_KEY, SEED, THETA = args.m, args.prompt, args.seed, args.theta
    run_name, n = args.run_name, args.n
    full_pack = THETA >= 1.0 - 1e-12

    run_dir = RESULTS_DIR / f"amazon_1m_run_{run_name}"
    if args.sim:
        real_cache = run_dir / "call_cache.jsonl"
        run_dir = RESULTS_DIR / f"amazon_1m_run_{run_name}_sim"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "call_cache.jsonl").exists() and real_cache.exists():
            shutil.copyfile(real_cache, run_dir / "call_cache.jsonl")
            real_log = real_cache.with_name("greedy_log.jsonl")
            if real_log.exists():
                shutil.copyfile(real_log, run_dir / "greedy_log.jsonl")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "call_cache.jsonl"
    progress_path = run_dir / "progress.json"
    final_path = run_dir / "final_result.json"
    check_run_config(run_dir, cache_path, {"prompt": PROMPT_KEY, "n": n, "m": M, "seed": SEED,
                                           "model": args.model_id,
                                           "shuffle_presentation": args.shuffle_presentation})
    if not args.sim:
        snapshot_code(run_dir)
        keep_system_awake()

    print(f"=== run_name={run_name} n={n} m={M} backend={args.backend} sim={args.sim} ===", flush=True)
    print(f"cache={cache_path}", flush=True)

    sample = load_sample(n, SEED)
    text_map = dict(zip(sample.parent_asin, sample.display))
    if PROMPT_KEY.startswith("explicit"):
        # The ground-truth price is part of what the LLM sees (explicit sorting).
        text_map = {a: explicit_display(t, p) for a, t, p in zip(sample.parent_asin, sample.display, sample.price)}
    gt_rank = dict(zip(sample.parent_asin, sample.gt_rank))
    price = dict(zip(sample.parent_asin, sample.price))
    items = list(sample.parent_asin)
    random.Random(SEED).shuffle(items)
    del sample

    _true_order_cache = {}

    def true_order_fn():
        if "v" not in _true_order_cache:
            _true_order_cache["v"] = sorted(items, key=lambda a: gt_rank[a])
        return _true_order_cache["v"]

    SorterCls = make_resumable_sorter_class(
        cache_path=cache_path,
        true_order_fn=true_order_fn,
        price_fn=lambda: price,
        r1_workers=R1_WORKERS,
        tau_check_interval=TAU_CHECK_INTERVAL,
        tau_stop_threshold=args.tau_stop_threshold,
        tau_stop_consecutive=TAU_STOP_CONSECUTIVE,
        progress_path=progress_path,
        max_live_calls=args.max_live_calls,
        prompt_key=PROMPT_KEY,
        strict_complete=args.strict_complete,
        max_attempts=args.max_attempts,
        stop_after_round1=args.stop_after_round1,
        r2_workers=args.r2_workers,
        shuffle_seed=SEED if args.shuffle_presentation else None,
    )

    cached_count, cached_prompt, cached_completion = SorterCls.load_cache()
    print(
        f"loaded cache: {cached_count} entries, {cached_prompt} prompt tok, "
        f"{cached_completion} completion tok",
        flush=True,
    )

    if args.sim:
        est = dict(SorterCls._price_estimates)
        vals = sorted(est.values())
        client = SimClient(
            est,
            default_price=vals[len(vals) // 2] if vals else 20.0,
            prompt_per_call=cached_prompt / max(1, cached_count),
            completion_per_call=cached_completion / max(1, cached_count),
        )
        model_id = "sim"

        class Cls(SorterCls):
            def _sort_with_llm_once(self, presentation, round_id, payload=None, **kw):
                client._local.presentation = list(presentation)
                return super()._sort_with_llm_once(presentation, round_id, payload, **kw)

        SorterCls = Cls
    else:
        client, model_id = make_client_for_id(args.model_id)
    SorterCls.prompt_template = PROMPTS[PROMPT_KEY]

    sorter = SorterCls(
        items=items, batch_size=M, client=client, model=model_id,
        verbose=False, sort_by="alphabetical", text_map=text_map,
        completion_threshold=THETA,
        enable_completion_early_stop=not full_pack,
        enable_stall_early_stop=not full_pack,
        round1_partition="contiguous", prompt_order="input",
        round2_mode="recursive", round2_all_rows=args.round2_all_rows,
        round3_mode=args.round3_mode, graph_backend=args.backend,
    )
    if isinstance(sorter.graph, DagTransitiveOrder):
        sorter.graph.block = args.block
        # Any valid topological order gives identical answers; starting from
        # the model's own price estimates just means fewer reorders later.
        sorter.graph.seed_order(price if PROMPT_KEY.startswith("explicit") else dict(SorterCls._price_estimates))
    sorter.r3_prefetch = args.r3_prefetch
    sorter.greedy_parallel = args.greedy_parallel
    if args.finalize_at_cache_end:
        sorter.greedy_stop_exceptions = (LiveCallBudgetError,)
    # Seed stats with whatever was already paid for in prior (interrupted) runs.
    sorter.stats.count = cached_count
    sorter.stats.prompt_tokens = cached_prompt
    sorter.stats.completion_tokens = cached_completion

    rss_sampler = RssSampler(interval_s=5.0).start()
    tracemalloc.start()
    t0 = time.time()
    try:
        predicted, run_stats = sorter.sort()
    except TauTooLowError as e:
        print(f"\n!!! HALTED: {e}", flush=True)
        final_path.write_text(json.dumps({"halted": True, "reason": str(e)}, indent=2), encoding="utf-8")
        return
    except StopAfterRound1 as e:
        print(f"=== STOPPED AFTER ROUND 1: {e}", flush=True)
        return
    except IncompleteAnswerError as e:
        print(f"\n!!! INCOMPLETE ANSWER: {e}", flush=True)
        return
    except LiveCallBudgetError as e:
        print(f"\n!!! BUDGET: {e}", flush=True)
        return
    except KeyboardInterrupt:
        print("\n!!! INTERRUPTED by user. Cache is safe on disk - rerun this exact command to resume.", flush=True)
        return
    except Exception as e:
        print(f"\n!!! ERROR: {e}", flush=True)
        print("Cache is safe on disk - rerun this exact command to resume.", flush=True)
        raise

    elapsed = time.time() - t0
    _, traced_peak_mb = tracemalloc.get_traced_memory()
    traced_peak_mb = round(traced_peak_mb / 2**20, 1)
    tracemalloc.stop()
    peak_rss_mb = round(rss_sampler.stop(), 1)

    true_order = true_order_fn()
    tau_final = fast_kendalls_tau(predicted, true_order)
    pred_pos = {a: i for i, a in enumerate(predicted)}
    tau_b_final = kendalltau([pred_pos[a] for a in items], [price[a] for a in items]).statistic
    known_pairs, total_pairs, rho = sorter.known_pair_counts()
    g = sorter.graph
    overshoot = getattr(sorter, "round3_overshoot", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "batches": 0})

    price_in, price_out = model_prices(args.model_id or "gpt-5.1")

    def cost(prompt_tokens, completion_tokens):
        return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out

    # Per-round paid calls from the cache (the sorter's round counters only see
    # calls made in this process, so they read 0 for rounds replayed on resume).
    paid_by_round = {1: 0, 2: 0, 3: 0}
    prices_by_key = {}
    round_by_key = {}
    missing_by_key = {}
    omissions = {"calls_logged": 0, "calls_with_missing_ids": 0, "missing_ids": 0}
    models_served = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rid = rec.get("round_id")
                if rid in paid_by_round:
                    paid_by_round[rid] += 1
                prices_by_key[tuple(rec["presentation"])] = rec.get("prices", {})
                round_by_key[tuple(rec["presentation"])] = rid
                missing_by_key[tuple(rec["presentation"])] = rec.get("n_missing", 0)
                if "n_missing" in rec:
                    omissions["calls_logged"] += 1
                    omissions["missing_ids"] += rec["n_missing"]
                    omissions["calls_with_missing_ids"] += rec["n_missing"] > 0
                ms = rec.get("model_served")
                if ms:
                    models_served[ms] = models_served.get(ms, 0) + 1

    # Alternative final orders from exactly the calls this run consumed
    # (see rank_aggregation.py): the first-come closure lets one noisy early
    # judgment propagate through every transitive path, so at large n its
    # linear extension degrades toward random; these weigh all judgments.
    # Sorted so the fit does not depend on the order concurrent calls finished in.
    # A batch whose call never completed (stopped by budget / incomplete answer)
    # is recorded as requested but has no answer: it is not evidence.
    used = sorted(k for k in SorterCls._used_keys if k in SorterCls._cache)
    used_orders = [SorterCls._cache[k] for k in used]
    pl_scores = plackett_luce_scores(items, used_orders)
    pl_order = order_from_scores(items, pl_scores, ascending_score=False)
    # Same fit, but ids the model omitted (appended in input order by the
    # parser) are treated as unranked instead of "most expensive".
    pl_clean_scores = plackett_luce_scores(
        items, (o[: len(o) - missing_by_key.get(k, 0)] for k, o in zip(used, used_orders)))
    pl_clean_order = order_from_scores(items, pl_clean_scores, ascending_score=False)
    po_scores = price_offset_scores(items, (prices_by_key.get(k, {}) for k in used))
    po_order = order_from_scores(items, po_scores, ascending_score=True)
    # Baseline: Round-1 calls only (each item priced once, no cross-batch linking).
    r1_scores = price_offset_scores(items, (prices_by_key.get(k, {}) for k in used if round_by_key.get(k) == 1),
                                    iters=1, shrink=1e18)
    r1_order = order_from_scores(items, r1_scores, ascending_score=True)

    def taus(order):
        pos = {a: i for i, a in enumerate(order)}
        return {
            "tau_a_vs_gt_rank": round(fast_kendalls_tau(order, true_order), 4),
            "tau_b_vs_price": round(float(kendalltau([pos[a] for a in items], [price[a] for a in items]).statistic), 4),
        }

    aggregates = {
        "closure_linear_extension": {"tau_a_vs_gt_rank": round(tau_final, 4), "tau_b_vs_price": round(float(tau_b_final), 4)},
        "plackett_luce": taus(pl_order),
        "plackett_luce_omissions_dropped": taus(pl_clean_order),
        "price_estimates_with_call_offsets": taus(po_order),
        "price_estimates_round1_only": taus(r1_order),
        "calls_used": len(used),
    }
    print("final-order methods:", json.dumps(aggregates), flush=True)

    paid_cost = cost(run_stats.prompt_tokens, run_stats.completion_tokens)
    overshoot_cost = cost(overshoot["prompt_tokens"], overshoot["completion_tokens"])
    transitive = known_pairs - g.direct_relations

    summary = {
        "run_name": run_name, "n": n, "m": M, "theta": THETA, "seed": SEED,
        "model": "GPT-5.1", "prompt_key": PROMPT_KEY, "sim": args.sim,
        "round1_partition": "contiguous", "round2_mode": "recursive", "round2_all_rows": args.round2_all_rows,
        "round3_mode": args.round3_mode, "graph_backend": args.backend,
        "r1_workers": R1_WORKERS,
        "tau_a_vs_gt_rank": round(tau_final, 4),
        "tau_b_vs_price": round(float(tau_b_final), 4),
        "final_order_methods": aggregates,
        "omitted_ids_in_llm_output": omissions,
        "models_served": models_served,
        "model_requested": model_id,
        "strict_complete": args.strict_complete,
        "r2_workers": args.r2_workers,
        "r3_prefetch": getattr(sorter, "r3_prefetch_stats", {"window": args.r3_prefetch}),
        "calls_used_by_algorithm": len(used),
        "shuffle_presentation": args.shuffle_presentation,
        # Algorithm calls exclude Round-3 overshoot (calls made past the theta
        # stop before the lazy exact check caught it; see scalable_round3.py).
        "calls": run_stats.count - overshoot["calls"],
        "prompt_tokens": run_stats.prompt_tokens - overshoot["prompt_tokens"],
        "completion_tokens": run_stats.completion_tokens - overshoot["completion_tokens"],
        "estimated_cost_usd": round(paid_cost - overshoot_cost, 2),
        "paid_calls_including_overshoot": run_stats.count,
        "paid_cost_usd_including_overshoot": round(paid_cost, 2),
        "round3_overshoot": overshoot,
        "known_pairs": known_pairs, "total_pairs": total_pairs, "rho": round(rho, 4),
        "direct_relations": g.direct_relations, "transitive_relations": transitive,
        "leverage_ratio": round(transitive / g.direct_relations, 2) if g.direct_relations else None,
        "rejected_contradictions": g.rejected_contradictions_count,
        "rejected_pair_flips": g.rejected_pair_flips_count,
        "rejected_transitivity_cycles": g.rejected_transitivity_cycles_count,
        "round1_calls": paid_by_round[1], "round2_calls": paid_by_round[2],
        "round3_calls": paid_by_round[3] - overshoot["calls"],
        "chain_stats": dict(sorter.chain_stats),
        "dag_edges": getattr(g, "num_edges", None),
        "round3_stats": dataclasses.asdict(sorter.round3_stats),
        "timing_s": dict(SorterCls._timing),
        "elapsed_s_this_process": round(elapsed, 1),
        "peak_traced_python_memory_mb": traced_peak_mb,
        "peak_process_rss_mb": peak_rss_mb,
        "tau_checkpoint_log": SorterCls._tau_log,
    }
    final_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== DONE === wrote {final_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # Full sorted result: id, title, predicted rank, ground-truth price and rank.
    pl_rank = {a: i for i, a in enumerate(pl_order)}
    po_rank = {a: i for i, a in enumerate(po_order)}
    sorted_result = [
        {
            "predicted_rank": i,
            "plackett_luce_rank": pl_rank[a],
            "price_offset_rank": po_rank[a],
            "parent_asin": a,
            "display": text_map[a],
            "true_price": price[a],
            "true_gt_rank": gt_rank[a],
            "estimated_price": SorterCls._price_estimates.get(a),
        }
        for i, a in enumerate(predicted)
    ]
    sorted_result_path = run_dir / "sorted_result.json"
    sorted_result_path.write_text(json.dumps(sorted_result, indent=1), encoding="utf-8")
    print(f"wrote sorted result ({len(sorted_result)} items) to {sorted_result_path}", flush=True)

    pl_rows = sorted(sorted_result, key=lambda r: r["plackett_luce_rank"])
    pl_path = run_dir / "sorted_result_plackett_luce.json"
    pl_path.write_text(json.dumps(pl_rows, indent=1), encoding="utf-8")
    print(f"wrote Plackett-Luce sorted result ({len(pl_rows)} items) to {pl_path}", flush=True)

    price_estimates_path = run_dir / "price_estimates.json"
    price_estimates_path.write_text(
        json.dumps(dict(SorterCls._price_estimates), indent=1), encoding="utf-8"
    )
    print(
        f"wrote {len(SorterCls._price_estimates)} price estimates to {price_estimates_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
