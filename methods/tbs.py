"""TBS method entry point: build a sorter for a dataset and run it with the paper's switches.

Default configuration (paper, Section V-A): contiguous Round 1, recursive R0 Round 2,
diagonals then Pack in Round 3, full packing (theta=1.0), transitive inference on,
OrderByLLM structured output. Every switch below is one ablation row in the paper.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Type

from tbs.metrics import kendalls_tau
from tbs.sorter import BatchedLLMSorter


def add_tbs_args(parser, required: bool = True) -> None:
    """CLI switches shared by every dataset runner (defaults = paper default TBS)."""
    g = parser.add_argument_group("TBS configuration")
    g.add_argument("--batch-size", "-m", type=int, required=required, help="listwise batch size m")
    g.add_argument("--theta", type=float, default=1.0, help="Round-3 completion threshold (1.0 = full packing; <1 = TBS-EarlyStop)")
    g.add_argument("--round1-partition", choices=("contiguous", "random", "similarity", "similarity_embed"), default="contiguous")
    g.add_argument("--round2-mode", choices=("recursive", "windows"), default="recursive")
    g.add_argument("--round2-all-rows", action="store_true", help="Round 2 over all rows (default: head row R0 only)")
    g.add_argument("--round3-mode", choices=("diagonals_then_unresolved", "diagonals", "unresolved", "unresolved_greedy"), default="diagonals_then_unresolved")
    g.add_argument("--no-transitivity", action="store_true", help="TBS-NoTrans: record only directly observed pairs")
    g.add_argument("--graph-backend", choices=("sets", "chain", "dag"), default="sets", help="dag = reachability index")


def tbs_kwargs(args) -> Dict[str, Any]:
    theta = args.theta
    return dict(
        batch_size=args.batch_size,
        completion_threshold=theta,
        enable_completion_early_stop=theta < 1.0,
        enable_stall_early_stop=theta < 1.0,
        round1_partition=args.round1_partition,
        round2_mode=args.round2_mode,
        round2_all_rows=args.round2_all_rows,
        round3_mode=args.round3_mode,
        use_transitivity=not args.no_transitivity,
        graph_backend=args.graph_backend,
    )


def run_tbs(
    sorter_cls: Type[BatchedLLMSorter],
    items: List[str],
    true_order: Optional[List[str]] = None,
    **sorter_kwargs: Any,
) -> Dict[str, Any]:
    """Sort ``items`` with TBS; return order, Kendall tau (if ground truth given), calls and tokens."""
    if str(sorter_kwargs.get("round1_partition", "")).startswith("similarity") and not sorter_kwargs.get("text_map"):
        sorter_kwargs["text_map"] = {x: x for x in items}  # group by item text only, never by true values
    sorter = sorter_cls(items=items, **sorter_kwargs)
    t0 = time.time()
    predicted, stats = sorter.sort()
    elapsed = time.time() - t0
    known, total, rho = sorter.known_pair_counts()
    out: Dict[str, Any] = {
        "n": len(items),
        "total_calls": stats.count,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "round1_calls": sorter.round1_calls,
        "round2_calls": sorter.round2_calls,
        "round3_calls": stats.count - sorter.round1_calls - sorter.round2_calls,
        "rho": round(rho, 6),
        "known_pairs": known,
        "total_pairs": total,
        "time_s": round(elapsed, 2),
        "stop_reason": getattr(getattr(sorter, "round3_stats", None), "stop_reason", ""),
        "predicted_order": list(predicted),
    }
    if true_order is not None:
        out["tau"] = round(kendalls_tau(predicted, true_order), 6)
    return out
