#!/usr/bin/env python3
"""Select the batch size m from a sweep (Section V-C): geometric knee + endpoint-aware utility
(alpha = 0.7) on (calls, quality), and the call-save rule (< 5% calls saved by the next m).

  python scripts/select_batch_size.py results/sweep/nba/GPT_4o
  python scripts/select_batch_size.py results/sweep/movielens/GPT_4o --rule call-save
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tbs.batch_size import call_save_select, knee_index_calls_tau, select_with_endpoint_check  # noqa: E402


def load_points(sweep_dir: Path):
    pts = []
    for p in sweep_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("batch_size_m") or (int(re.search(r"_m(\d+)", p.name).group(1)) if re.search(r"_m(\d+)", p.name) else None)
        if m is None or "total_calls" not in d:
            continue
        q = d.get("tau", d.get("ndcg_full", d.get("ndcg")))
        pts.append((int(m), float(d["total_calls"]), float(q)))
    return sorted(pts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--rule", choices=("knee", "call-save"), default="knee")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    pts = load_points(Path(args.sweep_dir))
    if len(pts) < 2:
        raise SystemExit("need at least two sweep points")
    for m, c, q in pts:
        print(f"  m={m:3d} calls={c:9.0f} quality={q:.4f}")
    if args.rule == "call-save":
        ms, calls = [p[0] for p in pts], [p[1] for p in pts]
        print(f"call-save rule (<{args.threshold:.0%}): m={call_save_select(ms, calls, args.threshold)}")
        return
    rows = sorted(({"batch_size": m, "api_calls": c, "kendalls_tau": q} for m, c, q in pts), key=lambda r: r["api_calls"])
    k = knee_index_calls_tau([r["api_calls"] for r in rows], [r["kendalls_tau"] for r in rows])
    s, reason = select_with_endpoint_check(rows, k, alpha=args.alpha)
    print(f"knee: m={rows[k]['batch_size']}   selected: m={rows[s]['batch_size']} ({reason})")


if __name__ == "__main__":
    main()
