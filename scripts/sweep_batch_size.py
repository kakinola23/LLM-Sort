#!/usr/bin/env python3
"""Batch-size sweep: run TBS for every m in a grid (default {4,...,14}) on one dataset and model.

  python scripts/sweep_batch_size.py --dataset nba --model GPT-4o-mini --extra "--structured-schema recall"
  python scripts/sweep_batch_size.py --dataset movielens --model GPT-4o --grid 4-14

Each m is an ordinary runner call, so it is resumable and cached; results land in
results/sweep/<dataset>/<model>/. Select m afterwards with scripts/select_batch_size.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {"nba": "run_nba.py", "population": "run_population.py", "dl19": "run_dl19.py",
           "scifact": "run_scifact.py", "movielens": "run_movielens.py"}


def parse_grid(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=tuple(RUNNERS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", default="4-14", help="e.g. 4-14 or 4,6,8,10,12")
    ap.add_argument("--extra", default="", help="extra runner flags, quoted")
    args = ap.parse_args()

    tag = args.model.replace(".", "_").replace("-", "_")
    out = ROOT / "results" / "sweep" / args.dataset / tag
    for m in parse_grid(args.grid):
        cmd = [sys.executable, RUNNERS[args.dataset], "--method", "tbs", "--model", args.model, "-m", str(m)]
        if args.dataset in ("nba", "population"):
            cmd += ["--out", str(out / f"m{m}.json")]
        else:
            cmd += ["--out-dir", str(out)]
        cmd += shlex.split(args.extra)
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
