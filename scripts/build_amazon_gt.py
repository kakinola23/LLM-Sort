"""
Build the Amazon products 1M global-ranking ground truth (price ascending).

Source: Amazon Reviews 2023 product metadata (McAuley Lab), pre-filtered to priced
products by scripts/fetch_amazon_priced.py into data/amazon/priced/<Category>.csv.gz.

Ranking: price ascending, ties broken alphabetically by display text, then parent_asin
(same convention as NBA heights). The LLM only sees ``display`` (the product title,
shortened at a word boundary), never the price, so it has to infer price from the title.

Cleaning:
  * one row per parent_asin (a product can be listed under several categories);
  * one row per display text, so the LLM never sees two identical items with
    different ground-truth prices;
  * --min-price / --max-price drop placeholder and outlier listings.

Output: <out>.csv and <out>_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "amazon"
PRICED_DIR = DATA_DIR / "priced"


def shorten(title: str, limit: int) -> str:
    if len(title) <= limit:
        return title
    cut = title[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-|/(")


def tie_stats(key: pd.Series) -> dict:
    n = len(key)
    counts = key.value_counts()
    total_pairs = n * (n - 1) // 2
    tied_pairs = int((counts * (counts - 1) // 2).sum())
    return {
        "n": n,
        "distinct_key_values": int(counts.size),
        "largest_tie_group": int(counts.max()),
        "largest_tie_value": float(counts.idxmax()),
        "tied_pair_fraction": tied_pairs / total_pairs,
        "max_tau_b_vs_key": math.sqrt(1 - tied_pairs / total_pairs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-price", type=float, default=0.5)
    ap.add_argument("--max-price", type=float, default=10_000.0)
    ap.add_argument("--min-title-words", type=int, default=3)
    ap.add_argument("--max-title-chars", type=int, default=150)
    ap.add_argument("--out", default=str(DATA_DIR / "amazon_1m_price_gt.csv"))
    args = ap.parse_args()

    files = sorted(PRICED_DIR.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no priced files in {PRICED_DIR}; run fetch_amazon_priced.py first")
    frames = []
    for f in files:
        part = pd.read_csv(f, dtype={"parent_asin": str, "title": str, "store": str})
        part["source_category"] = f.name.removesuffix(".csv.gz")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    funnel = {"priced_rows": int(len(df))}

    df = df.drop_duplicates("parent_asin")
    funnel["unique_asin"] = int(len(df))
    df = df[(df.price >= args.min_price) & (df.price <= args.max_price)]
    funnel["price_in_range"] = int(len(df))
    df = df[df.title.str.split().str.len() >= args.min_title_words]
    funnel["title_min_words"] = int(len(df))
    df["display"] = df.title.map(lambda t: shorten(t, args.max_title_chars))
    df = df.drop_duplicates("display", keep=False)
    funnel["unique_display"] = int(len(df))

    if len(df) < args.n:
        print(f"WARNING: only {len(df):,} eligible products (< {args.n:,}); using all of them")
    else:
        df = df.sample(n=args.n, random_state=args.seed)

    df = df.sort_values(["price", "display", "parent_asin"], kind="mergesort").reset_index(drop=True)
    df["gt_rank"] = range(1, len(df) + 1)

    cols = ["gt_rank", "parent_asin", "display", "price", "source_category",
            "main_category", "store", "average_rating", "rating_number"]
    df[cols].to_csv(args.out, index=False)

    q = df.price.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2)
    stats = {
        **tie_stats(df.price),
        "funnel": funnel,
        "seed": args.seed,
        "price_min": float(df.price.min()),
        "price_quantiles": {str(k): float(v) for k, v in q.items()},
        "price_max": float(df.price.max()),
        "top_price_points": {str(k): int(v) for k, v in df.price.value_counts().head(10).items()},
        "display_chars_median": float(df.display.str.len().median()),
        "display_chars_mean": float(df.display.str.len().mean()),
        "source_categories": df.source_category.value_counts().to_dict(),
        "out": args.out,
    }
    Path(args.out).with_name(Path(args.out).stem + "_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(df[cols[:5]].head(5).to_string(index=False))
    print(df[cols[:5]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
