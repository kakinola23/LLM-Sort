"""
Stream Amazon Reviews 2023 product metadata (McAuley Lab, UCSD) category by
category and keep only products that have both a title and a price.

Output per category: data/amazon/priced/<Category>.csv.gz with columns
parent_asin, title, price, main_category, store, average_rating, rating_number.
A category whose output already exists is skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{}.jsonl.gz"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "amazon" / "priced"
# Raw dumps total ~14 GB; they are staged here and can be deleted afterwards (override with --raw-dir).
RAW_DIR = ROOT / "data" / "amazon" / "raw"

CATEGORIES = [
    "Electronics",
    "Home_and_Kitchen",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Sports_and_Outdoors",
    "Automotive",
    "Beauty_and_Personal_Care",
    "Cell_Phones_and_Accessories",
    "Office_Products",
    "Health_and_Household",
    "Pet_Supplies",
    "Patio_Lawn_and_Garden",
    "Arts_Crafts_and_Sewing",
    "Industrial_and_Scientific",
    "Baby_Products",
    "Musical_Instruments",
    "Video_Games",
    "Appliances",
    "Grocery_and_Gourmet_Food",
    "All_Beauty",
]

FIELDS = ["parent_asin", "title", "price", "main_category", "store", "average_rating", "rating_number"]


def _price(v) -> float | None:
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def fetch(category: str) -> dict:
    out = OUT_DIR / f"{category}.csv.gz"
    if out.exists():
        return {"category": category, "skipped": True}
    tmp = out.with_suffix(".tmp")
    raw = RAW_DIR / f"meta_{category}.jsonl.gz"
    seen = kept = 0
    start = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # The server drops long transfers mid-stream, so resume with -C - until curl exits cleanly.
    for attempt in range(200):
        rc = subprocess.run(["curl", "-sS", "-f", "-C", "-", "-o", str(raw),
                             BASE.format(category)]).returncode
        if rc == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"download failed after {attempt + 1} attempts")
    download_min = round((time.time() - start) / 60, 1)
    with gzip.open(raw, "rt", encoding="utf-8") as src, \
            gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(FIELDS)
        for line in src:
            seen += 1
            if seen % 500_000 == 0:
                print(f"  {category}: {seen:,} scanned, {kept:,} priced", flush=True)
            d = json.loads(line)
            price, title = _price(d.get("price")), (d.get("title") or "").strip()
            if price is None or not title:
                continue
            kept += 1
            writer.writerow([d.get("parent_asin"), " ".join(title.split()), price,
                             d.get("main_category"), d.get("store"),
                             d.get("average_rating"), d.get("rating_number")])
    tmp.rename(out)
    raw.unlink()
    return {"category": category, "scanned": seen, "priced": kept,
            "download_min": download_min, "minutes": round((time.time() - start) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "fetch_log.jsonl"
    total = 0
    for cat in args.categories:
        try:
            info = fetch(cat)
        except Exception as exc:  # keep going; a failed category can be retried later
            info = {"category": cat, "error": repr(exc)}
        total += info.get("priced", 0)
        print(json.dumps(info), f"running priced total={total:,}", flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
