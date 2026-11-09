#!/usr/bin/env python3
"""Build the MovieLens Phase-1 task manifest (63 users x 70 movies) from the raw ML-1M dump.

Downloads https://files.grouplens.org/datasets/movielens/ml-1m.zip (GroupLens; see their license),
joins ratings with movie titles/genres in ratings.dat order, then for each of 63 users sampled with
seed 42 (among users with >= 70 ratings) draws 70 rated movies as candidates; the remaining ratings
form the profile (top 12 by rating). Writes data/movielens/movielens_phase1_manifest.json.

  python scripts/prepare_movielens.py
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

try:  # use certifi's CA bundle when the system store lacks the issuer (common on Windows Python)
    import ssl

    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "movielens"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

NUM_USERS, NUM_MOVIES, SEED, MAX_HISTORY = 63, 70, 42, 12


def load_ml1m() -> pd.DataFrame:
    raw = OUT_DIR / "ml-1m"
    if not (raw / "ratings.dat").is_file():
        print(f"downloading {URL} ...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(OUT_DIR)
    ratings = pd.read_csv(raw / "ratings.dat", sep="::", engine="python", encoding="latin-1",
                          names=["user_id", "movie_id", "ground_truth_rating", "timestamp"])
    movies = pd.read_csv(raw / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["movie_id", "movie_title", "genres"])
    return ratings.merge(movies, on="movie_id", how="left")  # keeps ratings.dat order


def movie_key(movie_id: int) -> str:
    return f"m_{movie_id}"


def select_users(df: pd.DataFrame) -> List[int]:
    counts = df.groupby("user_id").size()
    eligible = sorted(counts[counts >= NUM_MOVIES].index.astype(int).tolist())
    return sorted(int(u) for u in pd.Series(eligible).sample(n=NUM_USERS, random_state=SEED, replace=False))


def profile_text(history: pd.DataFrame) -> str:
    if history.empty:
        return "No prior viewing history available."
    hist = history.sort_values(["ground_truth_rating", "timestamp"], ascending=[False, True]).head(MAX_HISTORY)
    lines = [f"- {r['movie_title']} ({int(r['ground_truth_rating'])}/5 {'★' * int(r['ground_truth_rating'])})"
             for _, r in hist.iterrows()]
    return "User viewing history (higher rating = stronger preference):\n" + "\n".join(lines)


def build_task(df: pd.DataFrame, user_id: int) -> Dict:
    user_df = df[df["user_id"] == user_id].copy()
    sampled = user_df.sample(n=NUM_MOVIES, random_state=SEED + user_id * 1000)
    held_out = user_df.drop(sampled.index)
    gt = sampled.sort_values(["ground_truth_rating", "timestamp", "movie_title"], ascending=[False, True, True])
    return {
        "user_id": int(user_id), "n_movies": NUM_MOVIES, "sample_seed": SEED + int(user_id) * 1000,
        "movie_ids": [movie_key(int(m)) for m in sampled.sort_values("movie_id")["movie_id"]],
        "ground_truth_order": [movie_key(int(m)) for m in gt["movie_id"]],
        "relevance_map": {movie_key(int(r.movie_id)): int(r.ground_truth_rating) for r in sampled.itertuples()},
        "display_map": {movie_key(int(r.movie_id)): f"{r.movie_title} [{r.genres}]" for r in sampled.itertuples()},
        "profile_text": profile_text(held_out),
    }


def main() -> None:
    df = load_ml1m()
    users = select_users(df)
    manifest = {"dataset": "MovieLens-1M", "phase": 1, "num_users": NUM_USERS, "n_movies": NUM_MOVIES,
                "seed": SEED, "user_ids": users, "tasks": [build_task(df, u) for u in users]}
    out = OUT_DIR / "movielens_phase1_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(users)} users)")


if __name__ == "__main__":
    sys.exit(main())
