"""NBA player heights (n=200, OrderByLLM's ``nba_heights_200.csv``).

Items are player names; the model ranks them by height from its own knowledge (names only).
Ground truth: ascending height in meters, ties broken by name.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "nba" / "nba_heights_200.csv"


def load_heights(csv_path: Path = DATA_CSV) -> Tuple[List[str], Dict[str, float]]:
    names: List[str] = []
    height_map: Dict[str, float] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["full_name"])
            height_map[row["full_name"]] = float(row["h_meters"])
    return names, height_map


def load(seed: int = 0, shuffle: bool = True) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Return (input items, true order shortest->tallest, height map)."""
    names, height_map = load_heights()
    items = names[:]
    if shuffle:
        random.Random(seed).shuffle(items)
    true_order = sorted(names, key=lambda n: (height_map[n], n))
    return items, true_order, height_map
