"""Batch-size selection: geometric knee and endpoint-aware utility (Section V-C)."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


def knee_index_calls_tau(calls: List[float], taus: List[float]) -> int:
    """Points sorted by calls ascending; knee = index of max perpendicular distance to chord (interior)."""
    n = len(calls)
    if n < 3:
        return max(0, n // 2)
    x0, y0 = float(calls[0]), float(taus[0])
    x1, y1 = float(calls[-1]), float(taus[-1])
    dx, dy = x1 - x0, y1 - y0
    L_sq = dx * dx + dy * dy + 1e-12
    best_i, best_d = 1, -1.0
    for i in range(1, n - 1):
        px, py = float(calls[i]), float(taus[i])
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L_sq))
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        d = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def select_with_endpoint_check(
    rows_sorted_by_calls: List[Dict[str, Any]],
    knee_idx: int,
    alpha: float = 0.7,
) -> Tuple[int, str]:
    """
    Compare elbow-knee with endpoints using a normalized utility score.

    Utility:
      U = alpha * tau_norm + (1-alpha) * (1 - calls_norm)

    Returns:
      (selected_idx, selection_reason)
      selection_reason in {"knee", "endpoint_min_calls", "endpoint_max_calls"}.
      Endpoints are along the sweep sorted by API calls (ascending): index 0 is
      fewest calls (not necessarily largest batch_size), last index is most calls.
    """
    n = len(rows_sorted_by_calls)
    if n == 0:
        return 0, "knee"
    if n == 1:
        return 0, "knee"

    alpha = max(0.0, min(1.0, float(alpha)))
    calls = [float(r["api_calls"]) for r in rows_sorted_by_calls]
    taus = [float(r["kendalls_tau"]) for r in rows_sorted_by_calls]
    c_min, c_max = min(calls), max(calls)
    t_min, t_max = min(taus), max(taus)
    c_span = (c_max - c_min) if c_max != c_min else 1.0
    t_span = (t_max - t_min) if t_max != t_min else 1.0

    def score(i: int) -> float:
        c_norm = (calls[i] - c_min) / c_span
        t_norm = (taus[i] - t_min) / t_span
        return alpha * t_norm + (1.0 - alpha) * (1.0 - c_norm)

    first_idx, last_idx = 0, n - 1
    knee_idx = max(0, min(n - 1, knee_idx))
    s_knee = score(knee_idx)
    s_first = score(first_idx)
    s_last = score(last_idx)
    endpoint_idx = first_idx if s_first >= s_last else last_idx
    s_endpoint = max(s_first, s_last)

    if s_endpoint > s_knee:
        if endpoint_idx == first_idx:
            return endpoint_idx, "endpoint_min_calls"
        return endpoint_idx, "endpoint_max_calls"
    return knee_idx, "knee"


def _is_endpoint_selection_reason(reason: Optional[str]) -> bool:
    """True if selection picked an endpoint of the calls-sorted sweep (not the interior knee)."""
    return reason in ("endpoint_override", "endpoint_min_calls", "endpoint_max_calls")



def call_save_select(ms: List[int], calls: List[float], threshold: float = 0.05) -> int:
    """Call-save rule (used when quality is flat in m): the smallest m after which moving to the
    next larger m saves less than ``threshold`` of calls. ``ms`` ascending; returns the chosen m."""
    for i in range(len(ms) - 1):
        if (calls[i] - calls[i + 1]) / calls[i] < threshold:
            return ms[i]
    return ms[-1]
