"""
Order items from all the listwise LLM judgments collected during a TBS run,
instead of from the first-come transitive closure.

The closure accepts each relation in arrival order and rejects later evidence
that contradicts it, so one noisy early judgment propagates through every
transitive path it lies on. These aggregators weigh every judgment against
all the others:

  * plackett_luce_scores: maximum-likelihood Plackett-Luce worths from every
    returned ordering (Hunter 2004 MM algorithm, with a weak prior that ties
    each item to a virtual reference so worths stay finite and identified).
  * price_offset_scores: per-item log-price level from each call's stated
    price estimates, with a per-call offset absorbing batch-level calibration
    drift (alternating least squares).

Orders are ascending (index 0 = cheapest / "greatest" in TBS terms), matching
the LLM output convention used by the sorters.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit


def _flatten(orders: Sequence[Sequence[str]], index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(orders) + 1, np.int64)
    for j, o in enumerate(orders):
        offsets[j + 1] = offsets[j] + len(o)
    flat = np.empty(offsets[-1], np.int64)
    k = 0
    for o in orders:
        for x in o:
            flat[k] = index[x]
            k += 1
    return flat, offsets


@njit(cache=True)
def _pl_mm(flat, offsets, n, iters, prior, tol):
    """Plackett-Luce MM. Each list is best-first; worth w_i > 0.
    Prior: one win and one loss against a virtual item of worth 1."""
    w = np.ones(n)
    wins = np.zeros(n)
    nl = offsets.shape[0] - 1
    for j in range(nl):
        a, b = offsets[j], offsets[j + 1]
        for t in range(a, b - 1):
            wins[flat[t]] += 1.0
    wins += prior
    denom = np.zeros(n)
    for it in range(iters):
        denom[:] = 0.0
        for j in range(nl):
            a, b = offsets[j], offsets[j + 1]
            k = b - a
            if k < 2:
                continue
            # suffix sums of worths: S_t = sum_{s >= t} w
            suf = np.empty(k)
            s = 0.0
            for t in range(k - 1, -1, -1):
                s += w[flat[a + t]]
                suf[t] = s
            # running sum of 1/S_t over choice stages t = 0..k-2
            acc = 0.0
            for t in range(k):
                if t < k - 1:
                    acc += 1.0 / suf[t]
                denom[flat[a + t]] += acc
        # prior pairwise games against the virtual item (worth 1)
        for i in range(n):
            denom[i] += 2.0 * prior / (w[i] + 1.0)
        maxrel = 0.0
        for i in range(n):
            nw = wins[i] / denom[i]
            rel = abs(math.log(nw) - math.log(w[i]))
            if rel > maxrel:
                maxrel = rel
            w[i] = nw
        if maxrel < tol:
            break
    return np.log(w), it + 1


def plackett_luce_scores(
    items: Sequence[str],
    orders: Iterable[Sequence[str]],
    iters: int = 500,
    prior: float = 0.1,
    tol: float = 1e-6,
) -> np.ndarray:
    """Log-worths, higher = earlier in the ascending orders (cheaper)."""
    index = {x: i for i, x in enumerate(items)}
    orders = [list(o) for o in orders if len(o) >= 2]
    flat, offsets = _flatten(orders, index)
    scores, _ = _pl_mm(flat, offsets, len(items), iters, prior, tol)
    return scores


def price_offset_scores(
    items: Sequence[str],
    calls: Iterable[Dict[str, float]],
    iters: int = 50,
    shrink: float = 1.0,
) -> np.ndarray:
    """Estimated log-price level per item (higher = pricier). `calls` are
    {item: stated price} dicts, one per LLM call. Items never priced get the
    global mean."""
    index = {x: i for i, x in enumerate(items)}
    ii: List[int] = []
    cc: List[int] = []
    yy: List[float] = []
    for c, prices in enumerate(calls):
        for x, p in prices.items():
            if x in index and p is not None and p > 0:
                ii.append(index[x])
                cc.append(c)
                yy.append(math.log(p))
    if not ii:
        return np.zeros(len(items))
    i_arr = np.array(ii)
    c_arr = np.array(cc)
    y = np.array(yy)
    n, ncall = len(items), int(c_arr.max()) + 1
    mu = y.mean()
    theta = np.zeros(n)
    off = np.zeros(ncall)
    cnt_i = np.bincount(i_arr, minlength=n).astype(float)
    cnt_c = np.bincount(c_arr, minlength=ncall).astype(float)
    for _ in range(iters):
        r = y - mu - off[c_arr]
        theta = np.bincount(i_arr, weights=r, minlength=n) / (cnt_i + 1e-12)
        r = y - mu - theta[i_arr]
        # shrink offsets toward 0 so a call's offset needs several items to move
        off = np.bincount(c_arr, weights=r, minlength=ncall) / (cnt_c + shrink)
    theta[cnt_i == 0] = 0.0
    return mu + theta


def order_from_scores(items: Sequence[str], scores: np.ndarray, ascending_score: bool) -> List[str]:
    """Sort items by score (ties by name). ascending_score=False puts the
    highest score first."""
    key = scores if ascending_score else -scores
    idx = sorted(range(len(items)), key=lambda i: (key[i], items[i]))
    return [items[i] for i in idx]
