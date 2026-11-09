"""
Transitive-order backend that scales to ~1M items.

ChainTransitiveOrder keeps two labels per item per chain, i.e. O(n*w) memory;
after Round 1 of a 1M-item run w ~= n/m = 13,334 chains, which is ~200 GB of
Python list slots. This backend stores only the accepted direct edges (a DAG,
a few million int32s) and answers every closure question from them:

  * is_greater / comparable: DFS over out-edges, pruned by a topological order
    that is maintained incrementally (Pearce-Kelly 2006) as edges are added.
  * known_pairs: exact closure size, computed on demand by chain-label passes
    over a chain cover (the Round-1 columns), a block of chains at a time so
    peak memory is n * block bytes instead of n * w.
  * is_complete: closure is total iff consecutive items of the topological
    order are joined by a direct edge (covering pairs are always edges).
  * total_order: Kahn's algorithm with the alphabetical tie-break that
    TransitiveOrder.total_order uses, via a heap.

Semantics match ChainTransitiveOrder exactly: add_total_order processes pairs
in the same order (adjacent pairs first, then the rest), skips known pairs,
counts contradictions as pair flips, and known_pairs() equals the number of
ordered pairs in the transitive closure.  Any valid topological order gives
the same answers, so seed_order() may use any heuristic ranking.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

_NO_LABEL = 255  # uint8 "not reachable" sentinel; chains must be shorter than this


# ------------------------------- numba kernels -------------------------------


@njit(cache=True)
def _reaches(x, y, out_head, out_next, e_dst, ordv, col, pos, visit, stampbox, stack):
    """True iff there is a directed path x -> ... -> y (x is known greater)."""
    if x == y:
        return False
    oy = ordv[y]
    if ordv[x] >= oy:
        return False
    cy = col[y]
    py = pos[y]
    if col[x] == cy:
        return pos[x] < py
    stampbox[0] += 1
    s = stampbox[0]
    visit[x] = s
    stack[0] = x
    sp = 1
    while sp > 0:
        sp -= 1
        u = stack[sp]
        e = out_head[u]
        while e != -1:
            v = e_dst[e]
            if visit[v] != s:
                visit[v] = s
                if v == y:
                    return True
                if ordv[v] < oy:
                    if col[v] == cy and pos[v] <= py:
                        return True
                    stack[sp] = v
                    sp += 1
            e = out_next[e]
    return False


@njit(cache=True)
def _insert_edge(u, v, e_src, e_dst, out_head, out_next, in_head, in_next, nedges,
                 ordv, node_at, visit_pk, stampbox, stack, buf_f, buf_b):
    """Append edge u -> v (caller guarantees neither endpoint reaches the other)
    and restore the topological order with Pearce-Kelly if it is violated."""
    e = nedges[0]
    nedges[0] = e + 1
    e_src[e] = u
    e_dst[e] = v
    out_next[e] = out_head[u]
    out_head[u] = e
    in_next[e] = in_head[v]
    in_head[v] = e
    ub = ordv[u]
    lb = ordv[v]
    if ub < lb:
        return

    # Forward region: reachable from v with ord < ord[u].
    stampbox[0] += 1
    s = stampbox[0]
    nf = 0
    visit_pk[v] = s
    buf_f[nf] = v
    nf += 1
    stack[0] = v
    sp = 1
    while sp > 0:
        sp -= 1
        a = stack[sp]
        e2 = out_head[a]
        while e2 != -1:
            b = e_dst[e2]
            if visit_pk[b] != s and ordv[b] < ub:
                visit_pk[b] = s
                buf_f[nf] = b
                nf += 1
                stack[sp] = b
                sp += 1
            e2 = out_next[e2]

    # Backward region: reaches u with ord > ord[v].
    stampbox[0] += 1
    s = stampbox[0]
    nb = 0
    visit_pk[u] = s
    buf_b[nb] = u
    nb += 1
    stack[0] = u
    sp = 1
    while sp > 0:
        sp -= 1
        a = stack[sp]
        e2 = in_head[a]
        while e2 != -1:
            b = e_src[e2]
            if visit_pk[b] != s and ordv[b] > lb:
                visit_pk[b] = s
                buf_b[nb] = b
                nb += 1
                stack[sp] = b
                sp += 1
            e2 = in_next[e2]

    f_nodes = buf_f[:nf].copy()
    b_nodes = buf_b[:nb].copy()
    f_ord = ordv[f_nodes]
    b_ord = ordv[b_nodes]
    f_nodes = f_nodes[np.argsort(f_ord)]
    b_nodes = b_nodes[np.argsort(b_ord)]
    pool = np.sort(np.concatenate((b_ord, f_ord)))
    for i in range(nb):
        x = b_nodes[i]
        ordv[x] = pool[i]
        node_at[pool[i]] = x
    for i in range(nf):
        x = f_nodes[i]
        ordv[x] = pool[nb + i]
        node_at[pool[nb + i]] = x


@njit(cache=True)
def _adjacent_phase(order, e_src, e_dst, out_head, out_next, in_head, in_next, nedges,
                    ordv, node_at, col, pos, visit, visit_pk, stampbox, stack, buf_f, buf_b):
    """add_relation(order[i], order[i+1]) for each i. Returns (added, rejected)."""
    added = 0
    rejected = 0
    for i in range(order.shape[0] - 1):
        h = order[i]
        lo = order[i + 1]
        if h == lo:
            continue
        if _reaches(h, lo, out_head, out_next, e_dst, ordv, col, pos, visit, stampbox, stack):
            continue
        if _reaches(lo, h, out_head, out_next, e_dst, ordv, col, pos, visit, stampbox, stack):
            rejected += 1
            continue
        _insert_edge(h, lo, e_src, e_dst, out_head, out_next, in_head, in_next, nedges,
                     ordv, node_at, visit_pk, stampbox, stack, buf_f, buf_b)
        added += 1
    return added, rejected


@njit(cache=True)
def _mark_desc(start, bound, out_head, out_next, e_dst, ordv, mark, s, stack):
    """Mark start's descendants with ord <= bound (expanding only those)."""
    stack[0] = start
    sp = 1
    while sp > 0:
        sp -= 1
        a = stack[sp]
        e = out_head[a]
        while e != -1:
            b = e_dst[e]
            if mark[b] != s and ordv[b] <= bound:
                mark[b] = s
                stack[sp] = b
                sp += 1
            e = out_next[e]


@njit(cache=True)
def _mark_anc(start, bound, in_head, in_next, e_src, ordv, mark, s, stack):
    """Mark start's ancestors with ord >= bound (expanding only those)."""
    stack[0] = start
    sp = 1
    while sp > 0:
        sp -= 1
        a = stack[sp]
        e = in_head[a]
        while e != -1:
            b = e_src[e]
            if mark[b] != s and ordv[b] >= bound:
                mark[b] = s
                stack[sp] = b
                sp += 1
            e = in_next[e]


@njit(cache=True)
def _nonadjacent_rows(order, i0, i1, e_src, e_dst, out_head, out_next, in_head, in_next, nedges,
                      ordv, node_at, mark_d, mark_a, visit_pk, stampbox, stack, buf_f, buf_b):
    """add_relation(order[i], order[j]) for i in [i0, i1), j >= i + 2, in that
    order. Per row i, desc(h)/anc(h) are marked once (restricted to the ord
    window of the remaining members) and desc is extended through each newly
    inserted edge, which gives the same answers as one query per pair."""
    L = order.shape[0]
    added = 0
    rejected = 0
    for i in range(i0, i1):
        if i + 2 >= L:
            break
        h = order[i]
        hi_bound = -1
        lo_bound = 1 << 30
        for j in range(i + 2, L):
            o = ordv[order[j]]
            if o > hi_bound:
                hi_bound = o
            if o < lo_bound:
                lo_bound = o
        stampbox[0] += 1
        sd = stampbox[0]
        if ordv[h] < hi_bound:
            _mark_desc(h, hi_bound, out_head, out_next, e_dst, ordv, mark_d, sd, stack)
        stampbox[0] += 1
        sa = stampbox[0]
        if ordv[h] > lo_bound:
            _mark_anc(h, lo_bound, in_head, in_next, e_src, ordv, mark_a, sa, stack)
        for j in range(i + 2, L):
            lo = order[j]
            if lo == h or mark_d[lo] == sd:
                continue
            if mark_a[lo] == sa:
                rejected += 1
                continue
            _insert_edge(h, lo, e_src, e_dst, out_head, out_next, in_head, in_next, nedges,
                         ordv, node_at, visit_pk, stampbox, stack, buf_f, buf_b)
            added += 1
            mark_d[lo] = sd
            bound = -1
            for jj in range(j + 1, L):
                o = ordv[order[jj]]
                if o > bound:
                    bound = o
            if bound >= 0:
                _mark_desc(lo, bound, out_head, out_next, e_dst, ordv, mark_d, sd, stack)
    return added, rejected


@njit(cache=True)
def _count_desc(node_at, out_head, out_next, e_dst, col, pos, chain_len, block, e_limit, lab, nonempty, out):
    """out[x] = |descendants of x| using edges with id < e_limit."""
    n = node_at.shape[0]
    w = chain_len.shape[0]
    out[:] = 0
    for c0 in range(0, w, block):
        bw = min(block, w - c0)
        for i in range(n - 1, -1, -1):
            x = node_at[i]
            ne = False
            cx = col[x]
            if c0 <= cx < c0 + bw and pos[x] + 1 < chain_len[cx]:
                for j in range(bw):
                    lab[x, j] = _NO_LABEL
                lab[x, cx - c0] = pos[x] + 1
                ne = True
            e = out_head[x]
            while e != -1:
                if e < e_limit:
                    y = e_dst[e]
                    cy = col[y]
                    yin = c0 <= cy < c0 + bw
                    if yin or nonempty[y]:
                        if not ne:
                            for j in range(bw):
                                lab[x, j] = _NO_LABEL
                            ne = True
                        if yin and pos[y] < lab[x, cy - c0]:
                            lab[x, cy - c0] = pos[y]
                        if nonempty[y]:
                            for j in range(bw):
                                if lab[y, j] < lab[x, j]:
                                    lab[x, j] = lab[y, j]
                e = out_next[e]
            nonempty[x] = ne
            if ne:
                s = 0
                for j in range(bw):
                    v = lab[x, j]
                    if v != _NO_LABEL:
                        s += chain_len[c0 + j] - v
                out[x] += s
    return out.sum()


@njit(cache=True)
def _count_anc(node_at, in_head, in_next, e_src, col, pos, chain_len, block, e_limit, lab, nonempty, out):
    """out[x] = |ancestors of x|; lab holds 'how many of chain c reach x'."""
    n = node_at.shape[0]
    w = chain_len.shape[0]
    out[:] = 0
    for c0 in range(0, w, block):
        bw = min(block, w - c0)
        for i in range(n):
            x = node_at[i]
            ne = False
            cx = col[x]
            if c0 <= cx < c0 + bw and pos[x] > 0:
                for j in range(bw):
                    lab[x, j] = 0
                lab[x, cx - c0] = pos[x]
                ne = True
            e = in_head[x]
            while e != -1:
                if e < e_limit:
                    p = e_src[e]
                    cp = col[p]
                    pin = c0 <= cp < c0 + bw
                    if pin or nonempty[p]:
                        if not ne:
                            for j in range(bw):
                                lab[x, j] = 0
                            ne = True
                        if pin and pos[p] + 1 > lab[x, cp - c0]:
                            lab[x, cp - c0] = pos[p] + 1
                        if nonempty[p]:
                            for j in range(bw):
                                if lab[p, j] > lab[x, j]:
                                    lab[x, j] = lab[p, j]
                e = in_next[e]
            nonempty[x] = ne
            if ne:
                s = 0
                for j in range(bw):
                    s += lab[x, j]
                out[x] += s
    return out.sum()


@njit(cache=True)
def _is_complete(node_at, out_head, out_next, e_dst):
    n = node_at.shape[0]
    for i in range(n - 1):
        a = node_at[i]
        b = node_at[i + 1]
        found = False
        e = out_head[a]
        while e != -1:
            if e_dst[e] == b:
                found = True
                break
            e = out_next[e]
        if not found:
            return False
    return True


@njit(cache=True)
def _kahn_heap(n, in_head, in_next, out_head, out_next, e_dst, key):
    """Kahn's algorithm, always emitting the ready item with the smallest key."""
    indeg = np.zeros(n, np.int64)
    for x in range(n):
        e = in_head[x]
        while e != -1:
            indeg[x] += 1
            e = in_next[e]
    heap = np.empty(n, np.int64)
    hs = 0
    for x in range(n):
        if indeg[x] == 0:
            # push
            heap[hs] = x
            k = hs
            hs += 1
            while k > 0:
                parent = (k - 1) // 2
                if key[heap[parent]] > key[heap[k]]:
                    t = heap[parent]
                    heap[parent] = heap[k]
                    heap[k] = t
                    k = parent
                else:
                    break
    out = np.empty(n, np.int64)
    no = 0
    while hs > 0:
        top = heap[0]
        hs -= 1
        heap[0] = heap[hs]
        k = 0
        while True:
            l = 2 * k + 1
            r = l + 1
            sm = k
            if l < hs and key[heap[l]] < key[heap[sm]]:
                sm = l
            if r < hs and key[heap[r]] < key[heap[sm]]:
                sm = r
            if sm == k:
                break
            t = heap[sm]
            heap[sm] = heap[k]
            heap[k] = t
            k = sm
        out[no] = top
        no += 1
        e = out_head[top]
        while e != -1:
            y = e_dst[e]
            indeg[y] -= 1
            if indeg[y] == 0:
                heap[hs] = y
                k = hs
                hs += 1
                while k > 0:
                    parent = (k - 1) // 2
                    if key[heap[parent]] > key[heap[k]]:
                        t = heap[parent]
                        heap[parent] = heap[k]
                        heap[k] = t
                        k = parent
                    else:
                        break
            e = out_next[e]
    return out[:no]


@njit(cache=True)
def _greedy_batch(deg, m, out_head, out_next, e_dst, in_head, in_next, e_src,
                  mark_d, mark_a, stampbox, stack, in_batch, gain):
    """Same selection as BatchedLLMSorter._next_unresolved_batch_greedy: seed with
    the first max-degree item, then repeatedly add the first item that has the
    most still-unknown pairs with the current batch (gain must exceed 0)."""
    n = deg.shape[0]
    batch = np.empty(m, np.int64)
    if n == 0 or deg.max() == 0:
        return batch[:0]
    seed = np.argmax(deg)
    in_batch[:] = 0
    gain[:] = 0
    batch[0] = seed
    in_batch[seed] = 1
    nb = 1
    cur = seed
    while nb < m:
        stampbox[0] += 1
        sd = stampbox[0]
        mark_d[cur] = sd
        stack[0] = cur
        sp = 1
        while sp > 0:
            sp -= 1
            a = stack[sp]
            e = out_head[a]
            while e != -1:
                b = e_dst[e]
                if mark_d[b] != sd:
                    mark_d[b] = sd
                    stack[sp] = b
                    sp += 1
                e = out_next[e]
        stampbox[0] += 1
        sa = stampbox[0]
        stack[0] = cur
        sp = 1
        while sp > 0:
            sp -= 1
            a = stack[sp]
            e = in_head[a]
            while e != -1:
                b = e_src[e]
                if mark_a[b] != sa:
                    mark_a[b] = sa
                    stack[sp] = b
                    sp += 1
                e = in_next[e]
        best = -1
        best_gain = 0
        for c in range(n):
            if in_batch[c]:
                continue
            if mark_d[c] != sd and mark_a[c] != sa:
                gain[c] += 1
            if gain[c] > best_gain:
                best_gain = gain[c]
                best = c
        if best == -1:
            break
        batch[nb] = best
        nb += 1
        in_batch[best] = 1
        cur = best
    return batch[:nb]


@njit(cache=True)
def _greedy_batch_taken(deg, m, out_head, out_next, e_dst, in_head, in_next, e_src,
                        mark_d, mark_a, stampbox, stack, in_batch, gain, taken, force_seed):
    """_greedy_batch restricted to keys not yet `taken` by another batch of the
    same round; marks the chosen keys as taken. With `taken` all zero it picks
    exactly the batch _greedy_batch picks. force_seed >= 0 seeds the batch with
    that key instead (a pivot shared by the round's batches; never marked taken)."""
    n = deg.shape[0]
    batch = np.empty(m, np.int64)
    seed = force_seed
    if seed < 0:
        best_d = 0
        for x in range(n):
            if not taken[x] and deg[x] > best_d:
                best_d = deg[x]
                seed = x
    if seed < 0:
        return batch[:0]
    in_batch[:] = 0
    gain[:] = 0
    batch[0] = seed
    in_batch[seed] = 1
    nb = 1
    cur = seed
    while nb < m:
        stampbox[0] += 1
        sd = stampbox[0]
        mark_d[cur] = sd
        stack[0] = cur
        sp = 1
        while sp > 0:
            sp -= 1
            a = stack[sp]
            e = out_head[a]
            while e != -1:
                b = e_dst[e]
                if mark_d[b] != sd:
                    mark_d[b] = sd
                    stack[sp] = b
                    sp += 1
                e = out_next[e]
        stampbox[0] += 1
        sa = stampbox[0]
        stack[0] = cur
        sp = 1
        while sp > 0:
            sp -= 1
            a = stack[sp]
            e = in_head[a]
            while e != -1:
                b = e_src[e]
                if mark_a[b] != sa:
                    mark_a[b] = sa
                    stack[sp] = b
                    sp += 1
                e = in_next[e]
        best = -1
        best_gain = 0
        for c in range(n):
            if in_batch[c] or taken[c]:
                continue
            if mark_d[c] != sd and mark_a[c] != sa:
                gain[c] += 1
            if gain[c] > best_gain:
                best_gain = gain[c]
                best = c
        if best == -1:
            break
        batch[nb] = best
        nb += 1
        in_batch[best] = 1
        cur = best
    for i in range(nb):
        if batch[i] != force_seed:
            taken[batch[i]] = 1
    return batch[:nb]


@njit(cache=True)
def _count_inversions(a):
    """Number of i < j with a[i] > a[j] for a permutation of 0..n-1 (Fenwick tree)."""
    n = a.shape[0]
    tree = np.zeros(n + 1, np.int64)
    inv = 0
    for i in range(n - 1, -1, -1):
        v = a[i]
        k = v
        s = 0
        while k > 0:
            s += tree[k]
            k -= k & (-k)
        inv += s
        k = v + 1
        while k <= n:
            tree[k] += 1
            k += k & (-k)
    return inv


def fast_kendalls_tau(predicted_order: Sequence[str], true_order: Sequence[str]) -> float:
    """Same value as tbs.metrics.kendalls_tau in O(n log n)."""
    if len(predicted_order) != len(true_order):
        raise ValueError("Orders must have the same length")
    n = len(true_order)
    if n < 2:
        return 1.0
    true_pos = {item: i for i, item in enumerate(true_order)}
    if len(true_pos) != n or any(x not in true_pos for x in predicted_order):
        raise ValueError("Orders have different items")
    a = np.fromiter((true_pos[x] for x in predicted_order), dtype=np.int64, count=n)
    discordant = int(_count_inversions(a))
    total = n * (n - 1) / 2
    return (total - 2 * discordant) / total


# ------------------------------- Python wrapper -------------------------------


class _DagBelowView:
    __slots__ = ("g", "x")

    def __init__(self, g: "DagTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return self.g.is_greater(self.x, y)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return int(self.g.desc_anc_counts()[0][self.g._index[self.x]])

    def __iter__(self) -> Iterator[str]:
        g = self.g
        xi = g._index[self.x]
        return (g.items[y] for y in range(len(g.items)) if g._reaches_idx(xi, y))


class _DagAboveView:
    __slots__ = ("g", "x")

    def __init__(self, g: "DagTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return self.g.is_greater(y, self.x)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return int(self.g.desc_anc_counts()[1][self.g._index[self.x]])

    def __iter__(self) -> Iterator[str]:
        g = self.g
        xi = g._index[self.x]
        return (g.items[y] for y in range(len(g.items)) if g._reaches_idx(y, xi))


class _DagUnknownView:
    __slots__ = ("g", "x")

    def __init__(self, g: "DagTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return y != self.x and y in self.g._index and not self.g.comparable(self.x, y)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.g.unresolved_degree(self.x)

    def __iter__(self) -> Iterator[str]:
        return (y for y in self.g.items if y != self.x and not self.g.comparable(self.x, y))


class _DagViewMap:
    __slots__ = ("g", "view_cls")

    def __init__(self, g: "DagTransitiveOrder", view_cls: type) -> None:
        self.g, self.view_cls = g, view_cls

    def __getitem__(self, x: str):
        if x not in self.g._index:
            raise KeyError(x)
        return self.view_cls(self.g, x)

    def __contains__(self, x: object) -> bool:
        return x in self.g._index

    def __iter__(self) -> Iterator[str]:
        return iter(self.g.items)

    def __len__(self) -> int:
        return len(self.g.items)


class DagTransitiveOrder:
    """Drop-in replacement for ChainTransitiveOrder (graph_backend="dag")."""

    def __init__(
        self,
        items: Sequence[str],
        verbose: bool = False,
        use_transitivity: bool = True,
        block: int = 256,
        edge_capacity: Optional[int] = None,
    ):
        if not use_transitivity:
            raise ValueError("DagTransitiveOrder requires use_transitivity=True")
        self.items = list(items)
        n = len(self.items)
        self._index: Dict[str, int] = {x: i for i, x in enumerate(self.items)}
        if len(self._index) != n:
            raise ValueError("items must be unique")
        self.verbose = verbose
        self.use_transitivity = True
        self.block = block

        cap = edge_capacity or max(1024, 4 * n)
        self._e_src = np.empty(cap, np.int32)
        self._e_dst = np.empty(cap, np.int32)
        self._out_next = np.empty(cap, np.int32)
        self._in_next = np.empty(cap, np.int32)
        self._nedges = np.zeros(1, np.int64)
        self._out_head = np.full(n, -1, np.int32)
        self._in_head = np.full(n, -1, np.int32)
        self._ordv = np.arange(n, dtype=np.int32)
        self._node_at = np.arange(n, dtype=np.int32)
        # Chain cover used by the counting passes and as a query shortcut.
        # Default: every item is its own chain (always valid).
        self._col = np.arange(n, dtype=np.int32)
        self._pos = np.zeros(n, np.int32)
        self._chain_len = np.ones(n, np.int32)

        self._visit = np.zeros(n, np.int64)
        self._visit_pk = np.zeros(n, np.int64)
        self._mark_d = np.zeros(n, np.int64)
        self._mark_a = np.zeros(n, np.int64)
        self._stampbox = np.zeros(1, np.int64)
        self._stack = np.empty(max(n, 1), np.int32)
        self._buf_f = np.empty(max(n, 1), np.int32)
        self._buf_b = np.empty(max(n, 1), np.int32)

        self._total_pairs = n * (n - 1) // 2
        self._counts_cache: Optional[Tuple[int, np.ndarray, Optional[np.ndarray]]] = None

        self.greater = _DagViewMap(self, _DagBelowView)
        self.less = _DagViewMap(self, _DagAboveView)
        self.unknown = _DagViewMap(self, _DagUnknownView)

        self.direct_relations = 0
        self.rejected_contradictions_count = 0
        # Contradiction records are not kept (they always classify as a pair
        # flip, see ChainTransitiveOrder: greater[] is the full closure).
        self.rejected_contradictions_list: List[Dict[str, Any]] = []
        self.rejected_pair_flips_count = 0
        self.rejected_transitivity_cycles_count = 0

    # ------------------------------ setup helpers ------------------------------

    @property
    def num_edges(self) -> int:
        return int(self._nedges[0])

    def seed_order(self, rank: Dict[str, float]) -> None:
        """Initialize the topological order from any heuristic ranking (smaller
        = greater). Only allowed before edges exist; never changes answers,
        only how much reordering later insertions trigger."""
        if self.num_edges:
            raise RuntimeError("seed_order must be called before any edges are added")
        big = float("inf")
        keys = np.array([rank.get(x, big) for x in self.items], dtype=np.float64)
        node_at = np.argsort(keys, kind="stable").astype(np.int32)
        self._node_at[:] = node_at
        self._ordv[node_at] = np.arange(len(self.items), dtype=np.int32)

    def set_chain_cover(self, chains: Sequence[Sequence[str]]) -> None:
        """Use these chains (each ordered greatest-first, already a path in the
        graph, together covering every item exactly once) for counting."""
        n = len(self.items)
        col = np.full(n, -1, np.int32)
        pos = np.zeros(n, np.int32)
        lens = np.zeros(len(chains), np.int32)
        for c, chain in enumerate(chains):
            if len(chain) >= _NO_LABEL:
                raise ValueError(f"chain {c} has {len(chain)} items; max is {_NO_LABEL - 1}")
            lens[c] = len(chain)
            for p, x in enumerate(chain):
                i = self._index[x]
                if col[i] != -1:
                    raise ValueError(f"item {x!r} appears in more than one chain")
                col[i] = c
                pos[i] = p
        if (col == -1).any():
            raise ValueError("chain cover must include every item")
        for chain in chains:
            for a, b in zip(chain, chain[1:]):
                if not self.is_greater(a, b):
                    raise ValueError(f"chain cover is not a path in the graph: {a!r} !> {b!r}")
        self._col, self._pos, self._chain_len = col, pos, lens
        self._counts_cache = None

    def _ensure_capacity(self, extra: int) -> None:
        need = self.num_edges + extra
        cap = self._e_src.shape[0]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap)
        for name in ("_e_src", "_e_dst", "_out_next", "_in_next"):
            old = getattr(self, name)
            arr = np.empty(new_cap, np.int32)
            arr[: old.shape[0]] = old
            setattr(self, name, arr)

    # ------------------------------ queries ------------------------------------

    def _reaches_idx(self, x: int, y: int) -> bool:
        return bool(
            _reaches(x, y, self._out_head, self._out_next, self._e_dst, self._ordv,
                     self._col, self._pos, self._visit, self._stampbox, self._stack)
        )

    def is_greater(self, x: str, y: str) -> bool:
        xi = self._index.get(x)
        yi = self._index.get(y)
        if xi is None or yi is None:
            return False
        return self._reaches_idx(xi, yi)

    def comparable(self, a: str, b: str) -> bool:
        ai = self._index.get(a)
        bi = self._index.get(b)
        if ai is None or bi is None or ai == bi:
            return False
        return self._reaches_idx(ai, bi) or self._reaches_idx(bi, ai)

    def all_pairs_known(self, batch: Sequence[str]) -> bool:
        for i, hi in enumerate(batch):
            for lo in batch[i + 1 :]:
                if not self.comparable(hi, lo):
                    return False
        return True

    def desc_anc_counts(self, with_anc: bool = True, e_limit: Optional[int] = None):
        """Per-item (|descendants|, |ancestors| or None), exact, cached per edge count."""
        ne = self.num_edges if e_limit is None else e_limit
        cache = self._counts_cache
        if cache is not None and cache[0] == ne and (cache[2] is not None or not with_anc):
            return cache[1], cache[2]
        n = len(self.items)
        block = max(1, min(self.block, self._chain_len.shape[0]))
        lab = np.empty((n, block), np.uint8)
        nonempty = np.zeros(n, np.bool_)
        desc = np.zeros(n, np.int64)
        _count_desc(self._node_at, self._out_head, self._out_next, self._e_dst, self._col,
                    self._pos, self._chain_len, block, ne, lab, nonempty, desc)
        anc = None
        if with_anc:
            anc = np.zeros(n, np.int64)
            nonempty[:] = False
            _count_anc(self._node_at, self._in_head, self._in_next, self._e_src, self._col,
                       self._pos, self._chain_len, block, ne, lab, nonempty, anc)
        del lab
        if e_limit is None or e_limit == self.num_edges:
            self._counts_cache = (ne, desc, anc)
        return desc, anc

    def known_pairs(self, e_limit: Optional[int] = None) -> int:
        ne = self.num_edges if e_limit is None else e_limit
        cache = self._counts_cache
        if cache is not None and cache[0] == ne:
            return int(cache[1].sum())
        desc, _ = self.desc_anc_counts(with_anc=False, e_limit=e_limit)
        return int(desc.sum())

    @property
    def u(self) -> int:
        return self._total_pairs - self.known_pairs()

    @property
    def transitive_relations(self) -> int:
        # Every accepted direct edge contributes itself plus its transitive
        # pairs, so known = direct + transitive (as in ChainTransitiveOrder).
        return self.known_pairs() - self.direct_relations

    @property
    def direct_relation_pairs(self) -> List[Tuple[str, str]]:
        ne = self.num_edges
        return [(self.items[a], self.items[b]) for a, b in zip(self._e_src[:ne], self._e_dst[:ne])]

    def unresolved_degree(self, x: str) -> int:
        desc, anc = self.desc_anc_counts()
        i = self._index[x]
        return len(self.items) - 1 - int(desc[i]) - int(anc[i])

    def is_complete(self) -> bool:
        if len(self.items) < 2:
            return True
        if self.num_edges < len(self.items) - 1:
            return False
        return bool(_is_complete(self._node_at, self._out_head, self._out_next, self._e_dst))

    def residual_pairs(self) -> List[Tuple[str, str]]:
        if len(self.items) > 20000:
            raise RuntimeError("residual_pairs is O(n^2); not available at this scale")
        pairs = []
        for i, x in enumerate(self.items):
            for y in self.items[i + 1 :]:
                if not self.comparable(x, y):
                    pairs.append((x, y) if x < y else (y, x))
        return pairs

    # ------------------------------ updates ------------------------------------

    def add_relation(self, higher: str, lower: str, direct: bool = False) -> None:
        self.add_total_order([higher, lower], direct=direct)

    def add_total_order(self, ordered_items: Sequence[str], direct: bool = False) -> int:
        """Same pair order and outcomes as ChainTransitiveOrder.add_total_order.
        Returns the number of newly accepted direct edges."""
        order = np.fromiter((self._index[x] for x in ordered_items), dtype=np.int32,
                            count=len(ordered_items))
        L = order.shape[0]
        if L < 2:
            return 0
        self._ensure_capacity(L)
        added, rejected = _adjacent_phase(
            order, self._e_src, self._e_dst, self._out_head, self._out_next, self._in_head,
            self._in_next, self._nedges, self._ordv, self._node_at, self._col, self._pos,
            self._visit, self._visit_pk, self._stampbox, self._stack, self._buf_f, self._buf_b,
        )
        if rejected:
            # A broken link means pairs spanning it may still be unknown.
            i = 0
            while i < L - 2:
                rows = max(1, min(L - 2 - i, 4096 // max(1, L - i)))
                self._ensure_capacity(rows * L)
                a, r = _nonadjacent_rows(
                    order, i, i + rows, self._e_src, self._e_dst, self._out_head, self._out_next,
                    self._in_head, self._in_next, self._nedges, self._ordv, self._node_at,
                    self._mark_d, self._mark_a, self._visit_pk, self._stampbox, self._stack,
                    self._buf_f, self._buf_b,
                )
                added += a
                rejected += r
                i += rows
        if direct:
            self.direct_relations += int(added)
        self.rejected_contradictions_count += int(rejected)
        self.rejected_pair_flips_count += int(rejected)
        if self.verbose and rejected:
            print(f"  {rejected} contradictory relation(s) ignored")
        return int(added)

    def truncate_edges(self, keep: int) -> None:
        """Drop every edge with id >= keep (edges are numbered in insertion
        order, so this restores the graph as it was after `keep` insertions).
        The topological order stays valid for the smaller edge set."""
        ne = self.num_edges
        if keep >= ne:
            return
        self._out_head[:] = -1
        self._in_head[:] = -1
        for e in range(keep):
            u, v = int(self._e_src[e]), int(self._e_dst[e])
            self._out_next[e] = self._out_head[u]
            self._out_head[u] = e
            self._in_next[e] = self._in_head[v]
            self._in_head[v] = e
        self._nedges[0] = keep
        self._counts_cache = None

    # ------------------------------ outputs ------------------------------------

    def greedy_batch(self, deg: np.ndarray, m: int) -> List[str]:
        n = len(self.items)
        in_batch = np.zeros(n, np.uint8)
        gain = np.zeros(n, np.int32)
        idx = _greedy_batch(
            deg.astype(np.int64), m, self._out_head, self._out_next, self._e_dst, self._in_head,
            self._in_next, self._e_src, self._mark_d, self._mark_a, self._stampbox, self._stack,
            in_batch, gain,
        )
        return [self.items[i] for i in idx]

    def greedy_batches(self, deg: np.ndarray, m: int, count: int,
                       shared_seed: bool = False) -> List[List[str]]:
        """Up to `count` greedy batches from one degree count. The first batch
        equals greedy_batch(deg, m). Later batches use keys no earlier batch of
        the round used; with shared_seed they all keep the first batch's seed
        (one pivot, disjoint companions, as consecutive sequential batches do),
        otherwise each takes the highest-degree unused key as its own seed."""
        n = len(self.items)
        in_batch = np.zeros(n, np.uint8)
        gain = np.zeros(n, np.int32)
        taken = np.zeros(n, np.uint8)
        d = deg.astype(np.int64)
        out: List[List[str]] = []
        force = -1
        for _ in range(count):
            idx = _greedy_batch_taken(
                d, m, self._out_head, self._out_next, self._e_dst, self._in_head,
                self._in_next, self._e_src, self._mark_d, self._mark_a, self._stampbox, self._stack,
                in_batch, gain, taken, force,
            )
            if len(idx) < 2:
                break
            if shared_seed and force < 0:
                force = int(idx[0])
            out.append([self.items[i] for i in idx])
        return out

    def total_order(self) -> List[str]:
        """Kahn's topological sort, ties broken alphabetically (as TransitiveOrder)."""
        n = len(self.items)
        alpha = np.empty(n, np.int64)
        alpha[np.array(sorted(range(n), key=self.items.__getitem__), dtype=np.int64)] = np.arange(n)
        out = _kahn_heap(n, self._in_head, self._in_next, self._out_head, self._out_next,
                         self._e_dst, alpha)
        if out.shape[0] != n:
            raise ValueError("Cycle detected or incomplete ordering.")
        return [self.items[i] for i in out]
