"""Transitive-closure graph of learned pairwise relations (sets and chain backends)."""

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


class TransitiveOrder:
    """
    Maintains the full transitive closure of "greater than" relations
    between items so that redundant comparisons are never issued.
    """

    def __init__(self, items: Sequence[str], verbose: bool = False, use_transitivity: bool = True):
        self.items = list(items)
        self.greater: Dict[str, Set[str]] = {item: set() for item in items}
        self.less: Dict[str, Set[str]] = {item: set() for item in items}
        # Unknown-neighborhood tracking (paper Sec 4.4): U[x] and u, maintained
        # incrementally on every accepted edge so completeness is O(1) and
        # residual-pair enumeration is O(n+u) instead of an O(n^2) rescan.
        self.unknown: Dict[str, Set[str]] = {
            item: set(self.items) - {item} for item in items
        }
        n = len(self.items)
        self.u: int = n * (n - 1) // 2
        self.verbose = verbose
        self.use_transitivity = use_transitivity
        self.direct_relations = 0
        self.transitive_relations = 0
        self.direct_relation_pairs: List[Tuple[str, str]] = []
        self.rejected_contradictions_count = 0
        self.rejected_contradictions_list: List[Dict[str, Any]] = []
        self.rejected_pair_flips_count = 0
        self.rejected_transitivity_cycles_count = 0

    def all_pairs_known(self, batch: Sequence[str]) -> bool:
        if len(batch) < 2:
            return True
        for i, hi in enumerate(batch):
            for lo in batch[i + 1 :]:
                if not self.comparable(hi, lo):
                    return False
        return True

    def comparable(self, a: str, b: str) -> bool:
        return b in self.greater[a] or a in self.greater[b]

    def known_pairs(self) -> int:
        n = len(self.items)
        return n * (n - 1) // 2 - self.u

    def add_relation(self, higher: str, lower: str, direct: bool = False) -> None:
        if higher == lower or lower in self.greater[higher]:
            return

        # Skip contradictory evidence that would create a cycle.
        if higher in self.greater[lower]:
            self.rejected_contradictions_count += 1
            chain_path = self._path_from_to(lower, higher)
            chain_str = " > ".join(chain_path) if chain_path else f"{lower} > {higher}"
            cycle_str = f"{chain_str} > {lower}" if chain_path else f"{lower} > {higher} > {lower}"
            chain_length = len(chain_path) if chain_path else 2
            cycle_type = "Pair flip" if chain_length == 2 else "Transitivity cycle"
            if cycle_type == "Pair flip":
                self.rejected_pair_flips_count += 1
            else:
                self.rejected_transitivity_cycles_count += 1
            record = {
                "rejected_pair": f"{higher} > {lower}",
                "existing_chain": chain_str,
                "cycle_would_be": cycle_str,
                "items_in_cycle": list(chain_path) + [lower] if chain_path else [lower, higher, lower],
                "chain_length": chain_length,
                "cycle_type": cycle_type,
            }
            self.rejected_contradictions_list.append(record)
            if self.verbose:
                print(
                    f"Conflict ignored: attempted to add {higher} > {lower} "
                    f"but {lower} is already known greater than {higher} (chain: {chain_str})."
                )
            return

        if not self.use_transitivity:
            self.greater[higher].add(lower)
            self.less[lower].add(higher)
            self._mark_known(higher, lower)
            if direct:
                self.direct_relations += 1
                self.direct_relation_pairs.append((higher, lower))
            return

        higher_chain = {higher} | self.less[higher]
        lower_chain = {lower} | self.greater[lower]

        new_relations = []
        for hi in higher_chain:
            for lo in lower_chain:
                if hi == lo or lo in self.greater[hi]:
                    continue
                self.greater[hi].add(lo)
                self.less[lo].add(hi)
                self._mark_known(hi, lo)
                if (hi, lo) != (higher, lower):
                    new_relations.append((hi, lo))

        if direct:
            self.direct_relations += 1
            self.direct_relation_pairs.append((higher, lower))
            # Also count transitive relations triggered by this direct relation
            if new_relations:
                self.transitive_relations += len(new_relations)
                if self.verbose:
                    print(f"  Direct: {higher} > {lower}")
                    print(f"    -> Also inferred {len(new_relations)} relations via transitivity:")
                    for hi, lo in new_relations:
                        chain_explanation = self._explain_transitive_chain(hi, lo, higher, lower)
                        print(f"      -> {hi} > {lo} {chain_explanation}")
            elif self.verbose:
                print(f"  Direct: {higher} > {lower}")
        else:
            self.transitive_relations += len(new_relations) + 1  # +1 for the relation itself
            if self.verbose and new_relations:
                print(f"  Via transitivity from {higher} > {lower}:")
                for hi, lo in new_relations[:5]:
                    print(f"    -> {hi} > {lo}")
                if len(new_relations) > 5:
                    print(f"    ... and {len(new_relations) - 5} more")

    def _explain_transitive_chain(self, hi: str, lo: str, new_higher: str, new_lower: str) -> str:
        """Explain how hi > lo was inferred from new_higher > new_lower."""
        # Case 1: hi is new_higher, lo is something new_lower is greater than
        if hi == new_higher and lo != new_lower:
            # This means: new_higher > new_lower and new_lower > lo (from previous knowledge)
            # So: new_higher > lo via transitivity
            return f"(because {new_higher} > {new_lower} and we know {new_lower} > {lo})"
        # Case 2: hi is something less than new_higher, lo is new_lower
        elif hi != new_higher and lo == new_lower:
            # This means: hi > new_higher (from previous knowledge) and new_higher > new_lower
            # So: hi > new_lower via transitivity
            return f"(because we know {hi} > {new_higher} and {new_higher} > {new_lower})"
        # Case 3: Both are different (longer chain)
        elif hi != new_higher and lo != new_lower:
            return f"(because we know {hi} > {new_higher}, {new_higher} > {new_lower}, and {new_lower} > {lo})"
        return ""

    def _path_from_to(self, start: str, end: str) -> List[str]:
        """Find a path from start to end following the greater relation (start > ... > end). Returns list of nodes [start, ..., end] or empty list."""
        if start == end:
            return [start]
        from collections import deque
        parent: Dict[str, Optional[str]] = {start: None}
        q: deque = deque([start])
        while q:
            u = q.popleft()
            for v in self.greater[u]:
                if v not in parent:
                    parent[v] = u
                    if v == end:
                        path = []
                        cur = end
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    q.append(v)
        return []

    def add_total_order(self, ordered_items: Sequence[str], direct: bool = False) -> None:
        for i, higher in enumerate(ordered_items):
            for lower in ordered_items[i + 1 :]:
                self.add_relation(higher, lower, direct=direct)

    def _mark_known(self, hi: str, lo: str) -> None:
        """Remove (hi,lo) from the tracked unknown neighborhoods U[hi]/U[lo]
        and decrement u, if it was still unresolved (paper Sec 4.4, incremental
        update). Called once per newly-inserted edge, so each unordered pair
        is removed at most once over the whole run."""
        if lo in self.unknown[hi]:
            self.unknown[hi].discard(lo)
            self.unknown[lo].discard(hi)
            self.u -= 1

    def is_complete(self) -> bool:
        """O(1) completeness test via the tracked counter u (paper Sec 4.4)."""
        return self.u == 0

    def residual_pairs(self) -> List[Tuple[str, str]]:
        """List every still-unresolved unordered pair in O(n+u) by walking the
        non-empty unknown neighborhoods, instead of rescanning all C(n,2)
        candidates (paper Sec 4.4)."""
        pairs: List[Tuple[str, str]] = []
        for x in self.items:
            for y in self.unknown[x]:
                if x < y:
                    pairs.append((x, y))
        return pairs

    def _break_ties_alphabetically(self, tied_items: List[str]) -> List[str]:
        """Break ties using alphabetical order (correct for incomparable items)."""
        # Items still tied are incomparable under current knowledge; break ties alphabetically.
        return sorted(tied_items)

    def total_order(self) -> List[str]:
        """Return a total order via Kahn's topological sort."""
        from collections import Counter
        
        in_degree = {item: len(self.less[item]) for item in self.items}
        
        deg_counts = Counter(in_degree.values())
        ties = {deg: count for deg, count in deg_counts.items() if count > 1}
        
        print("\n" + "="*70)
        print("TIE ANALYSIS")
        print("="*70)
        print(f"In-degree distribution: {dict(sorted(deg_counts.items()))}")
        print(f"Potential tie groups (items with same in_degree): {ties}")
        print(f"Total potential tie groups: {len(ties)}")
        print(f"Total items in tie groups: {sum(count - 1 for count in ties.values())}")
        print(f"\nNote: Not all tie groups become actual ties during processing.")
        print(f"Some resolve naturally as the graph structure unfolds.")
        
        for deg, count in sorted(ties.items()):
            tied_items = sorted([item for item, d in in_degree.items() if d == deg])
            print(f"  in_degree={deg}: {tied_items} ({count} items tied)")
        print("="*70)
        
        ready_initial = [item for item, deg in in_degree.items() if deg == 0]
        tie_count = 0
        if len(ready_initial) > 1:
            tie_count += 1
            print(f"\nInitial tie (in_degree=0): {ready_initial}")
            # Use alphabetical order to break ties
            ready_initial = self._break_ties_alphabetically(ready_initial)
            print(f"  -> Resolved alphabetically: {ready_initial}")
        
        ready = ready_initial.copy()
        order: List[str] = []
        iteration = 0

        placed: set = set()
        self.cycle_breaks = 0
        while True:
            if not ready:
                if len(order) == len(self.items) or self.use_transitivity:
                    break
                # Without transitivity, contradictory direct answers can form cycles, and
                # Kahn's sort stalls with items left. Place next the remaining item with the
                # most direct wins over the other remaining items (Copeland score; ties
                # alphabetical), then continue.
                rem = set(self.items) - placed
                def _copeland(x: str) -> int:
                    return len(self.greater[x] & rem) - len(self.less[x] & rem)
                best = max(_copeland(x) for x in rem)
                ready = [min(x for x in rem if _copeland(x) == best)]
                self.cycle_breaks += 1
            iteration += 1
            current = ready.pop(0)
            if current in placed:
                continue
            order.append(current)
            placed.add(current)
            
            new_ready = []
            for lower in sorted(self.greater[current]):
                if lower in placed:
                    continue
                in_degree[lower] -= 1
                if in_degree[lower] == 0:
                    new_ready.append(lower)
            
            if len(new_ready) > 1:
                tie_count += 1
                print(f"Iteration {iteration}: Tie encountered (in_degree=0): {new_ready}")
                # Use alphabetical order to break ties
                new_ready = self._break_ties_alphabetically(new_ready)
                print(f"  -> Resolved alphabetically: {new_ready}")
            
            ready.extend(new_ready)
            if len(ready) > 1:
                # Break ties alphabetically if multiple items are ready
                ready = self._break_ties_alphabetically(ready)

        print(f"\nTotal ties actually encountered during processing: {tie_count}")
        print(f"Explanation: Only {tie_count} out of {len(ties)} potential tie groups became actual ties.")
        print(f"  - Potential ties: Items that start with the same in_degree")
        print(f"  - Actual ties: Only when multiple items reach in_degree=0 at the same time")
        print(f"  - Resolution: Ties are broken using alphabetical sorting")
        print("="*70 + "\n")
        
        if len(order) != len(self.items):
            if self.use_transitivity:
                raise ValueError("Cycle detected or incomplete ordering.")
            missing = [item for item in self.items if item not in order]
            order.extend(sorted(missing))
        return order


_CHAIN_INF = 1 << 60


class _ChainBelowView:
    """Read-only set-like view of greater[x] (items below x) from chain labels."""

    __slots__ = ("g", "x")

    def __init__(self, g: "ChainTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return self.g.is_greater(self.x, y)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.g.count_below(self.x)

    def __iter__(self) -> Iterator[str]:
        g, labels = self.g, self.g.down[self.x]
        for c, members in enumerate(g.chain_members):
            head = g.chain_head[c]
            start = labels[c]
            if start == _CHAIN_INF:
                continue
            yield from members[max(0, int(start) - head):]


class _ChainAboveView:
    """Read-only set-like view of less[x] (items above x) from chain labels."""

    __slots__ = ("g", "x")

    def __init__(self, g: "ChainTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return self.g.is_greater(y, self.x)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.g.count_above(self.x)

    def __iter__(self) -> Iterator[str]:
        g, labels = self.g, self.g.up[self.x]
        for c, members in enumerate(g.chain_members):
            head = g.chain_head[c]
            end = labels[c]
            if end == -_CHAIN_INF:
                continue
            yield from members[: max(0, int(end) - head + 1)]


class _ChainUnknownView:
    """Read-only set-like view of U[x]: items not yet comparable to x."""

    __slots__ = ("g", "x")

    def __init__(self, g: "ChainTransitiveOrder", x: str) -> None:
        self.g, self.x = g, x

    def __contains__(self, y: object) -> bool:
        return y != self.x and y in self.g._index and not self.g.comparable(self.x, y)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.g.unresolved_degree(self.x)

    def __iter__(self) -> Iterator[str]:
        x = self.x
        for y in self.g.items:
            if y != x and not self.g.comparable(x, y):
                yield y


class _ChainViewMap:
    __slots__ = ("g", "view_cls")

    def __init__(self, g: "ChainTransitiveOrder", view_cls: type) -> None:
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


class ChainTransitiveOrder(TransitiveOrder):
    """
    Transitive closure stored as a chain-cover reachability index
    (Jagadish 1990) instead of per-item pair sets.

    Items are partitioned into chains (each a known total order, index 0 =
    greatest). For every item x and chain c we keep two integers:
      down[x][c] = smallest chain position that x is known to beat
      up[x][c]   = largest chain position known to beat x
    so "x > y?" is one comparison, |greater[x]| / |U[x]| cost O(w), and memory
    is O(n*w) for w chains rather than O(n^2) for greater/less/unknown sets.
    ``greater``, ``less`` and ``unknown`` remain available as read-only views.
    """

    def __init__(self, items: Sequence[str], verbose: bool = False, use_transitivity: bool = True):
        if not use_transitivity:
            raise ValueError("ChainTransitiveOrder requires use_transitivity=True")
        self.items = list(items)
        self._index: Dict[str, int] = {x: i for i, x in enumerate(self.items)}
        self.chain_of: Dict[str, int] = {}
        self.pos: Dict[str, int] = {}
        self.chain_members: List[List[str]] = []
        self.chain_head: List[int] = []
        self.chain_tail: List[int] = []
        self.down: Dict[str, List[int]] = {x: [] for x in self.items}
        self.up: Dict[str, List[int]] = {x: [] for x in self.items}
        n = len(self.items)
        self._total_pairs = n * (n - 1) // 2
        self._known = 0
        self.rechain_count = 0
        self.greater = _ChainViewMap(self, _ChainBelowView)  # type: ignore[assignment]
        self.less = _ChainViewMap(self, _ChainAboveView)  # type: ignore[assignment]
        self.unknown = _ChainViewMap(self, _ChainUnknownView)  # type: ignore[assignment]
        self.verbose = verbose
        self.use_transitivity = True
        self.direct_relations = 0
        self.transitive_relations = 0
        self.direct_relation_pairs: List[Tuple[str, str]] = []
        self.rejected_contradictions_count = 0
        self.rejected_contradictions_list: List[Dict[str, Any]] = []
        self.rejected_pair_flips_count = 0
        self.rejected_transitivity_cycles_count = 0

    # ------------------------------ index queries ------------------------------

    @property
    def u(self) -> int:
        return self._total_pairs - self._known

    @property
    def num_chains(self) -> int:
        return len(self.chain_members)

    def label_entries(self) -> int:
        """Stored integers: two labels per item per chain."""
        return 2 * len(self.items) * self.num_chains

    def known_pairs(self) -> int:
        return self._known

    def is_greater(self, x: str, y: str) -> bool:
        c = self.chain_of.get(y)
        if c is None or x == y:
            return False
        return self.down[x][c] <= self.pos[y]

    def comparable(self, x: str, y: str) -> bool:
        if x == y:
            return False
        cy = self.chain_of.get(y)
        cx = self.chain_of.get(x)
        if cy is None or cx is None:
            return False
        return self.down[x][cy] <= self.pos[y] or self.down[y][cx] <= self.pos[x]

    def _chain_tail(self, c: int) -> int:
        return self.chain_tail[c]

    def count_below(self, x: str) -> int:
        return sum(t - d + 1 for d, t in zip(self.down[x], self.chain_tail) if d <= t)

    def count_above(self, x: str) -> int:
        return sum(v - h + 1 for v, h in zip(self.up[x], self.chain_head) if v >= h)

    def unresolved_degree(self, x: str) -> int:
        """|U[x]| = (n-1) - |greater[x]| - |less[x]|, computed in O(w)."""
        return len(self.items) - 1 - self.count_below(x) - self.count_above(x)

    # ----------------------------- index maintenance ----------------------------

    def _new_chain(self, members: Sequence[str]) -> None:
        c = len(self.chain_members)
        self.chain_members.append(list(members))
        self.chain_head.append(0)
        self.chain_tail.append(len(members) - 1)
        for x in self.items:
            self.down[x].append(_CHAIN_INF)
            self.up[x].append(-_CHAIN_INF)
        for p, x in enumerate(members):
            self.chain_of[x] = c
            self.pos[x] = p
            self.down[x][c] = p + 1
            self.up[x][c] = p - 1

    def _append(self, c: int, x: str) -> None:
        p = self.chain_tail[c] + 1
        self.chain_members[c].append(x)
        self.chain_tail[c] = p
        self.chain_of[x], self.pos[x] = c, p
        self.down[x][c] = p + 1
        self.up[x][c] = p - 1

    def _prepend(self, c: int, x: str) -> None:
        p = self.chain_head[c] - 1
        self.chain_members[c].insert(0, x)
        self.chain_head[c] = p
        self.chain_of[x], self.pos[x] = c, p
        self.down[x][c] = p + 1
        self.up[x][c] = p - 1

    def _place_for_edge(self, higher: str, lower: str, n_ancestors: int) -> int:
        """Put unplaced endpoints on a chain; unplaced items have no relations yet.

        Returns how many new higher>lower pairs the placement alone already makes
        visible through existing labels (they are counted before label updates).
        """
        hi_c = self.chain_of.get(higher)
        lo_c = self.chain_of.get(lower)
        if hi_c is not None and lo_c is not None:
            return 0
        if hi_c is None and lo_c is None:
            self._new_chain([higher, lower])
            return 1
        if lo_c is None:
            if self.pos[higher] == self.chain_tail[hi_c]:  # type: ignore[index]
                self._append(hi_c, lower)  # type: ignore[arg-type]
                return n_ancestors
            self._new_chain([lower])
            return 0
        if self.pos[lower] == self.chain_head[lo_c]:
            self._prepend(lo_c, higher)
            return len(self.chain_members[lo_c]) - 1
        self._new_chain([higher])
        return 0

    def _ancestors_or_self(self, x: str) -> List[str]:
        if x not in self.chain_of:
            return [x]
        c, p = self.chain_of[x], self.pos[x]
        down = self.down
        return [a for a in self.items if down[a][c] <= p or a == x]

    def _descendants_or_self(self, x: str) -> List[str]:
        if x not in self.chain_of:
            return [x]
        c, p = self.chain_of[x], self.pos[x]
        up = self.up
        return [b for b in self.items if up[b][c] >= p or b == x]

    def add_relation(self, higher: str, lower: str, direct: bool = False) -> None:
        if higher == lower or self.is_greater(higher, lower):
            return
        if self.is_greater(lower, higher):
            super().add_relation(higher, lower, direct=direct)  # records the rejection
            return

        ancestors = self._ancestors_or_self(higher)
        descendants = self._descendants_or_self(lower)
        new_pairs = self._place_for_edge(higher, lower, len(ancestors))

        lo_c, lo_p = self.chain_of[lower], self.pos[lower]
        hi_c, hi_p = self.chain_of[higher], self.pos[higher]
        target_down = list(self.down[lower])
        if lo_p < target_down[lo_c]:
            target_down[lo_c] = lo_p
        target_up = list(self.up[higher])
        if hi_p > target_up[hi_c]:
            target_up[hi_c] = hi_p
        tails = self.chain_tail
        down_changes = [(c, t) for c, t in enumerate(target_down) if t <= tails[c]]
        up_changes = [(c, t) for c, t in enumerate(target_up) if t != -_CHAIN_INF]

        for a in ancestors:
            labels = self.down[a]
            for c, t in down_changes:
                old = labels[c]
                if t < old:
                    tail = tails[c]
                    new_pairs += (tail - t + 1) - (tail - old + 1 if old <= tail else 0)
                    labels[c] = t
        for b in descendants:
            labels = self.up[b]
            for c, t in up_changes:
                if t > labels[c]:
                    labels[c] = t

        self._known += new_pairs

        if direct:
            self.direct_relations += 1
            self.direct_relation_pairs.append((higher, lower))
            self.transitive_relations += max(0, new_pairs - 1)
            if self.verbose:
                print(f"  Direct: {higher} > {lower} (+{max(0, new_pairs - 1)} via transitivity)")
        else:
            self.transitive_relations += new_pairs

    def add_total_order(self, ordered_items: Sequence[str], direct: bool = False) -> None:
        # Adjacent pairs first so unplaced items extend one chain instead of
        # scattering into singletons; the closure is identical either way.
        for i in range(len(ordered_items) - 1):
            self.add_relation(ordered_items[i], ordered_items[i + 1], direct=direct)
        for i, higher in enumerate(ordered_items):
            for lower in ordered_items[i + 2 :]:
                self.add_relation(higher, lower, direct=direct)

    def _mark_known(self, hi: str, lo: str) -> None:
        return

    def residual_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for i, x in enumerate(self.items):
            for y in self.items[i + 1 :]:
                if not self.comparable(x, y):
                    pairs.append((x, y) if x < y else (y, x))
        return pairs

    def rechain(self) -> int:
        """Rebuild a smaller greedy chain cover from the current closure.

        Walks items in a linear extension and appends each to the first chain
        whose tail beats it. Queries against the old labels need no extra
        memory beyond the new O(n*w') labels. Returns the new chain count.
        """
        in_degree = {x: self.count_above(x) for x in self.items}
        ready = sorted(x for x, d in in_degree.items() if d == 0)
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for succ in list(self.greater[node]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)
            ready.sort()
        if len(order) != len(self.items):
            return self.num_chains

        new_chains: List[List[str]] = []
        for x in order:
            for chain in new_chains:
                if self.is_greater(chain[-1], x):
                    chain.append(x)
                    break
            else:
                new_chains.append([x])
        if len(new_chains) >= self.num_chains:
            return self.num_chains

        new_chain_of = {x: c for c, chain in enumerate(new_chains) for x in chain}
        new_pos = {x: p for chain in new_chains for p, x in enumerate(chain)}
        w = len(new_chains)
        new_down: Dict[str, List[int]] = {}
        new_up: Dict[str, List[int]] = {}
        is_greater = self.is_greater
        for x in self.items:
            d: List[int] = [_CHAIN_INF] * w
            v: List[int] = [-_CHAIN_INF] * w
            for c, chain in enumerate(new_chains):
                # Chains are descending, so "x beats chain[p]" holds on a suffix
                # and "chain[p] beats x" on a prefix: binary-search both ends.
                lo, hi = 0, len(chain)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if is_greater(x, chain[mid]):
                        hi = mid
                    else:
                        lo = mid + 1
                if lo < len(chain):
                    d[c] = lo
                lo, hi = 0, len(chain)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if is_greater(chain[mid], x):
                        lo = mid + 1
                    else:
                        hi = mid
                if lo > 0:
                    v[c] = lo - 1
            c0, p0 = new_chain_of[x], new_pos[x]
            d[c0] = p0 + 1
            v[c0] = p0 - 1
            new_down[x], new_up[x] = d, v

        self.chain_members = new_chains
        self.chain_head = [0] * w
        self.chain_tail = [len(chain) - 1 for chain in new_chains]
        self.chain_of, self.pos = new_chain_of, new_pos
        self.down, self.up = new_down, new_up
        self.rechain_count += 1
        return w


def make_transitive_order(
    items: Sequence[str],
    *,
    backend: str = "sets",
    verbose: bool = False,
    use_transitivity: bool = True,
) -> TransitiveOrder:
    if backend == "chain":
        return ChainTransitiveOrder(items, verbose=verbose, use_transitivity=use_transitivity)
    if backend == "sets":
        return TransitiveOrder(items, verbose=verbose, use_transitivity=use_transitivity)
    if backend == "dag":
        from .reach_index import DagTransitiveOrder

        return DagTransitiveOrder(items, verbose=verbose, use_transitivity=use_transitivity)  # type: ignore[return-value]
    raise ValueError("graph_backend must be 'sets', 'chain' or 'dag'")


def print_rejected_contradictions_table(
    rejected_list: List[Dict[str, Any]],
    rejected_count: int,
    title: str = "REJECTED CONTRADICTIONS (cycle prevention)",
    pair_flips_count: Optional[int] = None,
    transitivity_cycles_count: Optional[int] = None,
) -> None:
    """Print a tabular summary of rejected relations that would have created a cycle."""
    if not rejected_list:
        return
    n_pair = pair_flips_count if pair_flips_count is not None else sum(1 for r in rejected_list if r.get("cycle_type") == "Pair flip")
    n_trans = transitivity_cycles_count if transitivity_cycles_count is not None else sum(1 for r in rejected_list if r.get("cycle_type") == "Transitivity cycle")
    print("\n" + "=" * 110)
    print(title)
    print(f"Total rejected: {rejected_count}  (Pair flips: {n_pair}, Transitivity cycles: {n_trans})")
    print("=" * 110)
    col1 = "Rejected relation (LLM said)"
    col2 = "Existing chain"
    col3 = "Cycle would have been"
    col4 = "Cycle type"
    col5 = "Items in cycle"
    w1, w2, w3, w4 = 26, 32, 38, 18
    print(f"  {col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}} | {col4:<{w4}} | {col5}")
    print("-" * 110)
    for r in rejected_list:
        rej = r.get("rejected_pair", "")
        chain = r.get("existing_chain", "")
        cycle = r.get("cycle_would_be", "")
        ctype = r.get("cycle_type", "")
        items = r.get("items_in_cycle", [])
        items_str = ", ".join(items) if isinstance(items, list) else str(items)
        print(f"  {rej:<{w1}} | {chain:<{w2}} | {cycle:<{w3}} | {ctype:<{w4}} | {items_str}")
    print("=" * 110 + "\n")


def check_stored_relations_acyclic(graph: "TransitiveOrder") -> Tuple[bool, Optional[List[str]]]:
    """
    Check whether the relation graph (direct + transitive) contains a cycle.
    Returns (True, None) if acyclic, (False, cycle_list) if a cycle was found.
    Uses DFS to detect back edges; cycle_list is the sequence of nodes in the cycle (e.g. [v, ..., u] for cycle v -> ... -> u -> v).
    """
    nodes = list(graph.items)
    if not nodes:
        return True, None

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in nodes}
    parent: Dict[str, Optional[str]] = {n: None for n in nodes}
    back_edge_end: Optional[Tuple[str, str]] = None  # (u, v) where u -> v is the back edge

    def dfs(u: str) -> bool:
        nonlocal back_edge_end
        color[u] = GRAY
        for v in graph.greater[u]:
            if v not in color:
                continue
            if color[v] == GRAY:
                back_edge_end = (u, v)
                return True
            if color[v] == WHITE:
                parent[v] = u
                if dfs(v):
                    return True
        color[u] = BLACK
        return False

    for node in nodes:
        if color[node] == WHITE:
            if dfs(node):
                break
    if back_edge_end is None:
        return True, None
    u, v = back_edge_end
    path_uv: List[str] = [u]
    cur = u
    while cur != v and parent.get(cur) is not None:
        cur = parent[cur]
        path_uv.append(cur)
    cycle_list: List[str] = list(reversed(path_uv))
    return False, cycle_list if len(cycle_list) >= 2 else None


def check_graph_pair_flips(graph: "TransitiveOrder") -> Tuple[int, List[Tuple[str, str]]]:
    """
    Check the graph for pair flips: pairs (A, B) where both A > B and B > A are stored.
    Returns (count, list of (item_a, item_b) for each affected pair).
    """
    pairs: List[Tuple[str, str]] = []
    items = list(graph.items)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if b in graph.greater[a] and a in graph.greater[b]:
                pairs.append((a, b))
    return len(pairs), pairs

