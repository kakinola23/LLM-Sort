"""Transitive Batched Sort (TBS): Rounds 1-3 over a batch grid with transitive reuse."""

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

from .partition import apply_prompt_order, build_round1_batches
from .prompts import NBA_KNOWLEDGE_COMPARISON_PROMPT, NBA_KNOWLEDGE_COMPARISON_PROMPT_RESCUED, ZHAO_SYSTEM_PROMPT, _zhao_numbered_nba_keys
from .transitive import ChainTransitiveOrder, TransitiveOrder, make_transitive_order




@dataclass
class CallStats:
    count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    history: List[Dict[str, object]] = field(default_factory=list)

    def update(self, batch: Sequence[str], response) -> None:  # type: ignore[override]
        """Track usage stats per LLM call."""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        self.count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.history.append(
            {
                "batch": list(batch),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_response": getattr(response.choices[0].message, "content", ""),  # type: ignore[attr-defined]
            }
        )


@dataclass
class Round3StopStats:
    """Logged at end of Round 3 for early-stop experiments."""

    stop_reason: str = "not_run"
    completion_threshold: float = 0.9
    max_no_progress: int = 2
    enable_completion_early_stop: bool = True
    enable_stall_early_stop: bool = True
    known_pairs: int = 0
    total_pairs: int = 0
    rho: float = 0.0
    unknown_pairs: int = 0
    round3_llm_calls: int = 0
    round3_batches_planned: int = 0
    round3_batches_executed: int = 0
    round3_batches_skipped_known: int = 0

    def summary(self) -> Dict[str, int]:
        return {
            "api_calls": self.count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }






class BatchedLLMSorter:
    def __init__(
        self,
        items: Sequence[str],
        batch_size: int,
        client: Optional[Any] = None,  # Accepts OpenAI or GeminiClient
        model: Optional[str] = None,
        verbose: bool = False,
        sort_by: str = "height",  # "alphabetical" or "height"
        height_map: Optional[Dict[str, float]] = None,  # Required if sort_by="height"
        height_prompt_mode: str = "explicit",  # "explicit" (heights in prompt) or "knowledge" (Zhao-style)
        completion_threshold: float = 0.9,
        max_no_progress: int = 2,
        enable_completion_early_stop: bool = True,
        enable_stall_early_stop: bool = True,
        round1_partition: str = "contiguous",
        prompt_order: str = "input",
        partition_seed: int = 42,
        text_map: Optional[Dict[str, str]] = None,
        use_transitivity: bool = True,
        round2_all_rows: bool = False,
        round2_mode: str = "windows",
        round3_dense: bool = False,
        round3_mode: str = "diagonals",
        listwise_votes: int = 1,
        grid_update_mode: str = "none",
        graph_backend: str = "sets",
        structured_output: bool = False,
        structured_schema: str = "reasoning",
    ) -> None:
        if batch_size < 2:
            raise ValueError("Batch size must be at least 2.")
        
        if sort_by == "height" and height_map is None:
            raise ValueError("height_map is required when sort_by='height'")
        if round2_mode not in ("windows", "recursive"):
            raise ValueError("round2_mode must be 'windows' or 'recursive'")
        if round3_mode not in (
            "diagonals",
            "unresolved",
            "unresolved_greedy",
            "diagonals_then_unresolved",
        ):
            raise ValueError(
                "round3_mode must be 'diagonals', 'unresolved', 'unresolved_greedy', "
                "or 'diagonals_then_unresolved'"
            )
        if grid_update_mode not in ("none", "reorder_columns", "rebuild"):
            raise ValueError(
                "grid_update_mode must be 'none', 'reorder_columns', or 'rebuild'"
            )

        self.items = list(items)
        self.batch_size = batch_size
        if client is None or not model:
            raise ValueError("client and model are required (see tbs.llm_clients.make_client)")
        self.client = client
        self.model = model
        self.verbose = verbose
        self.sort_by = sort_by
        self.height_map = height_map or {}
        if height_prompt_mode not in ("explicit", "knowledge", "knowledge_rescued"):
            raise ValueError(
                "height_prompt_mode must be 'explicit', 'knowledge', or 'knowledge_rescued'"
            )
        self.height_prompt_mode = height_prompt_mode
        self.use_transitivity = use_transitivity
        self.round2_all_rows = round2_all_rows
        self.round2_mode = round2_mode
        self.round3_dense = round3_dense
        self.round3_mode = round3_mode
        if listwise_votes < 1:
            raise ValueError("listwise_votes must be >= 1")
        self.listwise_votes = listwise_votes
        self.structured_output = structured_output
        if structured_schema not in ("reasoning", "recall"):
            raise ValueError("structured_schema must be 'reasoning' or 'recall'")
        self.structured_schema = structured_schema
        self.grid_update_mode = grid_update_mode
        self.grid_update_applied = False

        self.graph_backend = graph_backend
        self.graph = make_transitive_order(
            self.items, backend=graph_backend, verbose=verbose, use_transitivity=use_transitivity
        )
        self.chain_stats: Dict[str, int] = {}
        self.stats = CallStats()
        self.completion_threshold = completion_threshold
        self.max_no_progress = max_no_progress
        self.enable_completion_early_stop = enable_completion_early_stop
        self.enable_stall_early_stop = enable_stall_early_stop
        self.round1_partition = round1_partition
        self.prompt_order = prompt_order
        self.partition_seed = partition_seed
        self.text_map = text_map or {}
        self.round3_stats = Round3StopStats(
            completion_threshold=completion_threshold,
            max_no_progress=max_no_progress,
            enable_completion_early_stop=enable_completion_early_stop,
            enable_stall_early_stop=enable_stall_early_stop,
        )
        self.round1_calls = 0
        self.round2_calls = 0

    def known_pair_counts(self) -> Tuple[int, int, float]:
        """Return (known_pairs, total_pairs, rho)."""
        total_pairs = len(self.graph.items) * (len(self.graph.items) - 1) // 2
        known_pairs = self.graph.known_pairs()
        rho = (known_pairs / total_pairs) if total_pairs > 0 else 1.0
        return known_pairs, total_pairs, rho

    # ------------------------------- public API --------------------------------

    def sort(self) -> Tuple[List[str], CallStats]:
        calls_before = self.stats.count
        columns = self._round_one()
        self.round1_calls = self.stats.count - calls_before
        if self.verbose and self.graph.is_complete():
            print("\n  Total order complete after Round 1!")
            self.round2_calls = 0
            return self.graph.total_order(), self.stats
        
        calls_before_r2 = self.stats.count
        self._round_two(columns)
        self.round2_calls = self.stats.count - calls_before_r2
        if self.verbose and self.graph.is_complete():
            print("\n  Total order complete after Round 2!")
            return self.graph.total_order(), self.stats

        if isinstance(self.graph, ChainTransitiveOrder):
            self.chain_stats["chains_after_r2"] = self.graph.num_chains
            self.chain_stats["label_ints_after_r2"] = self.graph.label_entries()
            self.graph.rechain()
            self.chain_stats["chains_after_rechain"] = self.graph.num_chains
            self.chain_stats["label_ints_after_rechain"] = self.graph.label_entries()

        if self.grid_update_mode != "none":
            columns = self._update_grid_after_r2(columns)
        
        self._round_three(columns)
        return self.graph.total_order(), self.stats

    def _quiet_total_order(self) -> List[str]:
        """Kahn linear extension without the verbose tie dump (for grid rebuild)."""
        in_degree = {item: len(self.graph.less[item]) for item in self.graph.items}
        ready = sorted(x for x, d in in_degree.items() if d == 0)
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for succ in list(self.graph.greater[node]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)
            ready.sort()
        if len(order) != len(self.graph.items):
            # Fallback: append any leftover alphabetically (should not happen if DAG).
            leftover = sorted(set(self.graph.items) - set(order))
            order.extend(leftover)
        return order

    def _update_grid_after_r2(self, columns: List[List[str]]) -> List[List[str]]:
        """
        Optionally rearrange the batch grid before Round 3.

        none (caller): keep post–Round-1 layout (stationary).
        reorder_columns: permute columns so heads follow current graph order.
        rebuild: refill columns from the current linear extension in chunks of m.
        """
        mode = self.grid_update_mode
        if mode == "none" or not columns:
            return columns

        before_heads = [col[0] for col in columns if col]
        if mode == "reorder_columns":
            rank = {item: i for i, item in enumerate(self._quiet_total_order())}
            new_columns = sorted(
                columns,
                key=lambda col: (rank.get(col[0], 10**9), col[0] if col else ""),
            )
            after_heads = [col[0] for col in new_columns if col]
            self.grid_update_applied = before_heads != after_heads
            if self.verbose:
                print("\n" + "=" * 70)
                print("GRID UPDATE after R2: reorder_columns")
                print("=" * 70)
                print(f"  heads before: {before_heads[:8]}{'...' if len(before_heads) > 8 else ''}")
                print(f"  heads after:  {after_heads[:8]}{'...' if len(after_heads) > 8 else ''}")
                print(f"  changed={self.grid_update_applied}")
            return new_columns

        if mode == "rebuild":
            order = self._quiet_total_order()
            m = self.batch_size
            new_columns = [order[i : i + m] for i in range(0, len(order), m)]
            after_heads = [col[0] for col in new_columns if col]
            self.grid_update_applied = True
            if self.verbose:
                print("\n" + "=" * 70)
                print("GRID UPDATE after R2: rebuild from current linear extension")
                print("=" * 70)
                print(f"  n={len(order)} m={m} -> {len(new_columns)} columns")
                print(f"  heads after: {after_heads[:8]}{'...' if len(after_heads) > 8 else ''}")
            return new_columns

        return columns

    # ------------------------------- rounds ------------------------------------

    def _round_one(self) -> List[List[str]]:
        if self.verbose:
            print("\n" + "="*70)
            print("ROUND 1: Sort items into columns (batches)")
            print("="*70)
        
        columns: List[List[str]] = []
        from .partition import build_round1_batches

        round1_batches = build_round1_batches(
            self.items,
            self.batch_size,
            self.round1_partition,
            seed=self.partition_seed,
            height_map=self.height_map if self.sort_by == "height" else None,
            text_map=self.text_map if self.text_map else None,
        )
        for batch in round1_batches:
            
            if len(batch) == 1:
                if self.verbose:
                    print(f"\n  Column {len(columns)}: Skipping batch {batch} (only 1 element, nothing to sort)")
                columns.append(batch)
                continue
            
            if self.verbose:
                print(f"\n  Column {len(columns)}: Sorting batch")
                print(f"    Original batch: {batch}")
            
            ordered = self._sort_with_llm(batch, round_id=1, payload={"column": len(columns)})
            
            if self.verbose:
                print(f"    LLM output:     {ordered}")
            
            # Six-check validation; a failed batch is repaired before use.
            is_valid, validation_report = self._validate_batch_output(ordered, batch)
            
            if not is_valid:
                if self.verbose:
                    print(f"FAILED (hallucinations detected)")
                    print(f"\n    HALLUCINATION DETECTED:")
                    for violation in validation_report["violations"]:
                        print(f"      - {violation}")
                    if validation_report["missing_items"]:
                        print(f"      Missing items: {validation_report['missing_items']}")
                    if validation_report["extra_items"]:
                        print(f"      Extra items: {validation_report['extra_items']}")
                    if validation_report["duplicates"]:
                        print(f"      Duplicates: {validation_report['duplicates']}")
                    print(f"\n    FIX REQUIRED: Using fix implementation to correct hallucinations...")
                
                if self.verbose:
                    print(f"\n    Applying corrections:")
                ordered = self._correct_batch_output(ordered, batch, validation_report)
                
                if self.verbose:
                    print(f"\n    FIX APPLIED: Corrected output: {ordered}")
            else:
                if self.verbose:
                    print(f"PASSED (all checks passed - no hallucinations detected)")
                    print(f"    FIX NOT NEEDED: LLM output is valid, no corrections required")
            
            if self.verbose:
                print(f"    -> Result: {' > '.join(ordered)}")
            
            columns.append(ordered)
            before_direct = self.graph.direct_relations
            before_trans = self.graph.transitive_relations if self.verbose else 0  # logging only
            self.graph.add_total_order(ordered, direct=True)
            direct_added = self.graph.direct_relations - before_direct
            trans_added = (self.graph.transitive_relations - before_trans) if self.verbose else 0
            if self.verbose:
                print(f"    -> Added {direct_added} direct relations, {trans_added} via transitivity")
        return columns

    def _batch_pairs_known(self, batch: Sequence[str]) -> bool:
        if not self.use_transitivity:
            return False
        for i, hi in enumerate(batch):
            for lo in batch[i + 1 :]:
                if not self.graph.comparable(hi, lo):
                    return False
        return True

    def _order_batch_from_graph(self, batch: Sequence[str]) -> List[str]:
        """Linear extension of known relations among batch (ascending / first-first)."""
        items = list(batch)
        in_degree = {
            x: sum(1 for y in items if y != x and x in self.graph.greater[y])
            for x in items
        }
        ready = sorted([x for x in items if in_degree[x] == 0])
        out: List[str] = []
        while ready:
            x = ready.pop(0)
            out.append(x)
            for y in items:
                if y in self.graph.greater[x]:
                    in_degree[y] -= 1
                    if in_degree[y] == 0:
                        ready.append(y)
                        ready.sort()
        if len(out) != len(items):
            # Cycle / incomplete — fall back to input order for missing
            missing = [x for x in items if x not in out]
            out.extend(missing)
        return out

    def _sort_batch_round2(self, batch: Sequence[str], row_id: int) -> List[str]:
        """Listwise sort a Round-2 batch; skip LLM if all pairs already known."""
        if len(batch) <= 1:
            return list(batch)
        if self._batch_pairs_known(batch):
            ordered = self._order_batch_from_graph(batch)
            if self.verbose:
                print(f"    skip LLM (known): {' > '.join(ordered)}")
            return ordered
        ordered = self._sort_with_llm(batch, round_id=2, payload={"row": row_id, "mode": self.round2_mode})
        self.graph.add_total_order(ordered, direct=True)
        if self.verbose:
            print(f"    -> {' > '.join(ordered)}")
        return ordered

    def _merge_two_sorted_runs(self, left: List[str], right: List[str], row_id: int) -> List[str]:
        """Zhao-style listwise merge of two ascending runs (size bound = batch_size)."""
        if not left:
            return right[:]
        if not right:
            return left[:]

        # Work descending (greatest/first-in-order at front), matching OrderByLLM
        l1 = left[::-1]
        l2 = right[::-1]
        half = max(self.batch_size // 2, 1)
        list_i, list_j = 0, 0
        merged: List[str] = []
        buffer: List[str] = []
        membership: Dict[str, List[str]] = defaultdict(list)
        buffer_metadata = {"l1": 0, "l2": 0}
        done_early = False

        def fix_duplicates(buf: List[str], original: List[str]) -> List[str]:
            if set(buf) == set(original) and all(buf.count(x) == original.count(x) for x in set(original)):
                return buf
            target = Counter(original)
            result: List[str] = []
            used: Counter = Counter()
            for item in buf:
                if used[item] < target[item]:
                    result.append(item)
                    used[item] += 1
            for item, count in target.items():
                while used[item] < count:
                    result.append(item)
                    used[item] += 1
            return result

        while list_i < len(l1) or list_j < len(l2):
            if list_i >= len(l1) and buffer_metadata["l1"] == 0:
                merged.extend(buffer + l2[list_j:])
                done_early = True
                break
            if list_j >= len(l2) and buffer_metadata["l2"] == 0:
                merged.extend(buffer + l1[list_i:])
                done_early = True
                break

            while buffer_metadata["l1"] < half and list_i < len(l1):
                item = l1[list_i]
                buffer.append(item)
                membership[item].append("l1")
                buffer_metadata["l1"] += 1
                list_i += 1
            while buffer_metadata["l2"] < half and list_j < len(l2):
                item = l2[list_j]
                buffer.append(item)
                membership[item].append("l2")
                buffer_metadata["l2"] += 1
                list_j += 1

            original_buffer = buffer[:]
            # LLM returns ascending; flip buffer to ascending for the call
            ascending = self._sort_batch_round2(buffer[::-1], row_id=row_id)
            ascending = fix_duplicates(ascending, original_buffer)
            buffer = ascending[::-1]

            items_taken = 0
            for key in buffer:
                if not membership[key]:
                    continue
                source = membership[key].pop()
                buffer_metadata[source] -= 1
                merged.append(key)
                items_taken += 1
                if buffer_metadata[source] == 0:
                    break
            buffer = buffer[items_taken:]

        if buffer and not done_early:
            merged.extend(buffer)
        # Back to ascending
        return merged[::-1]

    def _recursive_merge_sort_row(self, row_items: List[str], row_id: int) -> List[str]:
        """Fully sort one grid row with Zhao-style external merge (chunk size = m)."""
        m = self.batch_size
        if len(row_items) <= m:
            return self._sort_batch_round2(row_items, row_id=row_id)

        chunks = [row_items[i : i + m] for i in range(0, len(row_items), m)]
        if self.verbose:
            print(f"  Recursive R{row_id}: {len(row_items)} items -> {len(chunks)} chunks")
        sorted_chunks = [self._sort_batch_round2(ch, row_id=row_id) for ch in chunks]

        while len(sorted_chunks) > 1:
            next_level: List[List[str]] = []
            for i in range(0, len(sorted_chunks), 2):
                if i + 1 < len(sorted_chunks):
                    if self.verbose:
                        print(
                            f"  Merge R{row_id}: runs of "
                            f"{len(sorted_chunks[i])} + {len(sorted_chunks[i + 1])}"
                        )
                    next_level.append(
                        self._merge_two_sorted_runs(
                            sorted_chunks[i], sorted_chunks[i + 1], row_id=row_id
                        )
                    )
                else:
                    next_level.append(sorted_chunks[i])
            sorted_chunks = next_level

        ordered = sorted_chunks[0]
        # Ensure full row order is in G (merge already added buffer edges)
        self.graph.add_total_order(ordered, direct=True)
        return ordered

    def _round_two(self, columns: List[List[str]]) -> None:
        """Round 2: sliding windows or recursive merge on R0 (or all rows)."""
        if self.verbose:
            print("\n" + "=" * 70)
            scope = "all rows" if self.round2_all_rows else "R0 only"
            print(f"ROUND 2: {self.round2_mode} ({scope})")
            print("=" * 70)

        max_height = max((len(col) for col in columns), default=0)
        row_indices = range(max_height) if self.round2_all_rows else range(min(1, max_height))

        for row_id in row_indices:
            row_items = [col[row_id] for col in columns if len(col) > row_id]
            if len(row_items) < 2:
                continue

            if self.verbose:
                print(f"\n  Row R{row_id}: {len(row_items)} items")

            if self._batch_pairs_known(row_items):
                if self.verbose:
                    print(f"  Skipping R{row_id} (all pairs already ordered)")
                continue

            if self.graph.is_complete():
                if self.verbose:
                    print("\n  Total order complete!")
                return

            if self.round2_mode == "recursive":
                self._recursive_merge_sort_row(row_items, row_id=row_id)
                continue

            window_idx = 0
            for window in self._row_windows(row_items):
                if self.graph.is_complete():
                    if self.verbose:
                        print("\n  Total order complete! Skipping remaining windows.")
                    return

                if self._batch_pairs_known(window):
                    if self.verbose:
                        print(
                            f"\n  R{row_id} window {window_idx}: Skipping "
                            f"(all relations already known)"
                        )
                    window_idx += 1
                    continue

                if self.verbose:
                    print(f"\n  R{row_id} window {window_idx}: Sorting {window}")
                ordered = self._sort_with_llm(
                    window, round_id=2, payload={"row": row_id}
                )
                if self.verbose:
                    print(f"    -> Result: {' > '.join(ordered)}")
                before_direct = self.graph.direct_relations
                before_trans = self.graph.transitive_relations if self.verbose else 0  # logging only
                self.graph.add_total_order(ordered, direct=True)
                direct_added = self.graph.direct_relations - before_direct
                trans_added = (self.graph.transitive_relations - before_trans) if self.verbose else 0
                if self.verbose:
                    print(
                        f"    → Added {direct_added} direct relations, "
                        f"{trans_added} via transitivity"
                    )
                window_idx += 1

    def _round_three(self, columns: List[List[str]]) -> None:
        if self.round3_mode == "unresolved":
            self._round_three_unresolved_pack()
            return
        if self.round3_mode == "unresolved_greedy":
            self._round_three_unresolved_greedy()
            return
        if self.round3_mode == "diagonals_then_unresolved":
            # Perfect-oracle Algorithm 1: diagonal stitching, then unresolved
            # packing (driven by the tracked U[.]/u neighborhoods) until u=0.
            self._round_three_diagonals(columns)
            if not self.graph.is_complete():
                self._round_three_unresolved_greedy()
            return
        self._round_three_diagonals(columns)

    def _pair_unknown(self, a: str, b: str) -> bool:
        return not self.graph.comparable(a, b)

    def _next_unresolved_batch(self) -> List[str]:
        """Greedy pack up to m items that participate in at least one unknown pair."""
        batch: List[str] = []
        seen = set()
        items = self.graph.items
        m = self.batch_size
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if not self._pair_unknown(a, b):
                    continue
                for x in (a, b):
                    if x not in seen:
                        seen.add(x)
                        batch.append(x)
                        if len(batch) >= m:
                            return batch
        return batch

    def _unresolved_degree(self, x: str) -> int:
        """Count of items x is not yet comparable to: O(1) via the tracked
        U[x] neighborhood (paper Sec 4.4) instead of an O(n) rescan."""
        return len(self.graph.unknown[x])

    def _next_unresolved_batch_greedy(self) -> List[str]:
        """Degree-prioritized, coverage-maximizing pack: seed with the item that
        currently has the most unresolved pairs, then greedily add whichever
        remaining item adds the most NEW unresolved pairs relative to what's
        already in the batch (not just "touches some unknown pair somewhere",
        like ``_next_unresolved_batch`` does)."""
        items = self.graph.items
        m = self.batch_size
        degrees = {x: self._unresolved_degree(x) for x in items}
        if not degrees or max(degrees.values()) == 0:
            return []
        seed = max(items, key=lambda x: degrees[x])
        batch = [seed]
        while len(batch) < m:
            best, best_gain = None, 0
            for cand in items:
                if cand in batch:
                    continue
                gain = sum(1 for b in batch if self._pair_unknown(cand, b))
                if gain > best_gain:
                    best, best_gain = cand, gain
            if best is None:
                break
            batch.append(best)
        return batch

    def _round_three_unresolved_greedy(self) -> None:
        """Round 3 without diagonals: degree-prioritized, coverage-maximizing
        unresolved-pair packing until Complete(G) / stall / theta."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("ROUND 3: unresolved-greedy packing (degree-prioritized coverage)")
            print("=" * 70)

        known0, total0, rho0 = self.known_pair_counts()
        if self.verbose:
            print(
                f"  After R1-R2 closure: rho={rho0:.4f} "
                f"known={known0}/{total0} unresolved={total0 - known0}"
            )

        round3_calls_start = self.stats.count
        consecutive_no_progress = 0
        stop_reason = "complete"
        step = 0
        max_no_progress = self.max_no_progress
        max_batches = max(1, (total0 - known0) // max(1, self.batch_size - 1) + 50)

        while not self.graph.is_complete():
            if step >= max_batches:
                stop_reason = "max_batches"
                break

            batch = self._next_unresolved_batch_greedy()
            if len(batch) < 2:
                stop_reason = "no_batch"
                break

            before_pairs = self.graph.known_pairs()
            if self.verbose:
                print(f"\n  Greedy batch {step}: Sorting {batch} ({len(batch)} items)")
            ordered = self._sort_with_llm(
                batch, round_id=3, payload={"unresolved_greedy": True, "step": step}
            )
            if self.verbose:
                print(f"    -> Result: {' > '.join(ordered)}")
            before_direct = self.graph.direct_relations
            before_trans = self.graph.transitive_relations if self.verbose else 0  # logging only
            self.graph.add_total_order(ordered, direct=True)
            direct_added = self.graph.direct_relations - before_direct
            trans_added = (self.graph.transitive_relations - before_trans) if self.verbose else 0
            after_pairs = self.graph.known_pairs()
            pairs_added = after_pairs - before_pairs
            if pairs_added == 0:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            if self.verbose:
                print(
                    f"    -> Added {direct_added} direct, {trans_added} transitive "
                    f"({pairs_added} total pairs)"
                )

            step += 1
            total_pairs = len(self.graph.items) * (len(self.graph.items) - 1) // 2
            completion_ratio = after_pairs / total_pairs if total_pairs > 0 else 0

            if (
                self.enable_stall_early_stop
                and max_no_progress > 0
                and consecutive_no_progress >= max_no_progress
            ):
                if self.verbose:
                    print(
                        f"\n  -> Stopping early: {consecutive_no_progress} "
                        "consecutive batches with no new pairs"
                    )
                stop_reason = "stall"
                break

            if (
                self.enable_completion_early_stop
                and completion_ratio >= self.completion_threshold
                and not self.graph.is_complete()
            ):
                if self.verbose:
                    print(
                        f"\n  -> Stopping early: {completion_ratio*100:.1f}% complete "
                        f"(threshold {self.completion_threshold*100:.0f}%)"
                    )
                stop_reason = "theta"
                break

        known_pairs, total_pairs, rho = self.known_pair_counts()
        if self.graph.is_complete():
            stop_reason = "complete"
        self.round3_stats = Round3StopStats(
            stop_reason=stop_reason,
            completion_threshold=self.completion_threshold,
            max_no_progress=self.max_no_progress,
            enable_completion_early_stop=self.enable_completion_early_stop,
            enable_stall_early_stop=self.enable_stall_early_stop,
            known_pairs=known_pairs,
            total_pairs=total_pairs,
            rho=rho,
            unknown_pairs=total_pairs - known_pairs,
            round3_llm_calls=self.stats.count - round3_calls_start,
            round3_batches_planned=step,
            round3_batches_executed=step,
            round3_batches_skipped_known=0,
        )
        if self.verbose:
            print(
                f"\n  Round 3 unresolved-greedy summary: stop={stop_reason}, rho={rho:.4f}, "
                f"known={known_pairs}/{total_pairs}, R3 calls={self.round3_stats.round3_llm_calls}"
            )

    def _round_three_unresolved_pack(self) -> None:
        """
        Round 3 without diagonals: after R1–R2 closure, pack remaining unknown
        pairs into listwise batches until Complete(G) / stall / θ.
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("ROUND 3: unresolved-pair packing (skip diagonals; infer-first)")
            print("=" * 70)

        known0, total0, rho0 = self.known_pair_counts()
        if self.verbose:
            print(
                f"  After R1–R2 closure: rho={rho0:.4f} "
                f"known={known0}/{total0} unresolved={total0 - known0}"
            )

        round3_calls_start = self.stats.count
        consecutive_no_progress = 0
        stop_reason = "complete"
        step = 0
        max_no_progress = self.max_no_progress
        # Cap batches so a noisy LLM cannot burn forever if stall fails.
        max_batches = max(1, (total0 - known0) // max(1, self.batch_size - 1) + 50)

        while not self.graph.is_complete():
            if step >= max_batches:
                stop_reason = "max_batches"
                break

            batch = self._next_unresolved_batch()
            if len(batch) < 2:
                stop_reason = "no_batch"
                break
            if self._batch_pairs_known(batch):
                # Defensive size-2 fallback
                pair = None
                items = self.graph.items
                for i, a in enumerate(items):
                    for b in items[i + 1 :]:
                        if self._pair_unknown(a, b):
                            pair = [a, b]
                            break
                    if pair:
                        break
                if not pair:
                    stop_reason = "no_unresolved"
                    break
                batch = pair

            before_pairs = self.graph.known_pairs()
            if self.verbose:
                print(f"\n  Unresolved batch {step}: Sorting {batch} ({len(batch)} items)")
            ordered = self._sort_with_llm(
                batch, round_id=3, payload={"unresolved_pack": True, "step": step}
            )
            if self.verbose:
                print(f"    -> Result: {' > '.join(ordered)}")
            before_direct = self.graph.direct_relations
            before_trans = self.graph.transitive_relations if self.verbose else 0  # logging only
            self.graph.add_total_order(ordered, direct=True)
            direct_added = self.graph.direct_relations - before_direct
            trans_added = (self.graph.transitive_relations - before_trans) if self.verbose else 0
            after_pairs = self.graph.known_pairs()
            pairs_added = after_pairs - before_pairs
            if pairs_added == 0:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            if self.verbose:
                print(
                    f"    → Added {direct_added} direct, {trans_added} transitive "
                    f"({pairs_added} total pairs)"
                )

            step += 1
            total_pairs = len(self.graph.items) * (len(self.graph.items) - 1) // 2
            completion_ratio = after_pairs / total_pairs if total_pairs > 0 else 0

            if (
                self.enable_stall_early_stop
                and max_no_progress > 0
                and consecutive_no_progress >= max_no_progress
            ):
                if self.verbose:
                    print(
                        f"\n  → Stopping early: {consecutive_no_progress} "
                        "consecutive batches with no new pairs"
                    )
                stop_reason = "stall"
                break

            if (
                self.enable_completion_early_stop
                and completion_ratio >= self.completion_threshold
                and not self.graph.is_complete()
            ):
                if self.verbose:
                    print(
                        f"\n  → Stopping early: {completion_ratio*100:.1f}% complete "
                        f"(threshold {self.completion_threshold*100:.0f}%)"
                    )
                stop_reason = "theta"
                break

        known_pairs, total_pairs, rho = self.known_pair_counts()
        if self.graph.is_complete():
            stop_reason = "complete"
        self.round3_stats = Round3StopStats(
            stop_reason=stop_reason,
            completion_threshold=self.completion_threshold,
            max_no_progress=self.max_no_progress,
            enable_completion_early_stop=self.enable_completion_early_stop,
            enable_stall_early_stop=self.enable_stall_early_stop,
            known_pairs=known_pairs,
            total_pairs=total_pairs,
            rho=rho,
            unknown_pairs=total_pairs - known_pairs,
            round3_llm_calls=self.stats.count - round3_calls_start,
            round3_batches_planned=step,
            round3_batches_executed=step,
            round3_batches_skipped_known=0,
        )
        if self.verbose:
            print(
                f"\n  Round 3 unresolved summary: stop={stop_reason}, rho={rho:.4f}, "
                f"known={known_pairs}/{total_pairs}, R3 calls={self.round3_stats.round3_llm_calls}"
            )

    def _round_three_diagonals(self, columns: List[List[str]]) -> None:
        if self.verbose:
            print("\n" + "="*70)
            mode = "dense" if self.round3_dense else "standard"
            print(f"ROUND 3: Sort diagonals to stitch partial orders ({mode})")
            print("="*70)
        
        num_cols = len(columns)
        max_height = max(len(col) for col in columns)
        
        all_diags = list(self._diagonals(columns, max_height, num_cols))
        
        # Batch small diagonals together to reduce API calls (unless dense mode:
        # keep longer diags separate, but pack size-2 ladders into groups of m).
        batched_diags = []
        current_batch = []
        current_batch_size = 0
        pack_small = not self.round3_dense
        
        for diag in all_diags:
            if len(diag) < 2:
                continue
            
            if self.use_transitivity:
                all_known = True
                for i, hi in enumerate(diag):
                    for lo in diag[i + 1:]:
                        if not self.graph.comparable(hi, lo):
                            all_known = False
                            break
                    if not all_known:
                        break

                if all_known:
                    continue  # Skip this diagonal
            
            if pack_small and len(diag) <= 3 and current_batch_size + len(diag) <= self.batch_size:
                current_batch.extend(diag)
                current_batch_size = len(set(current_batch))
            elif self.round3_dense and len(diag) == 2:
                # Pack ladders into listwise batches of size m
                if current_batch_size + 2 <= self.batch_size:
                    current_batch.extend(diag)
                    current_batch_size = len(set(current_batch))
                else:
                    if current_batch:
                        batched_diags.append(list(dict.fromkeys(current_batch)))
                    current_batch = list(diag)
                    current_batch_size = len(set(current_batch))
            else:
                if current_batch:
                    batched_diags.append(list(dict.fromkeys(current_batch)))
                    current_batch = []
                    current_batch_size = 0
                
                if pack_small and (len(diag) > 3 or len(diag) > self.batch_size - current_batch_size):
                    batched_diags.append(diag)
                elif pack_small:
                    current_batch.extend(diag)
                    current_batch_size = len(set(current_batch))
                else:
                    if len(diag) <= self.batch_size:
                        batched_diags.append(diag)
                    else:
                        for i in range(0, len(diag), self.batch_size):
                            chunk = diag[i : i + self.batch_size]
                            if len(chunk) >= 2:
                                batched_diags.append(chunk)
        
        if current_batch:
            batched_diags.append(list(dict.fromkeys(current_batch)))

        diag_idx = 0
        round3_calls_start = self.stats.count
        skipped_pre_llm = 0
        consecutive_no_progress = 0
        stop_reason = "exhausted_schedule"
        max_no_progress = self.max_no_progress

        for batch_diag in batched_diags:
            if self.graph.is_complete():
                if self.verbose:
                    print(f"\n  Total order complete! Skipping remaining {len(batched_diags) - diag_idx} batches.")
                stop_reason = "complete"
                break

            if self.use_transitivity:
                all_known_batch = True
                for i, hi in enumerate(batch_diag):
                    for lo in batch_diag[i + 1 :]:
                        if not self.graph.comparable(hi, lo):
                            all_known_batch = False
                            break
                    if not all_known_batch:
                        break
                if all_known_batch:
                    skipped_pre_llm += 1
                    continue

            before_pairs = self.graph.known_pairs()
            
            if self.verbose:
                print(f"\n  Batch {diag_idx}: Sorting {batch_diag} ({len(batch_diag)} items)")
            ordered = self._sort_with_llm(batch_diag, round_id=3, payload={"diag": True, "batched": len(batch_diag)})
            if self.verbose:
                print(f"    -> Result: {' > '.join(ordered)}")
            before_direct = self.graph.direct_relations
            before_trans = self.graph.transitive_relations if self.verbose else 0  # logging only
            self.graph.add_total_order(ordered, direct=True)
            direct_added = self.graph.direct_relations - before_direct
            trans_added = (self.graph.transitive_relations - before_trans) if self.verbose else 0
            
            after_pairs = self.graph.known_pairs()
            pairs_added = after_pairs - before_pairs
            
            if pairs_added == 0:
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0
            
            if self.verbose:
                print(f"    -> Added {direct_added} direct relations, {trans_added} via transitivity ({pairs_added} total pairs)")
            
            diag_idx += 1
            
            # Stop if no progress for too long OR if we're very close to completion
            total_pairs = len(self.graph.items) * (len(self.graph.items) - 1) // 2
            completion_ratio = after_pairs / total_pairs if total_pairs > 0 else 0
            
            if (
                self.enable_stall_early_stop
                and max_no_progress > 0
                and consecutive_no_progress >= max_no_progress
            ):
                if self.verbose:
                    print(f"\n  -> Stopping early: {consecutive_no_progress} consecutive batches with no new pairs")
                stop_reason = "stall"
                break

            if (
                self.enable_completion_early_stop
                and completion_ratio >= self.completion_threshold
                and not self.graph.is_complete()
            ):
                if self.verbose:
                    print(
                        f"\n  → Stopping early: {completion_ratio*100:.1f}% complete "
                        f"(threshold {self.completion_threshold*100:.0f}%)"
                    )
                stop_reason = "theta"
                break

        known_pairs, total_pairs, rho = self.known_pair_counts()
        if self.graph.is_complete():
            stop_reason = "complete"
        self.round3_stats = Round3StopStats(
            stop_reason=stop_reason,
            completion_threshold=self.completion_threshold,
            max_no_progress=self.max_no_progress,
            enable_completion_early_stop=self.enable_completion_early_stop,
            enable_stall_early_stop=self.enable_stall_early_stop,
            known_pairs=known_pairs,
            total_pairs=total_pairs,
            rho=rho,
            unknown_pairs=total_pairs - known_pairs,
            round3_llm_calls=self.stats.count - round3_calls_start,
            round3_batches_planned=len(batched_diags),
            round3_batches_executed=diag_idx,
            round3_batches_skipped_known=skipped_pre_llm,
        )

        if self.verbose:
            print(
                f"\n  Round 3 summary: stop={stop_reason}, rho={rho:.4f}, "
                f"known={known_pairs}/{total_pairs}, R3 calls={self.round3_stats.round3_llm_calls}"
            )

    # --------------------------- batching utilities ----------------------------

    def _row_windows(self, row_items: Sequence[str]) -> Iterable[List[str]]:
        if not row_items:
            return []
        if len(row_items) <= self.batch_size:
            return [list(row_items)]

        windows: List[List[str]] = []
        stride = self.batch_size - 1
        start = 0

        while start < len(row_items):
            end = start + self.batch_size
            if end > len(row_items) and len(row_items) >= self.batch_size:
                # Force the last batch to have exactly batch_size by taking the last m items.
                window = list(row_items[-self.batch_size :])
                windows.append(window)
                break
            windows.append(list(row_items[start:end]))
            start += stride
        return windows

    def _diagonals(
        self, columns: List[List[str]], max_height: int, num_cols: int
    ) -> Iterable[List[str]]:
        """Generate diagonals, prioritizing boundary-crossing diagonals first.

        If ``round3_dense`` is True, also emit anti-diagonals and adjacent-column
        ladder stitches for denser cross-column coverage.
        """
        max_diag_id = max_height + num_cols - 2  # inclusive
        
        # Prioritize critical diagonals (those that cross column boundaries)
        critical_diags = []
        
        for diag_id in range(1, max_diag_id):
            diag: List[str] = []
            cols_included = set()
            for row in range(max_height):
                col = diag_id - row
                if 0 <= col < num_cols and len(columns[col]) > row:
                    diag.append(columns[col][row])
                    cols_included.add(col)
            
            if len(diag) >= 2 and len(cols_included) > 1:
                critical_diags.append(diag)
        
        for diag in critical_diags:
            yield diag

        if not self.round3_dense:
            return

        # Anti-diagonals: cells with row + col = s
        anti: List[List[str]] = []
        for s in range(0, max_height + num_cols - 1):
            diag = []
            cols_included = set()
            for row in range(max_height):
                col = s - row
                if 0 <= col < num_cols and len(columns[col]) > row:
                    diag.append(columns[col][row])
                    cols_included.add(col)
            if len(diag) >= 2 and len(cols_included) > 1:
                anti.append(diag)
        for diag in anti:
            yield diag

        # Adjacent-column ladders: local stitches between C_j and C_{j+1}
        for j in range(num_cols - 1):
            h0, h1 = len(columns[j]), len(columns[j + 1])
            h = max(h0, h1)
            for i in range(h):
                for di, dj in ((0, 0), (0, 1), (1, 0)):
                    r0, r1 = i + di, i + dj
                    if r0 < h0 and r1 < h1:
                        a, b = columns[j][r0], columns[j + 1][r1]
                        if a != b:
                            yield [a, b]

    # --------------------------- LLM interaction -------------------------------

    @staticmethod
    def _borda_aggregate(rankings: Sequence[Sequence[str]], batch: Sequence[str]) -> List[str]:
        """Aggregate listwise votes by Borda count (earlier position ⇒ more points)."""
        scores: Dict[str, float] = {x: 0.0 for x in batch}
        n = len(batch)
        for ranking in rankings:
            pos = {item: i for i, item in enumerate(ranking)}
            for item in batch:
                scores[item] += float(n - 1 - pos.get(item, n - 1))
        return sorted(batch, key=lambda x: (-scores[x], x))

    def _structured_listwise(self, prompt: str, presentation: Sequence[str]) -> List[str]:
        """OrderByLLM external_comparisons call: structured {Reasoning_steps, sorted_list}
        with 1-based ids; retried until the ids form a permutation of the batch.

        structured_schema="recall" replaces the free-text Reasoning_steps with a
        {id, height_m} entry per key (the model's own recalled value) before sorted_list;
        the prompt text is unchanged and only sorted_list is used."""
        from pydantic import BaseModel

        if self.structured_schema == "recall":
            class RecalledHeight(BaseModel):
                id: int
                height_m: float

            class ExternalComparisonReasoning(BaseModel):
                heights: List[RecalledHeight]
                sorted_list: List[int]
        else:
            class ExternalComparisonReasoning(BaseModel):
                Reasoning_steps: str
                sorted_list: List[int]

        n = len(presentation)
        best: List[str] = []
        for attempt in range(1, 6):
            suffix = (
                ""
                if attempt == 1
                else f"\nRespond in JSON. The sorted_list must contain each id from 1 to {n} exactly once.\n"
            )
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ZHAO_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + suffix},
                    ],
                    temperature=0.0,
                    response_format=ExternalComparisonReasoning,
                    max_completion_tokens=12 * 1024,
                )
            except Exception as e:  # noqa: BLE001
                err = str(e).lower()
                if self.verbose:
                    print(f"    structured listwise attempt {attempt} failed: {e}")
                time.sleep(10 if ("429" in err or "rate limit" in err) else 2)
                continue
            self.stats.update(list(presentation), response)
            parsed = response.choices[0].message.parsed
            ids = list(parsed.sorted_list) if parsed is not None else []
            ordered: List[str] = []
            for i in ids:
                if isinstance(i, int) and 1 <= i <= n and presentation[i - 1] not in ordered:
                    ordered.append(presentation[i - 1])
            if len(ordered) == n:
                return ordered
            if len(ordered) > len(best):
                best = ordered
        return best + [x for x in presentation if x not in best]

    def _sort_with_llm_once(
        self,
        presentation: Sequence[str],
        round_id: int,
        payload: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        """Single listwise LLM call on a fixed presentation order."""
        prompt = self._build_prompt(presentation, round_id, payload or {})
        if (
            self.structured_output
            and self.sort_by == "height"
            and self.height_prompt_mode == "knowledge"
        ):
            return self._structured_listwise(prompt, presentation)

        if self.sort_by == "height":
            use_knowledge = self.height_prompt_mode in ("knowledge", "knowledge_rescued")
            use_json_format = use_knowledge or "deepseek" in self.model.lower()

            if use_knowledge:
                system_message = (
                    "You are a helpful assistant. Respond with valid JSON only."
                )
            elif use_json_format:
                system_message = (
                    "You are a precise sorting oracle. Sort NBA players by their height in meters (ascending order, shortest to tallest). "
                    "When players have the same height, break ties alphabetically by player name. "
                    "Return a JSON list of integer IDs only, no additional text."
                )
            else:
                system_message = (
                    "You are a precise sorting oracle. Sort NBA players by their height from shortest to tallest. "
                    "Compare the height values (decimal numbers in meters) and order from smallest to largest. "
                    "When players have the same height, sort them alphabetically by name. "
                    "Return only the player names separated by commas, with no heights or additional text."
                )
        else:
            system_message = "You are a precise sorting oracle. Return the items in strict ascending alphabetical order (A to Z) separated by commas."

        last_error = None
        create_kw: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        }
        if _supports_temperature_zero(self.model):
            create_kw["temperature"] = 0
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**create_kw)
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if attempt < 2 and (
                    "503" in err_str
                    or "500" in err_str
                    or "overloaded" in err_str
                    or "unavailable" in err_str
                ):
                    if self.verbose:
                        print(f"    Retry {attempt + 1}/2 in 5s after: {e}")
                    time.sleep(5)
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
        ordered = self._parse_response(
            response.choices[0].message.content, presentation  # type: ignore[attr-defined]
        )
        self.stats.update(list(presentation), response)
        return ordered

    def _sort_with_llm(
        self, batch: Sequence[str], round_id: int, payload: Optional[Dict[str, object]] = None
    ) -> List[str]:
        """
        Ask the LLM to sort `batch`.  Returns the items ordered from least to greatest
        (shortest to tallest for height, A to Z for alphabetical).

        If listwise_votes > 1: ask with multiple presentation shuffles and
        aggregate by Borda (self-consistency / position-bias reduction).
        """
        from .partition import apply_prompt_order

        n_votes = getattr(self, "listwise_votes", 1) or 1
        if n_votes <= 1:
            presentation = apply_prompt_order(batch, self.prompt_order, self.partition_seed)
            return self._sort_with_llm_once(presentation, round_id, payload)

        rankings: List[List[str]] = []
        base = list(batch)
        for v in range(n_votes):
            if v == 0:
                presentation = apply_prompt_order(
                    base, self.prompt_order, self.partition_seed
                )
            else:
                rng = random.Random(self.partition_seed + 17 * round_id + 101 * v)
                presentation = base[:]
                rng.shuffle(presentation)
            rankings.append(self._sort_with_llm_once(presentation, round_id, payload))
        ordered = self._borda_aggregate(rankings, base)
        if self.verbose:
            n_unique = len({tuple(r) for r in rankings})
            print(
                f"    listwise votes={n_votes} unique_orders={n_unique} "
                f"→ {' > '.join(ordered)}"
            )
        return ordered

    def _build_prompt(self, batch: Sequence[str], round_id: int, payload: Dict[str, object]) -> str:
        details = ", ".join(f"{k}={v}" for k, v in payload.items())
        
        if self.sort_by == "height":
            use_knowledge = self.height_prompt_mode in ("knowledge", "knowledge_rescued")
            use_json_format = use_knowledge or "deepseek" in self.model.lower()

            if use_knowledge:
                keys_text = _zhao_numbered_nba_keys(batch)
                template = (
                    NBA_KNOWLEDGE_COMPARISON_PROMPT_RESCUED
                    if self.height_prompt_mode == "knowledge_rescued"
                    else NBA_KNOWLEDGE_COMPARISON_PROMPT
                )
                prompt = template.format(keys=keys_text)
            elif use_json_format:
                # Explicit heights, numbered ids, JSON id list back (DeepSeek models).
                batch_with_ids = []
                for i, name in enumerate(batch):
                    height = self.height_map[name]
                    batch_with_ids.append(f"{i}: {name} ({height:.2f}m)")
                
                keys_text = "\n".join(batch_with_ids)
                
                prompt = (
                    f"Given a list of NBA players with id at the front:\n\n"
                    f"{keys_text}\n\n"
                    f"Based on the provided height values (in meters, rounded to two decimal places), sort the players in ascending order based on player height.\n\n"
                    f"Break ties based on the player name in alphabetical order.\n\n"
                    f"Output a JSON list of integer ids, representing the players from shortest to tallest."
                )
            else:
                # Explicit heights next to names, comma-separated names back (OpenAI models).
                batch_with_heights = [f"{name} ({self.height_map[name]:.2f})" for name in batch]
                
                prompt = (
                    f"Sort these NBA players by their height from shortest to tallest.\n\n"
                    f"Players and heights:\n"
                    f"{', '.join(batch_with_heights)}\n\n"
                    f"Instructions:\n"
                    f"- Compare the height values (the decimal numbers)\n"
                    f"- Lower numbers = shorter players (come first)\n"
                    f"- Higher numbers = taller players (come last)\n"
                    f"- Sort from smallest height to largest height\n"
                    f"- If two players have the same height, sort them alphabetically by name\n\n"
                    f"Return ONLY the player names in order (shortest first), separated by commas.\n"
                    f"No heights, no explanations, just the names."
                )
        else:
            # sort_by='alphabetical': a value-free sanity task.
            prompt = (
                f"Round {round_id} request ({details}). "
                f"Sort these items in alphabetical order (A to Z): "
                f"{', '.join(batch)}.\n"
                "Respond with the items in ascending alphabetical order (A comes before Z) separated by commas."
            )
        
        return prompt

    def _parse_response(self, text: str, fallback_batch: Sequence[str]) -> List[str]:
        # JSON id list first, then names.
        if self.sort_by == "height":
            id_base = 1 if self.height_prompt_mode in ("knowledge", "knowledge_rescued") else 0
            json_match = re.search(r'\[[\s\d,]*\]', text)
            if json_match:
                try:
                    id_list = json.loads(json_match.group())
                    ordered = []
                    for id_val in id_list:
                        if not isinstance(id_val, int):
                            continue
                        idx = id_val - id_base
                        if 0 <= idx < len(fallback_batch):
                            ordered.append(fallback_batch[idx])
                    if len(ordered) == len(fallback_batch):
                        return ordered
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
        
        candidates = [token.strip() for token in text.replace("\n", ",").split(",")]
        cleaned = [item for item in candidates if item]
        # Preserve original casing / names by matching against batch items
        batch_set = {item.lower(): item for item in fallback_batch}
        ordered: List[str] = []
        for token in cleaned:
            key = token.lower()
            if key in batch_set and batch_set[key] not in ordered:
                ordered.append(batch_set[key])

        missing = [item for item in fallback_batch if item not in ordered]
        return ordered + missing

    def _validate_batch_output(
        self, llm_output: List[str], original_batch: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate LLM output against original batch.
        Detects hallucinations: missing items, extra items, duplicates, parsing issues.
        
        Returns:
            (is_valid, report)
            report contains: violations, missing_items, extra_items, duplicates, etc.
        """
        if self.verbose:
            print(f"    Validation Checks:")
        
        report: Dict[str, Any] = {
            "is_valid": True,
            "violations": [],
            "missing_items": [],
            "extra_items": [],
            "duplicates": [],
            "parsing_issues": [],
        }
        
        # Check 1: Parsing success - ensure we got items
        if self.verbose:
            print(f"    Check 1: Parsing success", end="... ")
        
        if not llm_output or len(llm_output) == 0:
            if self.verbose:
                print(f"FAILED (No items extracted)")
            report["is_valid"] = False
            report["violations"].append("No items extracted from LLM response")
            report["parsing_issues"].append("Empty output")
        else:
            if self.verbose:
                print(f"PASSED ({len(llm_output)} items extracted)")
        
        # Check 2: Minimum length
        if self.verbose:
            print(f"    Check 2: Minimum length", end="... ")
        
        if len(llm_output) < 2:
            if self.verbose:
                print(f"FAILED (only {len(llm_output)} item(s), need at least 2)")
            report["is_valid"] = False
            report["violations"].append(f"Output has only {len(llm_output)} item(s), expected at least 2")
            report["parsing_issues"].append("Insufficient items")
        else:
            if self.verbose:
                print(f"PASSED ({len(llm_output)} items, minimum is 2)")
        
        # Check 3: Completeness - all original items present
        if self.verbose:
            print(f"    Check 3: Completeness (all original items present)", end="... ")
        
        original_set = set(original_batch)
        output_set = set(llm_output)
        missing = original_set - output_set
        
        if missing:
            if self.verbose:
                print(f"FAILED (missing {len(missing)} item(s): {list(missing)})")
            report["is_valid"] = False
            report["violations"].append(f"Missing {len(missing)} item(s) from original batch")
            report["missing_items"] = list(missing)
        else:
            if self.verbose:
                print(f"PASSED (all {len(original_batch)} items present)")
        
        # Check 4: No extra items - only items from original batch
        if self.verbose:
            print(f"    Check 4: No extra items (only items from original batch)", end="... ")
        
        extra = output_set - original_set
        
        if extra:
            if self.verbose:
                print(f"FAILED (found {len(extra)} extra item(s): {list(extra)})")
            report["is_valid"] = False
            report["violations"].append(f"Found {len(extra)} extra item(s) not in original batch")
            report["extra_items"] = list(extra)
        else:
            if self.verbose:
                print(f"PASSED (no extra items)")
        
        # Check 5: No duplicates
        if self.verbose:
            print(f"    Check 5: No duplicates (each item appears once)", end="... ")
        
        seen = set()
        duplicates = []
        for item in llm_output:
            if item in seen:
                duplicates.append(item)
            else:
                seen.add(item)
        
        if duplicates:
            if self.verbose:
                print(f"FAILED (found {len(duplicates)} duplicate(s): {duplicates})")
            report["is_valid"] = False
            report["violations"].append(f"Found {len(duplicates)} duplicate(s)")
            report["duplicates"] = duplicates
        else:
            if self.verbose:
                print(f"PASSED (no duplicates, {len(llm_output)} unique items)")
        
        # Check 6: Item name matching - verify all items match original (case-insensitive)
        if self.verbose:
            print(f"    Check 6: Item name matching (case-insensitive)", end="... ")
        
        original_lower_map = {item.lower(): item for item in original_batch}
        invalid_items = []
        
        for item in llm_output:
            if item.lower() not in original_lower_map:
                invalid_items.append(item)
        
        if invalid_items:
            if self.verbose:
                print(f"FAILED (found {len(invalid_items)} invalid item(s): {invalid_items})")
            report["is_valid"] = False
            report["violations"].append(f"Found {len(invalid_items)} item(s) that don't match original")
            report["extra_items"].extend(invalid_items)
        else:
            if self.verbose:
                print(f"PASSED (all items match original batch)")
        
        if self.verbose:
            print(f"    Validation Result: ", end="")
        
        return report["is_valid"], report

    def _correct_batch_output(
        self, llm_output: List[str], original_batch: List[str], report: Dict[str, Any]
    ) -> List[str]:
        """
        Correct batch output for all 6 validation checks:
        1. Empty output → Use original batch as fallback
        2. Minimum length → Ensure at least 2 items
        3. Missing items → Add missing items
        4. Extra items → Remove extra items
        5. Duplicates → Remove duplicates
        6. Invalid items → Remove invalid items
        
        Preserves LLM ordering for items that are valid.
        """
        # Correction 1: Handle empty output (Check 1)
        if not llm_output or len(llm_output) == 0:
            if self.verbose:
                print(f"      Correction 1: Empty output -> Using original batch as fallback")
            # Use original batch as fallback (no LLM ordering, but we have items)
            return original_batch.copy()
        
        # Start with valid items from LLM output
        original_set = set(original_batch)
        original_lower_map = {item.lower(): item for item in original_batch}
        
        corrected = []
        seen = set()
        removed_duplicates = []
        removed_extras = []
        
        # Correction 5, 4, 6: Keep valid items in LLM order (removes duplicates, extras, invalid items)
        for item in llm_output:
            item_key = item.lower()
            # Check if item is valid (matches original, not duplicate)
            if item_key in original_lower_map:
                original_item = original_lower_map[item_key]
                if original_item not in seen:
                    corrected.append(original_item)
                    seen.add(original_item)
                else:
                    # Duplicate found (Correction 5)
                    removed_duplicates.append(original_item)
            else:
                # Invalid/extra item (Correction 4 & 6)
                removed_extras.append(item)
        
        # Report corrections applied
        if removed_duplicates and self.verbose:
            print(f"      Correction 5: Removed {len(removed_duplicates)} duplicate(s): {removed_duplicates}")
        if removed_extras and self.verbose:
            print(f"      Correction 4 & 6: Removed {len(removed_extras)} extra/invalid item(s): {removed_extras}")
        
        # Correction 3: Add missing items at the end (preserving original order as fallback)
        missing_items = [item for item in original_batch if item not in corrected]
        if missing_items and self.verbose:
            print(f"      Correction 3: Added {len(missing_items)} missing item(s): {missing_items}")
        for item in original_batch:
            if item not in corrected:
                corrected.append(item)
        
        # Correction 2: Ensure minimum length (at least 2 items for sorting)
        if len(corrected) < 2:
            if self.verbose:
                print(f"      Correction 2: Only {len(corrected)} item(s), ensuring minimum length")
            if len(corrected) == 0:
                # Fallback: use original batch
                if self.verbose:
                    print(f"      -> Fallback: Using original batch")
                return original_batch.copy()
            elif len(corrected) == 1 and len(original_batch) >= 2:
                # Add remaining items from original batch
                remaining = [item for item in original_batch if item not in corrected]
                if self.verbose:
                    print(f"      -> Adding remaining {len(remaining)} item(s) to reach minimum length")
                for item in original_batch:
                    if item not in corrected:
                        corrected.append(item)
        
        if len(corrected) >= 2 and len(original_batch) == len(corrected) and set(corrected) == original_set:
            if self.verbose and len(llm_output) < 2:
                print(f"      Correction 2: Ensured minimum length ({len(corrected)} items)")
        
        return corrected


# Models that support temperature=0 (GPT-5.x only support default 1 per API)
MODELS_SUPPORTING_TEMPERATURE_ZERO = frozenset({"gpt-4o", "gpt-4o-mini"})


def _supports_temperature_zero(model: str) -> bool:
    """Return True if this model accepts temperature=0 (otherwise API may return 400)."""
    return (model or "").strip().lower() in MODELS_SUPPORTING_TEMPERATURE_ZERO

