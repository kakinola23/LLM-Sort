"""
Round-3 loops for graph_backend="dag" (DagTransitiveOrder), for million-item runs.

BatchedLLMSorter's Round-3 loops call known_pairs() after every batch for the
theta stop rule. On a DAG backend that is an exact but O(E * w / block) pass,
too slow to run ~13k times. These overrides keep the same decisions:

* Stall rule: a batch adds no new pairs iff it inserted no new edge, which
  add_total_order reports directly, so it is still checked after every batch.
* Theta rule: checked with exact counts on an adaptive schedule. When a check
  finds rho >= theta, a binary search over per-batch edge-count snapshots finds
  the first batch where the original loop would have stopped; the graph is
  truncated back to that batch. LLM calls made past it ("overshoot") were paid
  for but are not part of the algorithm's run; they are reported separately
  in self.round3_overshoot.
* Greedy unresolved packing: degrees come from exact per-item counts, which
  also give the exact known-pair count after each batch, so no lazy rule is
  needed there.

For any other backend the inherited implementations are used unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor

from .sorter import Round3StopStats
from .reach_index import DagTransitiveOrder


class ScalableRound3Mixin:
    theta_check_initial_gap = 25
    theta_check_max_gap = 150

    def _uses_dag(self) -> bool:
        return isinstance(self.graph, DagTransitiveOrder)

    def _on_round3_truncate(self, snapshot: Dict[str, Any]) -> None:
        """Hook for subclasses (e.g. to drop progress-log entries past the stop)."""

    # --------------------------------------------------------------------- #

    def _build_diagonal_schedule(self, columns: List[List[str]]) -> List[List[str]]:
        """Identical batching to BatchedLLMSorter._round_three_diagonals."""
        num_cols = len(columns)
        max_height = max(len(col) for col in columns)
        all_diags = list(self._diagonals(columns, max_height, num_cols))
        batched_diags: List[List[str]] = []
        current_batch: List[str] = []
        current_batch_size = 0
        pack_small = not self.round3_dense

        for diag in all_diags:
            if len(diag) < 2:
                continue
            if self.use_transitivity and self.graph.all_pairs_known(diag):
                continue

            if pack_small and len(diag) <= 3 and current_batch_size + len(diag) <= self.batch_size:
                current_batch.extend(diag)
                current_batch_size = len(set(current_batch))
            elif self.round3_dense and len(diag) == 2:
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
        return batched_diags

    def _graph_snapshot(self, **extra: Any) -> Dict[str, Any]:
        g = self.graph
        snap = {
            "edges": g.num_edges,
            "direct": g.direct_relations,
            "rejected": g.rejected_contradictions_count,
            "pair_flips": g.rejected_pair_flips_count,
            "cycles": g.rejected_transitivity_cycles_count,
            "calls": self.stats.count,
            "prompt_tokens": self.stats.prompt_tokens,
            "completion_tokens": self.stats.completion_tokens,
        }
        snap.update(extra)
        return snap

    def _restore_snapshot(self, snap: Dict[str, Any], batches: int) -> None:
        g = self.graph
        overshoot = {
            "calls": self.stats.count - snap["calls"],
            "prompt_tokens": self.stats.prompt_tokens - snap["prompt_tokens"],
            "completion_tokens": self.stats.completion_tokens - snap["completion_tokens"],
            "batches": batches,
        }
        g.truncate_edges(snap["edges"])
        g.direct_relations = snap["direct"]
        g.rejected_contradictions_count = snap["rejected"]
        g.rejected_pair_flips_count = snap["pair_flips"]
        g.rejected_transitivity_cycles_count = snap["cycles"]
        self.round3_overshoot = overshoot
        self._on_round3_truncate(snap)

    def _theta_reached(self, known: int, total_pairs: int) -> bool:
        ratio = known / total_pairs if total_pairs > 0 else 0
        return ratio >= self.completion_threshold

    def _first_theta_batch(self, snaps: List[Dict[str, Any]], lo: int, hi: int, total_pairs: int) -> int:
        """Smallest k in [lo, hi] whose prefix graph reaches theta (snaps[hi] does)."""
        g = self.graph
        while lo < hi:
            mid = (lo + hi) // 2
            if self._theta_reached(g.known_pairs(e_limit=snaps[mid]["edges"]), total_pairs):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # --------------------------------------------------------------------- #

    def _round_three_diagonals(self, columns: List[List[str]]) -> None:
        if not self._uses_dag():
            return super()._round_three_diagonals(columns)  # type: ignore[misc]
        g = self.graph
        self.round3_overshoot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "batches": 0}
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"ROUND 3: Sort diagonals to stitch partial orders ({'dense' if self.round3_dense else 'standard'})")
            print("=" * 70)

        batched_diags = self._build_diagonal_schedule(columns)
        present = getattr(self, "_presentation_for", list)
        self._r3_diag_keys = {tuple(present(b)) for b in batched_diags}
        n = len(g.items)
        total_pairs = n * (n - 1) // 2

        diag_idx = 0
        round3_calls_start = self.stats.count
        skipped_pre_llm = 0
        consecutive_no_progress = 0
        stop_reason = "exhausted_schedule"
        max_no_progress = self.max_no_progress
        use_theta = self.enable_completion_early_stop

        snaps: List[Dict[str, Any]] = []  # one per executed batch
        checked_below = -1  # last snapshot index known exactly to be below theta
        exact_log = [(-1, g.known_pairs())]  # (snapshot index, known pairs) below theta
        next_check = self.theta_check_initial_gap - 1

        def settle_theta(upto: int) -> Optional[int]:
            """Exactly resolve theta over snapshots (checked_below, upto]. Returns
            the first crossing index, or None if still below at `upto`."""
            nonlocal checked_below
            if upto <= checked_below:
                return None
            known = g.known_pairs(e_limit=snaps[upto]["edges"])
            if not self._theta_reached(known, total_pairs):
                checked_below = upto
                exact_log.append((upto, known))
                return None
            return self._first_theta_batch(snaps, checked_below + 1, upto, total_pairs)

        def stop_at_theta(first: int, last: int) -> None:
            nonlocal diag_idx, skipped_pre_llm
            snap = snaps[first]
            if first < last:
                self._restore_snapshot(snap, last - first)
            diag_idx = snap["diag_idx"]
            skipped_pre_llm = snap["skipped"]

        # Optional prefetching (needs the _r3_fetch/_r3_consume hooks): LLM calls for
        # up to `r3_prefetch` upcoming batches run in worker threads while the main
        # thread applies answers strictly in schedule order. A batch whose pairs are
        # all known when it is reached is skipped exactly as before and its fetched
        # answer is discarded (relations only accumulate, so a batch known at
        # submission time is still known at its turn and is never submitted).
        # Workers never touch the graph, so decisions match the sequential loop.
        prefetch = int(getattr(self, "r3_prefetch", 0) or 0) if hasattr(self, "_r3_fetch") else 0
        pool = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 0 else None
        futures: Dict[int, Any] = {}
        decided = 0
        consumed = 0

        def payload_of(b):
            return {"diag": True, "batched": len(b)}

        def top_up(i: int) -> None:
            nonlocal decided
            decided = max(decided, i)
            limit = min(len(batched_diags), i + prefetch)
            while decided < limit:
                j = decided
                decided += 1
                b = batched_diags[j]
                if self.use_transitivity and g.all_pairs_known(b):
                    continue
                futures[j] = pool.submit(self._r3_fetch, b, payload_of(b))

        stall_at: Optional[int] = None
        try:
            for i, batch_diag in enumerate(batched_diags):
                if g.is_complete():
                    stop_reason = "complete"
                    break

                if self.use_transitivity and g.all_pairs_known(batch_diag):
                    skipped_pre_llm += 1
                    continue

                if self.verbose:
                    print(f"  Batch {diag_idx}: Sorting {batch_diag} ({len(batch_diag)} items)")
                if pool is not None:
                    top_up(i)
                    fut = futures.pop(i, None)
                    if fut is None:
                        fut = pool.submit(self._r3_fetch, batch_diag, payload_of(batch_diag))
                    pres, ordered = fut.result()
                    self._r3_consume(pres)
                    consumed += 1
                else:
                    ordered = self._sort_with_llm(batch_diag, round_id=3, payload=payload_of(batch_diag))
                added = g.add_total_order(ordered, direct=True)
                consecutive_no_progress = 0 if added else consecutive_no_progress + 1
                diag_idx += 1
                snaps.append(self._graph_snapshot(diag_idx=diag_idx, skipped=skipped_pre_llm))
                cur = len(snaps) - 1

                if (
                    self.enable_stall_early_stop
                    and max_no_progress > 0
                    and consecutive_no_progress >= max_no_progress
                ):
                    stall_at = cur
                    break

                if use_theta and cur >= next_check:
                    first = settle_theta(cur)
                    if first is not None:
                        stop_reason = "theta"
                        stop_at_theta(first, cur)
                        break
                    # Schedule the next exact check from the observed growth rate,
                    # aiming at half the predicted distance to theta.
                    (idx_prev, k_prev), (idx_now, k_now) = exact_log[-2], exact_log[-1]
                    rate = (k_now - k_prev) / max(1, idx_now - idx_prev)
                    remaining = self.completion_threshold * total_pairs - k_now
                    gap = self.theta_check_max_gap
                    if rate > 0:
                        gap = int(max(1, min(gap, 0.5 * remaining / rate)))
                    next_check = cur + gap
                    if self.verbose:
                        print(f"  [theta-check] batch {cur}: rho={k_now / total_pairs:.4f}, next check in {gap}")
        finally:
            if pool is not None:
                unused = len(futures)
                for f in futures.values():
                    f.cancel()
                pool.shutdown(wait=True)
                self.r3_prefetch_stats = {"window": prefetch, "consumed": consumed, "submitted_unused": unused}

        if use_theta and stop_reason != "theta" and snaps:
            # Theta may have been crossed between the last exact check and the
            # stop (stall / complete / end of schedule): resolve it exactly.
            # The original loop tests stall before theta within a batch.
            end = len(snaps) - 1
            first = settle_theta(end)
            if first is not None and first != stall_at:
                stop_reason = "theta"
                stop_at_theta(first, end)
                stall_at = None
        if stall_at is not None:
            stop_reason = "stall"

        known_pairs, total_pairs, rho = self.known_pair_counts()
        if g.is_complete():
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
            round3_llm_calls=self.stats.count - round3_calls_start - self.round3_overshoot["calls"],
            round3_batches_planned=len(batched_diags),
            round3_batches_executed=diag_idx,
            round3_batches_skipped_known=skipped_pre_llm,
        )
        print(
            f"  Round 3 diagonals done: stop={stop_reason}, rho={rho:.4f}, "
            f"batches executed={diag_idx}/{len(batched_diags)}, skipped as known={skipped_pre_llm}",
            flush=True,
        )

    def _round_three_unresolved_greedy(self) -> None:
        if not self._uses_dag():
            return super()._round_three_unresolved_greedy()  # type: ignore[misc]
        g = self.graph
        n = len(g.items)
        total0 = n * (n - 1) // 2
        desc, anc = g.desc_anc_counts()
        known0 = int(desc.sum())
        if self.verbose:
            print("\n" + "=" * 70)
            print("ROUND 3: unresolved-greedy packing (degree-prioritized coverage)")
            print("=" * 70)
            print(f"  After R1-R2 closure: rho={known0 / total0:.4f} known={known0}/{total0}")

        round3_calls_start = self.stats.count
        consecutive_no_progress = 0
        stop_reason = "complete"
        step = 0
        max_no_progress = self.max_no_progress
        max_batches = max(1, (total0 - known0) // max(1, self.batch_size - 1) + 50)

        # Resumed runs: greedy batches already answered are re-applied in their
        # recorded order instead of being re-selected, which would need a full
        # O(n * chains) recount per batch just to pick the same batch again.
        # The run that recorded them did not stop after any of them, so the
        # stop rules only need checking after the last one.
        replay = list(self._greedy_replay_batches()) if hasattr(self, "_greedy_replay_batches") else []
        n_replayed = len(replay)
        if replay:
            print(f"  Greedy packing: re-applying {len(replay)} recorded batches", flush=True)

        # Parallel pivots: after the replay, pick `greedy_parallel` key-disjoint
        # batches per degree count, fetch them concurrently and apply them in
        # selection order. 1 keeps the one-batch-per-count schedule.
        parallel = max(1, int(getattr(self, "greedy_parallel", 1) or 1))
        fetch = hasattr(self, "_r3_fetch")
        pool = ThreadPoolExecutor(max_workers=parallel) if parallel > 1 and fetch else None
        par_started = False
        stop_exc = getattr(self, "greedy_stop_exceptions", ())

        while not g.is_complete():
            if step >= max_batches:
                stop_reason = "max_batches"
                break
            if not replay and parallel > 1:
                if not par_started:
                    par_started = True
                    if hasattr(self, "_greedy_log_start"):
                        self._greedy_log_start(n_replayed)
                    print(f"  Greedy packing: {parallel} parallel batches per count from batch {step + 1}",
                          flush=True)
                deg = (n - 1) - desc - anc
                batches = g.greedy_batches(deg, self.batch_size, parallel)
                if not batches:
                    stop_reason = "no_batch"
                    break
                payloads = [{"unresolved_greedy": True, "step": step + i} for i in range(len(batches))]
                futures = [pool.submit(self._r3_fetch, b, p) for b, p in zip(batches, payloads)] if pool else None
                first = step + 1
                # A round is applied whole or not at all, so a run resumed after
                # a halt re-selects the same round (its answers are cached).
                if futures is not None:
                    try:
                        answers = [f.result() for f in futures]
                    except BaseException as e:
                        for f in futures:
                            f.cancel()
                        if isinstance(e, stop_exc):
                            print(f"  Greedy packing stopped before batch {first}: {e}", flush=True)
                            stop_reason = "stopped_no_cached_answer"
                            break
                        raise
                else:
                    answers = []
                    try:
                        for b, p in zip(batches, payloads):
                            answers.append((None, self._sort_with_llm(b, round_id=3, payload=p)))
                    except stop_exc as e:
                        print(f"  Greedy packing stopped before batch {first}: {e}", flush=True)
                        stop_reason = "stopped_no_cached_answer"
                        break
                round_added = 0
                for pres, ordered in answers:
                    added = g.add_total_order(ordered, direct=True)
                    if pres is not None:
                        self._r3_consume(pres)
                        if hasattr(self, "_greedy_log_applied"):
                            self._greedy_log_applied(pres)
                    consecutive_no_progress = 0 if added else consecutive_no_progress + 1
                    round_added += added
                    step += 1
                desc, anc = g.desc_anc_counts()
                after_pairs = int(desc.sum())
                completion_ratio = after_pairs / total0 if total0 > 0 else 0
                print(f"  Greedy batches {first}-{step}: {round_added} new direct edges, "
                      f"rho={completion_ratio:.4f}, calls_total={self.stats.count}", flush=True)
                if (
                    self.enable_stall_early_stop
                    and max_no_progress > 0
                    and consecutive_no_progress >= max_no_progress
                ):
                    stop_reason = "stall"
                    break
                if (
                    self.enable_completion_early_stop
                    and completion_ratio >= self.completion_threshold
                    and not g.is_complete()
                ):
                    stop_reason = "theta"
                    break
                continue
            replaying = bool(replay)
            if replaying:
                batch = replay.pop(0)
            else:
                deg = (n - 1) - desc - anc
                batch = g.greedy_batch(deg, self.batch_size)
            if len(batch) < 2:
                stop_reason = "no_batch"
                break
            if self.verbose:
                print(f"\n  Greedy batch {step}: Sorting {len(batch)} items")
            try:
                ordered = self._sort_with_llm(batch, round_id=3, payload={"unresolved_greedy": True, "step": step})
            except getattr(self, "greedy_stop_exceptions", ()) as e:
                # Offline replay: stop at the first greedy batch with no cached
                # answer and report the state reached so far.
                print(f"  Greedy packing stopped before batch {step + 1}: {e}", flush=True)
                stop_reason = "stopped_no_cached_answer"
                break
            added = g.add_total_order(ordered, direct=True)
            consecutive_no_progress = 0 if added else consecutive_no_progress + 1
            step += 1
            if replaying and replay:
                if step % 100 == 0:
                    print(f"  Greedy batch {step}: re-applied from cache", flush=True)
                continue
            desc, anc = g.desc_anc_counts()
            after_pairs = int(desc.sum())
            completion_ratio = after_pairs / total0 if total0 > 0 else 0
            print(f"  Greedy batch {step}: {added} new direct edges, rho={completion_ratio:.4f}, "
                  f"calls_total={self.stats.count}", flush=True)

            if (
                self.enable_stall_early_stop
                and max_no_progress > 0
                and consecutive_no_progress >= max_no_progress
            ):
                stop_reason = "stall"
                break
            if (
                self.enable_completion_early_stop
                and completion_ratio >= self.completion_threshold
                and not g.is_complete()
            ):
                stop_reason = "theta"
                break

        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        known_pairs, total_pairs, rho = self.known_pair_counts()
        if g.is_complete():
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
