#!/usr/bin/env python3
"""Resumable, parallelized AmazonPriceSorter for the 1M run.

- Every LLM call's (presentation -> order, tokens) result is appended to an
  on-disk JSONL cache immediately after it's received. On restart, the cache
  is replayed (zero API cost, zero wait) for every batch already done, and
  only genuinely new batches hit the live API - so a crash/interruption never
  wastes spend and never needs a manual "where did I stop" step.
- Round 1 batches are independent of each other (no shared graph state yet),
  so they're dispatched concurrently via a thread pool - this is the biggest
  lever on wall-clock time since R1 is a large fraction of total calls.
- Periodic tau checkpoints during Round 3 (gated to round_id==3 so we skip
  the uninformative pre-stitching R1/R2 noise): every CHECK_INTERVAL live
  calls, materialize the graph's current best-guess total order and compute
  tau against ground truth. If tau stays below TAU_STOP_THRESHOLD for
  TAU_STOP_CONSECUTIVE consecutive checks, raise TauTooLowError to halt the
  run cleanly (cache is already safe on disk; rerunning resumes normally).
"""
import json
import os
import random
import re
import zlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loaders.amazon import AmazonPriceSorter
from .sorter import _supports_temperature_zero
from .transitive import ChainTransitiveOrder
from .reach_index import DagTransitiveOrder, fast_kendalls_tau
from .scalable_round3 import ScalableRound3Mixin
from scipy.stats import kendalltau


class TauTooLowError(RuntimeError):
    """Raised to halt the run when tau looks like it's not going to recover."""


class StopAfterRound1(RuntimeError):
    """Raised after Round 1 when the run was asked to stop there (cache kept)."""


class LiveCallBudgetError(RuntimeError):
    """Raised before a live API call that would exceed the per-process budget."""


class IncompleteAnswerError(RuntimeError):
    """Raised when the LLM keeps omitting items after every retry (strict mode
    never drops items or fills them in, so the run halts instead)."""


def shuffled_presentation(batch: Sequence[str], seed: int) -> List[str]:
    """Deterministic per-batch shuffle of the order items are shown in, so
    position in the prompt carries no information (Rounds 2-3 batches are
    otherwise listed in grid order, e.g. diagonals cheapest-row first). Keyed
    on the batch's contents with crc32, so it is identical across processes
    (unlike hash()) and resumed runs hit the cache."""
    out = sorted(batch)  # start from a canonical order so grid order cannot leak through
    rng = random.Random(zlib.crc32(("|".join(out) + f"#{seed}").encode()))
    rng.shuffle(out)
    return out


def make_resumable_sorter_class(
    *,
    cache_path: Path,
    true_order_fn,
    price_fn=None,
    r1_workers: int = 12,
    tau_check_interval: int = 2000,
    tau_stop_threshold: float = 0.5,
    tau_stop_consecutive: int = 3,
    progress_path: Optional[Path] = None,
    max_live_calls: Optional[int] = None,
    prompt_key: Optional[str] = None,
    strict_complete: bool = False,
    max_attempts: int = 5,
    stop_after_round1: bool = False,
    r2_workers: int = 1,
    shuffle_seed: Optional[int] = None,
):
    """Build a ResumableAmazonSorter class bound to one run's cache/progress
    files. A factory (not a plain class) because the cache/progress paths and
    the tau-check callback need to be bound per-run without relying on
    fragile shared class state across different scripts/configs."""
    # Order in which parallel greedy batches were applied (see _greedy_replay_batches).
    greedy_log_path = cache_path.with_name("greedy_log.jsonl")

    class ResumableAmazonSorter(ScalableRound3Mixin, AmazonPriceSorter):
        _cache: Dict[Tuple[str, ...], List[str]] = {}
        _cache_lock = threading.Lock()
        _stats_lock = threading.Lock()
        _progress_lock = threading.Lock()
        _cache_file_handle = None
        _live_calls = 0
        # Presentation keys of every call this run consumed (cached or live),
        # in order, so the final aggregation uses exactly the run's evidence.
        _used_keys: List[Tuple[str, ...]] = []
        # Serializes every graph read/write when Round-2 merges run in threads
        # (the numba graph kernels are not thread-safe); LLM calls run outside it.
        _graph_lock = threading.RLock()
        _live_calls_since_check = 0
        _consecutive_low_tau = 0
        _tau_log: List[Dict[str, Any]] = []
        _round3_seen = False
        _price_estimates: Dict[str, float] = {}
        _timing: Dict[str, float] = {}

        @classmethod
        def load_cache(cls):
            cls._cache = {}
            cls._used_keys = []
            cls._price_estimates = {}
            total_count = total_prompt = total_completion = 0
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = tuple(rec["presentation"])
                        total_count += 1
                        total_prompt += rec.get("prompt_tokens", 0)
                        total_completion += rec.get("completion_tokens", 0)
                        if strict_complete and rec.get("n_missing", 0) > 0:
                            continue  # paid for, but incomplete: ask again
                        cls._cache[key] = rec["order"]
                        cls._price_estimates.update(rec.get("prices", {}))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cls._cache_file_handle = cache_path.open("a", encoding="utf-8")
            return total_count, total_prompt, total_completion

        def _write_progress(self, extra: Optional[Dict[str, Any]] = None):
            if progress_path is None:
                return
            with self.__class__._progress_lock:
                payload = {
                    "calls_total": self.stats.count,
                    "prompt_tokens": self.stats.prompt_tokens,
                    "completion_tokens": self.stats.completion_tokens,
                    "cached_entries": len(self.__class__._cache),
                    "tau_log": self.__class__._tau_log[-20:],
                    "updated_at": time.time(),
                }
                if extra:
                    payload.update(extra)
                tmp = progress_path.with_name(f"{progress_path.stem}.{threading.get_ident()}.tmp")
                try:
                    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    os.replace(tmp, progress_path)
                except OSError:
                    # Progress reporting is best-effort; never let it take down the run.
                    pass

        def _compute_tau_rho(self):
            current_order = self.graph.total_order()
            true_order = true_order_fn()
            tau = fast_kendalls_tau(current_order, true_order)
            tau_b = None
            if price_fn is not None:
                price = price_fn()
                pred_pos = {a: i for i, a in enumerate(current_order)}
                items = list(price.keys())
                tau_b = float(
                    kendalltau([pred_pos[a] for a in items], [price[a] for a in items]).statistic
                )
            _, _, rho = self.known_pair_counts()
            return tau, tau_b, rho

        def checkpoint_tau(self, label: str):
            """Explicit per-round checkpoint (call after R1, after R2, ...) so
            we can see which round the quality loss actually comes from,
            not just periodic snapshots during Round 3."""
            cls = self.__class__
            tau, tau_b, rho = self._compute_tau_rho()
            entry = {
                "label": label, "calls": self.stats.count, "tau_a": round(tau, 4),
                "tau_b": round(tau_b, 4) if tau_b is not None else None,
                "rho": round(rho, 4), "time": time.time(),
            }
            cls._tau_log.append(entry)
            print(
                f"  [round-checkpoint:{label}] calls={self.stats.count} tau_a={tau:.4f} "
                f"tau_b={'n/a' if tau_b is None else f'{tau_b:.4f}'} rho={rho:.4f}",
                flush=True,
            )
            self._write_progress({"last_checkpoint": entry})
            return entry

        def _maybe_check_tau(self, round_id: int):
            if round_id != 3:
                return
            cls = self.__class__
            cls._live_calls_since_check += 1
            if cls._live_calls_since_check < tau_check_interval:
                return
            cls._live_calls_since_check = 0
            tau, tau_b, rho = self._compute_tau_rho()
            entry = {
                "label": "r3_progress", "calls": self.stats.count, "tau_a": round(tau, 4),
                "tau_b": round(tau_b, 4) if tau_b is not None else None,
                "rho": round(rho, 4),
                "time": time.time(),
            }
            cls._tau_log.append(entry)
            print(
                f"  [tau-check] calls={self.stats.count} tau_a={tau:.4f} "
                f"tau_b={'n/a' if tau_b is None else f'{tau_b:.4f}'} rho={rho:.4f}",
                flush=True,
            )
            self._write_progress({"last_tau_check": entry})
            if tau < tau_stop_threshold:
                cls._consecutive_low_tau += 1
            else:
                cls._consecutive_low_tau = 0
            if cls._consecutive_low_tau >= tau_stop_consecutive:
                raise TauTooLowError(
                    f"tau stayed below {tau_stop_threshold} for "
                    f"{tau_stop_consecutive} consecutive checks (last tau={tau:.4f} "
                    f"at {self.stats.count} calls) - halting. Rerun the same "
                    "script to resume from cache if you want to continue anyway."
                )

        @staticmethod
        def _parse_prices(text: str, presentation: Sequence[str]) -> Dict[str, float]:
            batch_prices: Dict[str, float] = {}
            obj = re.search(r"\{.*\}", text, re.S)
            if obj:
                try:
                    prices = json.loads(obj.group()).get("prices")
                    if isinstance(prices, dict):
                        for k, v in prices.items():
                            if str(k).strip().isdigit() and 1 <= int(k) <= len(presentation):
                                try:
                                    batch_prices[presentation[int(k) - 1]] = float(v)
                                except (TypeError, ValueError):
                                    pass
                except (json.JSONDecodeError, AttributeError):
                    pass
            return batch_prices

        def _record_call(self, key, presentation, ordered, n_missing, batch_prices, response,
                         round_id, answer_attempt, use):
            cls = self.__class__
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            with cls._stats_lock:
                self.stats.count += 1
                self.stats.prompt_tokens += prompt_tokens
                self.stats.completion_tokens += completion_tokens
            with cls._cache_lock:
                if use:
                    cls._cache[key] = ordered
                    cls._price_estimates.update(batch_prices)
                rec = {
                    "presentation": list(presentation), "order": ordered,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "round_id": round_id, "prices": batch_prices,
                    # Provenance: ids the model omitted (appended in input order
                    # by the parser), the prompt, the snapshot actually served,
                    # and which answer attempt this was for the batch.
                    "n_missing": n_missing, "prompt_key": prompt_key,
                    "model_served": getattr(response, "model", None),
                    "answer_attempt": answer_attempt,
                }
                cls._cache_file_handle.write(json.dumps(rec) + "\n")
                cls._cache_file_handle.flush()
                os.fsync(cls._cache_file_handle.fileno())

        def _sort_with_llm(self, batch, round_id, payload=None):
            if shuffle_seed is None:
                return super()._sort_with_llm(batch, round_id, payload)
            return self._sort_with_llm_once(shuffled_presentation(batch, shuffle_seed), round_id, payload)

        def _sort_with_llm_once(
            self, presentation: Sequence[str], round_id: int, payload: Optional[Dict[str, object]] = None,
            _prefetch: bool = False,
        ) -> List[str]:
            # _prefetch: called from a Round-3 prefetch worker. The batch is only
            # counted as used (and the tau monitor advanced) when the main loop
            # consumes it, in schedule order (see _r3_consume).
            key = tuple(presentation)
            cls = self.__class__
            with cls._cache_lock:
                hit = cls._cache.get(key)
                if not _prefetch:
                    cls._used_keys.append(key)
            if hit is not None:
                return list(hit)

            with cls._stats_lock:
                if max_live_calls is not None and cls._live_calls >= max_live_calls:
                    raise LiveCallBudgetError(
                        f"live-call budget of {max_live_calls} reached at {self.stats.count} total "
                        "calls - halting. Cache is safe; rerun with a larger budget to continue."
                    )
                cls._live_calls += 1

            keys = "\n".join(f"{i}: {self.text_map[a]}" for i, a in enumerate(presentation, 1))
            create_kw: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
                    {"role": "user", "content": self.prompt_template.format(keys=keys)},
                ],
            }
            if _supports_temperature_zero(self.model):
                create_kw["temperature"] = 0
            attempts = max_attempts if strict_complete else 1
            for answer_attempt in range(1, attempts + 1):
                response = None
                # Ride out network/API outages (~30 min of backoff in total)
                # before giving up; the cache keeps every answer either way.
                for attempt in range(20):
                    try:
                        response = self.client.chat.completions.create(**create_kw)
                        break
                    except Exception as e:
                        if attempt == 19 or "insufficient_quota" in str(e):
                            raise
                        time.sleep(min(120, 5 * (attempt + 1)))
                text = response.choices[0].message.content or ""
                ordered, n_missing = self._parse_ids_detail(text, presentation)
                batch_prices = self._parse_prices(text, presentation)
                complete = n_missing == 0
                # Every attempt is paid for and logged; in strict mode an
                # incomplete one is never used (load_cache skips it too).
                self._record_call(key, presentation, ordered, n_missing, batch_prices, response,
                                  round_id, answer_attempt, use=complete or not strict_complete)
                if complete or not strict_complete:
                    break
            else:
                raise IncompleteAnswerError(
                    f"model omitted {n_missing} of {len(presentation)} items in all {attempts} attempts "
                    f"for a round-{round_id} batch - halting rather than dropping or inventing positions. "
                    "Cache is safe; rerun to retry."
                )

            if round_id == 3 and not _prefetch:
                self._maybe_check_tau(round_id)
            if self.stats.count % 200 == 0:
                self._write_progress()
            return ordered

        # ---------------- parallel Round 2 / prefetched Round 3 ----------------

        def _presentation_for(self, batch):
            """The exact presentation _sort_with_llm would send (listwise_votes == 1)."""
            if shuffle_seed is not None:
                return shuffled_presentation(batch, shuffle_seed)
            from .partition import apply_prompt_order

            return apply_prompt_order(batch, self.prompt_order, self.partition_seed)

        def _r3_fetch(self, batch, payload):
            """Thread-safe LLM fetch for a Round-3 batch: no graph access."""
            pres = self._presentation_for(batch)
            return pres, self._sort_with_llm_once(pres, 3, payload, _prefetch=True)

        def _greedy_replay_batches(self):
            """Greedy-packing batches already answered, in the order they were
            asked: complete Round-3 cache records that are not diagonal-schedule
            batches (prefetched-but-unused answers are diagonal batches)."""
            cls = self.__class__
            diag = getattr(self, "_r3_diag_keys", set())
            out, seen = [], set()
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("round_id") != 3:
                        continue
                    k = tuple(rec["presentation"])
                    if k in diag or k in seen or k not in cls._cache:
                        continue
                    if strict_complete and rec.get("n_missing", 0) > 0:
                        continue
                    seen.add(k)
                    out.append(list(k))
            # Parallel greedy rounds finish their LLM calls in any order, so the
            # cache order is not the order they were applied in. Once a run has
            # switched to parallel batches, the batches before the switch keep
            # their cache order and the later ones follow greedy_log.jsonl.
            start, applied = self._greedy_log_read()
            if start is None:
                return out
            tail = [list(k) for k in applied if k in cls._cache]
            return out[:start] + tail

        def _greedy_log_read(self):
            start, applied = None, []
            if greedy_log_path.exists():
                with greedy_log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("event") == "parallel_start" and start is None:
                            start = int(rec["serial_batches"])
                        elif rec.get("event") == "applied":
                            applied.append(tuple(rec["presentation"]))
            return start, applied

        def _greedy_log_start(self, serial_batches):
            """Record, once, how many replayed greedy batches were applied serially."""
            start, _ = self._greedy_log_read()
            if start is None:
                self._greedy_log_write({"event": "parallel_start", "serial_batches": serial_batches})

        def _greedy_log_applied(self, presentation):
            self._greedy_log_write({"event": "applied", "presentation": list(presentation)})

        def _greedy_log_write(self, rec):
            # One handle for the whole run, like the cache file: reopening it per batch
            # lets file-sync tools lock it in between.
            cls = self.__class__
            if getattr(cls, "_greedy_log_handle", None) is None:
                for attempt in range(30):
                    try:
                        cls._greedy_log_handle = greedy_log_path.open("a", encoding="utf-8")
                        break
                    except PermissionError:
                        if attempt == 29:
                            raise
                        time.sleep(2)
            cls._greedy_log_handle.write(json.dumps(rec) + "\n")
            cls._greedy_log_handle.flush()

        def _r3_consume(self, presentation):
            """Main thread: record a prefetched batch as used, in schedule order."""
            cls = self.__class__
            with cls._cache_lock:
                cls._used_keys.append(tuple(presentation))
            self._maybe_check_tau(3)

        def _sort_batch_round2(self, batch, row_id):
            # Same as BatchedLLMSorter._sort_batch_round2, with graph access locked
            # and the LLM call outside the lock.
            if len(batch) <= 1:
                return list(batch)
            with self._graph_lock:
                if self._batch_pairs_known(batch):
                    return self._order_batch_from_graph(batch)
            ordered = self._sort_with_llm(batch, round_id=2, payload={"row": row_id, "mode": self.round2_mode})
            with self._graph_lock:
                self.graph.add_total_order(ordered, direct=True)
            return ordered

        def _recursive_merge_sort_row(self, row_items, row_id):
            """BatchedLLMSorter._recursive_merge_sort_row with the independent work of
            each level run concurrently. Chunks, and the run pairs merged at one
            level, are disjoint sets of column heads with no graph path between
            them (Round-1 columns are disjoint), so their graph updates commute and
            the resulting graph and orders are the same as the sequential version."""
            if r2_workers <= 1:
                return super()._recursive_merge_sort_row(row_items, row_id)
            m = self.batch_size
            if len(row_items) <= m:
                return self._sort_batch_round2(row_items, row_id=row_id)
            chunks = [row_items[i : i + m] for i in range(0, len(row_items), m)]
            with ThreadPoolExecutor(max_workers=r2_workers) as pool:
                sorted_chunks = list(pool.map(lambda ch: self._sort_batch_round2(ch, row_id=row_id), chunks))
                level = 0
                while len(sorted_chunks) > 1:
                    pairs = [(sorted_chunks[i], sorted_chunks[i + 1]) for i in range(0, len(sorted_chunks) - 1, 2)]
                    merged = list(pool.map(lambda pr: self._merge_two_sorted_runs(pr[0], pr[1], row_id=row_id), pairs))
                    if len(sorted_chunks) % 2:
                        merged.append(sorted_chunks[-1])
                    sorted_chunks = merged
                    level += 1
                    print(f"  Round 2 merge level {level}: {len(sorted_chunks)} run(s), calls_total={self.stats.count}",
                          flush=True)
            ordered = sorted_chunks[0]
            with self._graph_lock:
                self.graph.add_total_order(ordered, direct=True)
            return ordered

        def _round_one(self) -> List[List[str]]:
            from .partition import build_round1_batches

            round1_batches = build_round1_batches(
                self.items, self.batch_size, self.round1_partition,
                seed=self.partition_seed,
                height_map=self.height_map if self.sort_by == "height" else None,
                text_map=self.text_map if self.text_map else None,
            )

            def sort_one(idx_batch):
                idx, batch = idx_batch
                if len(batch) == 1:
                    return idx, batch
                ordered = self._sort_with_llm(batch, round_id=1, payload={"column": idx})
                is_valid, validation_report = self._validate_batch_output(ordered, batch)
                if not is_valid:
                    ordered = self._correct_batch_output(ordered, batch, validation_report)
                return idx, ordered

            indexed = list(enumerate(round1_batches))
            results: List[Optional[List[str]]] = [None] * len(indexed)
            with ThreadPoolExecutor(max_workers=r1_workers) as pool:
                for idx, ordered in pool.map(sort_one, indexed):
                    results[idx] = ordered

            columns: List[List[str]] = []
            for ordered in results:
                columns.append(ordered)
                self.graph.add_total_order(ordered, direct=True)
            if isinstance(self.graph, DagTransitiveOrder):
                # Round-1 columns are disjoint paths covering every item: the
                # chain cover the exact pair-counting passes run over.
                self.graph.set_chain_cover(columns)
            print(f"  Round 1 done: {len(columns)} columns, calls_total={self.stats.count}", flush=True)
            self._write_progress({"round": "r1_done"})
            return columns

        def _on_round3_truncate(self, snapshot):
            # Checkpoints taken after the theta stop belong to rolled-back batches.
            cls = self.__class__
            cls._tau_log[:] = [e for e in cls._tau_log if e["calls"] <= snapshot["calls"]]
            # One LLM call per executed Round-3 batch (listwise_votes == 1).
            if self.round3_overshoot["batches"]:
                del cls._used_keys[-self.round3_overshoot["batches"]:]
            print(
                f"  [theta] rho crossed theta {self.round3_overshoot['batches']} batch(es) before the "
                f"last exact check; rolled back {self.round3_overshoot['calls']} overshoot call(s)",
                flush=True,
            )

        def sort(self):
            cls = self.__class__
            calls_before = self.stats.count
            t0 = time.time()
            columns = self._round_one()
            self.round1_calls = self.stats.count - calls_before
            cls._timing["r1_s"] = round(time.time() - t0, 1)
            self.checkpoint_tau("after_r1")
            if stop_after_round1:
                raise StopAfterRound1(f"Round 1 complete ({self.stats.count} calls cached); stopping as requested.")
            if self.graph.is_complete():
                self.round2_calls = 0
                cls._timing["r2_s"] = 0.0
                cls._timing["r3_s"] = 0.0
                return self.graph.total_order(), self.stats

            calls_before_r2 = self.stats.count
            t1 = time.time()
            self._round_two(columns)
            self.round2_calls = self.stats.count - calls_before_r2
            cls._timing["r2_s"] = round(time.time() - t1, 1)
            print(
                f"  Round 2 done: calls_total={self.stats.count} "
                f"(r1={self.round1_calls} r2={self.round2_calls})",
                flush=True,
            )
            self._write_progress({"round": "r2_done"})
            self.checkpoint_tau("after_r2")
            if self.graph.is_complete():
                cls._timing["r3_s"] = 0.0
                return self.graph.total_order(), self.stats

            if isinstance(self.graph, ChainTransitiveOrder):
                self.chain_stats["chains_after_r2"] = self.graph.num_chains
                self.chain_stats["label_ints_after_r2"] = self.graph.label_entries()
                self.graph.rechain()
                self.chain_stats["chains_after_rechain"] = self.graph.num_chains
                self.chain_stats["label_ints_after_rechain"] = self.graph.label_entries()

            if self.grid_update_mode != "none":
                columns = self._update_grid_after_r2(columns)

            t2 = time.time()
            self._round_three(columns)
            cls._timing["r3_s"] = round(time.time() - t2, 1)
            print(f"  Round 3 done: calls_total={self.stats.count}", flush=True)
            self._write_progress({"round": "r3_done"})
            self.checkpoint_tau("after_r3_final")
            return self.graph.total_order(), self.stats

    return ResumableAmazonSorter
