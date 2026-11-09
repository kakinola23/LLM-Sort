"""
Round-1 batch membership and within-batch prompt presentation helpers.

Used by batch-selection ablations (contiguous vs random vs similarity clusters).
"""

from __future__ import annotations

import math
import os
import random
import re
import threading
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

PARTITION_MODES = ("contiguous", "random", "similarity", "similarity_embed")
PROMPT_ORDER_MODES = ("input", "random_shuffle")

# Cheap local embedder (sentence-transformers). Override with BATCH_EMBED_MODEL env var.
DEFAULT_EMBED_MODEL = os.getenv(
    "BATCH_EMBED_MODEL",
    "BAAI/bge-small-en-v1.5",  # fastembed default; use sentence-transformers/all-MiniLM-L6-v2 with ST backend
)

# Round-1 membership presets.
SELECTION_PRESETS: Dict[str, Dict[str, str]] = {
    "contiguous": {"round1_partition": "contiguous", "prompt_order": "input"},
    "random": {"round1_partition": "random", "prompt_order": "input"},
    "similarity": {"round1_partition": "similarity", "prompt_order": "input"},
    "similarity_embed": {"round1_partition": "similarity_embed", "prompt_order": "input"},
}

_EMBEDDER = None
_EMBEDDER_MODEL_NAME: Optional[str] = None
_FASTEMBED = None
_FASTEMBED_MODEL_NAME: Optional[str] = None
_FASTEMBED_LOCK = threading.Lock()


def partition_contiguous(items: Sequence[str], batch_size: int) -> List[List[str]]:
    batches: List[List[str]] = []
    for start in range(0, len(items), batch_size):
        batches.append(list(items[start : start + batch_size]))
    return batches


def partition_random(items: Sequence[str], batch_size: int, seed: int = 42) -> List[List[str]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return partition_contiguous(shuffled, batch_size)


def _tokenize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def partition_similarity_greedy(
    items: Sequence[str],
    batch_size: int,
    similarity_fn: Callable[[str, str], float],
) -> List[List[str]]:
    """Seed with the first unassigned item; fill each batch with most similar neighbors."""
    remaining = list(items)
    batches: List[List[str]] = []
    while remaining:
        if len(remaining) <= batch_size:
            batches.append(remaining)
            break
        seed = remaining[0]
        batch = [seed]
        rest = remaining[1:]
        while len(batch) < batch_size and rest:
            best_idx = max(range(len(rest)), key=lambda i: similarity_fn(seed, rest[i]))
            batch.append(rest.pop(best_idx))
        batches.append(batch)
        remaining = rest
    return batches


def partition_similarity_text(
    items: Sequence[str],
    batch_size: int,
    text_map: Dict[str, str],
) -> List[List[str]]:
    vecs = {it: _tokenize(text_map.get(it, it)) for it in items}
    return partition_similarity_greedy(
        items,
        batch_size,
        lambda a, b: _cosine(vecs[a], vecs[b]),
    )


def _get_sentence_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Lazy-load a small sentence-transformers model (~80MB, CPU-friendly)."""
    global _EMBEDDER, _EMBEDDER_MODEL_NAME
    if _EMBEDDER is None or _EMBEDDER_MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_MODEL_NAME = model_name
    return _EMBEDDER


def _get_fastembed_embedder(model_name: str):
    """Lazy-load fastembed once per model (avoids re-fetching ONNX weights every query)."""
    global _FASTEMBED, _FASTEMBED_MODEL_NAME
    if _FASTEMBED is None or _FASTEMBED_MODEL_NAME != model_name:
        from fastembed import TextEmbedding

        _FASTEMBED = TextEmbedding(model_name=model_name)
        _FASTEMBED_MODEL_NAME = model_name
    return _FASTEMBED


def _embedding_vectors_fastembed(items: Sequence[str], text_map: Dict[str, str], model_name: str) -> Dict[str, "object"]:
    import numpy as np

    texts = [text_map.get(it, it) for it in items]
    # ONNX runtime is not reliably thread-safe; serialize embed calls on the shared model.
    with _FASTEMBED_LOCK:
        embedder = _get_fastembed_embedder(model_name)
        mat = np.array(list(embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-9, None)
    return {it: mat[i] for i, it in enumerate(items)}


def _embedding_vectors_sentence_transformers(
    items: Sequence[str], text_map: Dict[str, str], model_name: str
) -> Dict[str, "object"]:
    texts = [text_map.get(it, it) for it in items]
    model = _get_sentence_embedder(model_name)
    mat = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {it: mat[i] for i, it in enumerate(items)}


def _embedding_vectors(items: Sequence[str], text_map: Dict[str, str], model_name: str) -> Dict[str, "object"]:
    backend = os.getenv("BATCH_EMBED_BACKEND", "fastembed").lower()
    if backend == "sentence_transformers":
        return _embedding_vectors_sentence_transformers(items, text_map, model_name)
    try:
        return _embedding_vectors_fastembed(items, text_map, model_name)
    except Exception as exc:
        if backend == "fastembed":
            raise
        return _embedding_vectors_sentence_transformers(items, text_map, model_name)


def partition_similarity_embedding(
    items: Sequence[str],
    batch_size: int,
    text_map: Dict[str, str],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> List[List[str]]:
    """Greedy star clusters using cosine similarity of cheap sentence embeddings."""
    import numpy as np

    vecs = _embedding_vectors(items, text_map, model_name)

    def sim(a: str, b: str) -> float:
        return float(np.dot(vecs[a], vecs[b]))  # vectors are L2-normalized

    return partition_similarity_greedy(items, batch_size, sim)


def partition_similarity_height(
    items: Sequence[str],
    batch_size: int,
    height_map: Dict[str, float],
) -> List[List[str]]:
    """Height-proximity clusters (ablation only; uses oracle heights for grouping)."""

    def sim(a: str, b: str) -> float:
        return 1.0 / (1.0 + abs(height_map[a] - height_map[b]))

    return partition_similarity_greedy(items, batch_size, sim)


def build_round1_batches(
    items: Sequence[str],
    batch_size: int,
    partition_mode: str,
    *,
    seed: int = 42,
    height_map: Optional[Dict[str, float]] = None,
    text_map: Optional[Dict[str, str]] = None,
) -> List[List[str]]:
    mode = partition_mode.lower()
    if mode == "contiguous":
        return partition_contiguous(items, batch_size)
    if mode == "random":
        return partition_random(items, batch_size, seed=seed)
    if mode == "similarity":
        if text_map is not None:
            return partition_similarity_text(items, batch_size, text_map)
        if height_map is not None and os.getenv("TBS_ALLOW_ORACLE_SIMILARITY") == "1":
            # Oracle ablation only: groups by TRUE values. Never used unless explicitly enabled.
            return partition_similarity_height(items, batch_size, height_map)
        raise ValueError("similarity partition requires text_map (oracle height grouping is disabled)")
    if mode in ("similarity_embed", "similarity_embedding"):
        if text_map is None:
            raise ValueError("similarity_embed partition requires text_map (passage snippets)")
        return partition_similarity_embedding(items, batch_size, text_map)
    raise ValueError(f"Unknown partition_mode: {partition_mode!r}")


def apply_prompt_order(batch: Sequence[str], mode: str, seed: int = 42) -> List[str]:
    items = list(batch)
    m = mode.lower()
    if m in ("input", "contiguous"):
        return items
    if m in ("random", "random_shuffle"):
        rng = random.Random(seed + (hash(tuple(sorted(batch))) % 10_000))
        out = list(items)
        rng.shuffle(out)
        return out
    raise ValueError(f"Unknown prompt_order mode: {mode!r}")


def resolve_selection_preset(name: str) -> Dict[str, str]:
    key = name.lower().replace("-", "_")
    if key not in SELECTION_PRESETS:
        raise ValueError(
            f"Unknown selection preset {name!r}. Choose from: {list(SELECTION_PRESETS)}"
        )
    return dict(SELECTION_PRESETS[key])
