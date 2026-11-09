"""
Append-only replay cache for OpenAI chat calls, so an interrupted TBS run can be restarted
with the same command and resume where it stopped.

How it works
  * Every request (model, messages, temperature, response schema, ...) is hashed to a key.
  * The n-th time a key is requested in a run maps to the n-th stored reply for that key, so
    identical re-asks (retries after an omitted id) replay in their original order.
  * On a hit the stored reply is returned with its original `usage`, so call and token totals
    of a resumed run equal those of an uninterrupted run (each call is paid for once).
  * On a miss the live API is called and the reply is appended (one JSON line, flushed).

Enabled by the TBS_CALL_CACHE environment variable (a .jsonl path). The path is read at the
first API call, so runners may set it after the client is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_STORES: Dict[str, "_Store"] = {}


class _Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self.replies: Dict[str, List[dict]] = defaultdict(list)
        self.seen: Dict[str, int] = defaultdict(int)
        self.hits = 0
        self.misses = 0
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line from a crash: ignore, that call is re-asked
                    self.replies[row["k"]].append(row["resp"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def take(self, key: str) -> Optional[dict]:
        with _LOCK:
            i = self.seen[key]
            self.seen[key] += 1
            stored = self.replies.get(key, [])
            if i < len(stored):
                self.hits += 1
                return stored[i]
            self.misses += 1
            return None

    def put(self, key: str, resp: dict) -> None:
        with _LOCK:
            self.replies[key].append(resp)
            self._fh.write(json.dumps({"k": key, "resp": resp}) + "\n")
            self._fh.flush()


def _store() -> Optional[_Store]:
    path = os.getenv("TBS_CALL_CACHE", "").strip()
    if not path:
        return None
    with _LOCK:
        if path not in _STORES:
            _STORES[path] = _Store(path)
        return _STORES[path]


def cache_stats() -> Dict[str, int]:
    s = _store()
    return {"cache_hits": s.hits, "cache_misses": s.misses} if s else {}


def _key(api: str, kwargs: Dict[str, Any]) -> str:
    kw = dict(kwargs)
    rf = kw.get("response_format")
    if isinstance(rf, type):
        kw["response_format"] = {"name": rf.__name__, "schema": rf.model_json_schema()}
    blob = json.dumps({"api": api, **kw}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _Completions:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        from openai.types.chat import ChatCompletion

        s = _store()
        if s is None:
            return self._inner.create(**kwargs)
        k = _key("create", kwargs)
        hit = s.take(k)
        if hit is not None:
            return ChatCompletion.model_validate(hit)
        resp = self._inner.create(**kwargs)
        s.put(k, resp.model_dump(mode="json"))
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ParseCompletions:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def parse(self, **kwargs: Any) -> Any:
        from openai.types.chat import ParsedChatCompletion

        s = _store()
        if s is None:
            return self._inner.parse(**kwargs)
        k = _key("parse", kwargs)
        hit = s.take(k)
        rf = kwargs.get("response_format")
        if hit is not None:
            model = ParsedChatCompletion[rf] if isinstance(rf, type) else ParsedChatCompletion
            return model.model_validate(hit)
        resp = self._inner.parse(**kwargs)
        s.put(k, resp.model_dump(mode="json"))
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: Any, parse: bool) -> None:
        self._inner = inner
        self.completions = _ParseCompletions(inner.completions) if parse else _Completions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Beta:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat = _Chat(inner.chat, parse=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CachedOpenAI:
    """Drop-in wrapper over an OpenAI client; caches chat.completions.create and
    beta.chat.completions.parse when TBS_CALL_CACHE is set, else passes through."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _Chat(client.chat, parse=False)
        self.beta = _Beta(client.beta)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
