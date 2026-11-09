#!/usr/bin/env python3
"""
Build data/dl19/dl19_passages_43x100.csv: TREC DL19, 43 queries x BM25 top-100 passages
(OrderByLLM protocol). MS MARCO passage text is not redistributed, so it is downloaded here.

Output CSV (no header): query_id, query_text, bm25_rank, doc_id, passage_text, relevance

Sources (the first three ship with the repo):
  - data/dl19/run.msmarco-v1-passage.bm25-default.dl19.txt  (Pyserini BM25 run, top-100 per query)
  - data/dl19/dl19_queries_43.csv                           (query text)
  - data/dl19/2019qrels-pass.txt                            (TREC DL19 passage qrels, 0-3)
  - msmarco-passagetest2019-top1000.tsv.gz (downloaded) and, for the few passages it lacks,
    a single streaming pass over MS MARCO collection.tar.gz (~1 GB, not stored).

  python scripts/prepare_dl19.py
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

try:  # use certifi's CA bundle when the system store lacks the issuer (common on Windows Python)
    import ssl

    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dl19"
RUN_FILE = DATA / "run.msmarco-v1-passage.bm25-default.dl19.txt"
QUERIES_CSV = DATA / "dl19_queries_43.csv"
QRELS_FILE = DATA / "2019qrels-pass.txt"
PASSAGES_GZ = DATA / "msmarco-passagetest2019-top1000.tsv.gz"
PASSAGES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz"
OUT_CSV = DATA / "dl19_passages_43x100.csv"
OUT_JSON = DATA / "dl19_passages_43x100_summary.json"
HIT_DEPTH = 100


def load_queries() -> dict[str, str]:
    out: dict[str, str] = {}
    with QUERIES_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = row["query_text"]
    return out


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with QRELS_FILE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            qrels[qid][docid] = rel
    return dict(qrels)


def load_passage_lookup() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """(qid, pid) and pid -> passage text from MS MARCO DL19 top-1000 export."""
    by_qpid: dict[tuple[str, str], str] = {}
    by_pid: dict[str, str] = {}
    with gzip.open(PASSAGES_GZ, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, pid, passage = parts[0], parts[1], parts[3]
            by_qpid[(qid, pid)] = passage
            by_pid.setdefault(pid, passage)
    return by_qpid, by_pid


def fill_missing_from_collection(
    needed: set[str], by_pid: dict[str, str]
) -> dict[str, str]:
    """Stream MS MARCO collection.tar.gz and pull only missing pids."""
    still = {pid for pid in needed if pid not in by_pid}
    if not still:
        return {}

    import tarfile
    import urllib.request

    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
    print(f"Streaming {url} for {len(still)} missing passages...")
    found: dict[str, str] = {}
    with urllib.request.urlopen(url, timeout=900) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith("collection.tsv"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    break
                for raw in f:
                    pid, _, text = raw.partition(b"\t")
                    pid_s = pid.decode("utf-8", errors="replace")
                    if pid_s in still:
                        found[pid_s] = text.decode("utf-8", errors="replace").rstrip("\n")
                        still.remove(pid_s)
                        if not still:
                            break
                break
    print(f"Collection fill: found {len(found)} passages")
    return found


def load_bm25_top100() -> dict[str, list[str]]:
    by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with RUN_FILE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, _, _ = parts
            r = int(rank)
            if r <= HIT_DEPTH:
                by_q[qid].append((r, docid))
    out: dict[str, list[str]] = {}
    for qid, entries in by_q.items():
        entries.sort(key=lambda x: x[0])
        out[qid] = [docid for _, docid in entries]
    return out


def main() -> None:
    if not PASSAGES_GZ.is_file():
        import urllib.request

        print(f"downloading {PASSAGES_URL} ...", flush=True)
        urllib.request.urlretrieve(PASSAGES_URL, PASSAGES_GZ)
    for p in (RUN_FILE, QUERIES_CSV, QRELS_FILE, PASSAGES_GZ):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    queries = load_queries()
    qrels = load_qrels()
    by_qpid, by_pid = load_passage_lookup()
    bm25 = load_bm25_top100()

    qids = sorted(bm25.keys(), key=int)
    all_pids = {docid for docs in bm25.values() for docid in docs}
    hf_fill = fill_missing_from_collection(all_pids, by_pid)
    by_pid.update(hf_fill)
    rows: list[list[str | int]] = []
    missing_text: list[str] = []
    per_query: list[dict] = []

    for qid in qids:
        if qid not in queries:
            raise KeyError(f"Query {qid} missing from {QUERIES_CSV}")
        docs = bm25[qid]
        if len(docs) != HIT_DEPTH:
            raise ValueError(f"Query {qid}: expected {HIT_DEPTH} docs, got {len(docs)}")

        judged = qrels.get(qid, {})
        n_judged_in_top100 = sum(1 for d in docs if d in judged)
        n_rel = sum(1 for d in docs if judged.get(d, 0) >= 2)

        for rank, docid in enumerate(docs, start=1):
            text = by_qpid.get((qid, docid)) or by_pid.get(docid, "")
            if not text:
                missing_text.append(f"{qid}:{docid}")
                text = "[PASSAGE TEXT NOT FOUND]"
            rel = judged.get(docid, 0)
            rows.append([qid, queries[qid], rank, docid, text, rel])

        per_query.append(
            {
                "query_id": qid,
                "query_text": queries[qid],
                "n_passages": HIT_DEPTH,
                "n_judged_in_top100": n_judged_in_top100,
                "n_rel_ge2_in_top100": n_rel,
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)

    summary = {
        "n_queries": len(qids),
        "n_rows": len(rows),
        "hit_depth": HIT_DEPTH,
        "missing_passage_text": len(missing_text),
        "missing_passage_examples": missing_text[:20],
        "queries": per_query,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV} ({len(rows)} rows, {len(qids)} queries)")
    print(f"Wrote {OUT_JSON}")
    print(f"Missing passage text: {len(missing_text)}")
    if missing_text:
        print("Examples:", missing_text[:5])


if __name__ == "__main__":
    main()
