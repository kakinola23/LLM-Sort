#!/usr/bin/env python3
"""Download BEIR SciFact (corpus, queries, qrels) into data/scifact/.

The 50-claim slice and its E5 top-100 candidates ship in data/scifact_slice/e5_top100_test50.json;
``--retrieve`` recomputes them with intfloat/e5-base-v2 (needs sentence-transformers; embeddings
can differ slightly across hardware, so the shipped file is what the paper used).

  python scripts/prepare_scifact.py
  python scripts/prepare_scifact.py --retrieve      # regenerate the candidate file
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

try:  # use certifi's CA bundle when the system store lacks the issuer (common on Windows Python)
    import ssl

    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA = ROOT / "data"


def download() -> None:
    if (DATA / "scifact" / "corpus.jsonl").is_file():
        print("data/scifact already present")
        return
    print(f"downloading {URL} ...", flush=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA)
    print("wrote data/scifact/")


def retrieve() -> None:
    """Top-100 abstracts per claim with intfloat/e5-base-v2 ('query: ' / 'passage: ' prefixes)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from loaders.scifact import SLICE_FILE, load_corpus

    sl = json.loads(SLICE_FILE.read_text(encoding="utf-8"))
    corpus = load_corpus()
    ids = list(corpus)
    model = SentenceTransformer("intfloat/e5-base-v2")
    docs = model.encode([f"passage: {corpus[d]}" for d in ids], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    for c in sl["claims"]:
        q = model.encode([f"query: {c['claim']}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        top = np.argsort(-(docs @ q))[:100]
        c["top100"] = [ids[i] for i in top]
    sl.update({"retriever": "intfloat/e5-base-v2", "backend": "st"})
    SLICE_FILE.write_text(json.dumps(sl, indent=1), encoding="utf-8")
    print(f"rewrote {SLICE_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieve", action="store_true")
    args = ap.parse_args()
    download()
    if args.retrieve:
        retrieve()


if __name__ == "__main__":
    main()
