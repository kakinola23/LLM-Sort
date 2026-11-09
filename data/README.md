# Data

Each dataset has a folder here. Small public files are committed. Data whose license forbids
redistribution, or that is too large for GitHub, is built by a script under `scripts/`.

| Dataset | Folder | Items | Committed | Build / download |
|---|---|---|---|---|
| NBA heights | `nba/` | 200 players | `nba_heights_200.csv` (from OrderByLLM, MIT) | none |
| World population 2020 | `population/` | 200 countries | `population_by_country_2020.csv` (from OrderByLLM, MIT) | none |
| TREC DL19 passages | `dl19/` | 43 queries × BM25 top-100 | BM25 run, qrels, query text | `python scripts/prepare_dl19.py` (MS MARCO passage text) |
| SciFact | `scifact/`, `scifact_slice/` | 50 claims × E5 top-100 | `scifact_slice/e5_top100_test50.json` (claim text + candidate ids) | `python scripts/prepare_scifact.py` (BEIR corpus, queries, qrels) |
| MovieLens-1M | `movielens/` | 63 users × 70 movies | none (GroupLens license) | `python scripts/prepare_movielens.py` |
| Amazon Reviews 2023 prices | `amazon/` | up to 1,000,000 products | none (178 MB) | see [`amazon/README.md`](amazon/README.md) |

Expected SHA-256 of the built files:

| File | SHA-256 |
|---|---|
| `dl19/dl19_passages_43x100.csv` | `c556558a6916079e096c25264255851d774460a8a10180699c6a3eb211a21ebb` |
| `movielens/movielens_phase1_manifest.json` | `de5340eeb4b9563eea66fab64f636397fcdf6655fe42d814215e70fe0976aa08` |
| `amazon/amazon_1m_price_gt.csv` | `c6805cf9ec2f31ba24e40822d8148db7a44dce7b90c1c31211882692423e9c60` |

## Sources and licenses

- **NBA heights, world population:** the `data/` folder of OrderByLLM (Zhao et al.), https://github.com/ZhaoFuheng/OrderByLLM (MIT).
- **TREC DL19:**
  - Qrels from the TREC 2019 Deep Learning track.
  - BM25 run from Pyserini (`run.msmarco-v1-passage.bm25-default.dl19.txt`).
  - Passage text from MS MARCO (https://microsoft.github.io/msmarco/). MS MARCO is for non-commercial research use; the text is downloaded, not redistributed.
- **SciFact:** BEIR (https://github.com/beir-cellar/beir), CC BY-NC 2.0. Candidates were retrieved with `intfloat/e5-base-v2`, as in LOTUS.
- **MovieLens-1M:** GroupLens (https://grouplens.org/datasets/movielens/1m/). Redistribution needs GroupLens' permission, so only the builder script is committed.
- **Amazon Reviews 2023:** McAuley Lab, UCSD (https://amazon-reviews-2023.github.io/), Hou et al. 2024.
