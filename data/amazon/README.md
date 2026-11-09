# Amazon price ranking (1M products)

`amazon_1m_price_gt.csv` has 1,000,000 Amazon products with their listed USD price and a strict
ground-truth rank. It is 178 MB, too large for GitHub, so it is not committed.

| | |
|---|---|
| Download (prebuilt) | **TODO: add the hosted link (e.g. Zenodo / Hugging Face) before release** |
| SHA-256 | `c6805cf9ec2f31ba24e40822d8148db7a44dce7b90c1c31211882692423e9c60` |
| Source | Amazon Reviews 2023 product metadata, McAuley Lab: https://amazon-reviews-2023.github.io/ |
| Raw files | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz` |

Place the downloaded file here, as `data/amazon/amazon_1m_price_gt.csv`.

## Rebuilding it from the source

```bash
python scripts/fetch_amazon_priced.py       # streams 21 category dumps (~14 GB raw, deleted after
                                             # filtering) -> data/amazon/priced/<Category>.csv.gz (473 MB)
python scripts/build_amazon_gt.py           # -> data/amazon/amazon_1m_price_gt.csv (+ _stats.json)
```

The build keeps products with a title and a price in [$0.50, $10,000]. It dedupes by `parent_asin`
and by display text, then samples 1,000,000 products with seed 0.

The ground-truth rank sorts by price ascending. Ties are broken by display text, then by ASIN.

Columns: `gt_rank, parent_asin, display, price, source_category, main_category, store, average_rating, rating_number`.

Citation:

```bibtex
@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```
