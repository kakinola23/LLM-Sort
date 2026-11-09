# Vendored: OrderByLLM (Zhao et al.)

Unmodified files from the `order_by/` and `prompts/` packages of
https://github.com/ZhaoFuheng/OrderByLLM at commit `47e6a104abc196593a8393e3309a26194ef30297`
(MIT License, see `LICENSE`). Only the modules the baselines use are kept;
`order_by/optimizer.py` and `order_by/tools/` (web-search helpers) are omitted.

We use it for the baselines (external merge sort, quicksort, pointwise) and for the listwise
prompt wording that TBS shares with those baselines.
