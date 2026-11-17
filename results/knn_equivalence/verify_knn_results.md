# FAISS vs reference k-NN

Output of `verify_knn_equivalence.py`, shown in the thesis appendix. Each run
compares `faiss.knn` with a reference on the first 10 000 and 50 000 rows of a
dataset, with k ∈ {10, 30}. The query itself is removed before the comparison.

Each dataset is checked under the metric it uses in the thesis:

- BigANN (uint8 source, used as float32) - L2
- Deep1B (float32) - L2
- LAION (float16) - IP

References:

- **L2**: `utils.dist_rank`.
- **IP**: direct `torch.topk(X @ X.T, k+1, largest=True)`. The normalized branch
  of `utils.dist_rank` returns the farthest neighbors first.

Column meaning:

- `set overlap` - `|A ∩ B| / k`, averaged across queries.
- `exact`       - fraction of `(query, rank)` pairs where the reference ID
  equals the FAISS ID (drops with tie-breaking).
- `perfect`     - number of queries whose top-k *set* is identical.

---

## Deep1B-1M (d=96, float32) - L2

```
dataset                         n   k metric   set overlap       exact      perfect
----------------------------------------------------------------------------------------
deep1b[:10k]                10000  10     l2    100.0000%  100.0000%      10,000
deep1b[:10k]                10000  30     l2     99.9997%   99.9997%       9,999
deep1b[:50k]                50000  10     l2     99.9994%   99.9914%      49,997
deep1b[:50k]                50000  30     l2     99.9997%   99.9904%      49,996
```

Worst-case set overlap: **99.9994%**.

---

## BigANN-1M (d=128, uint8 source as float32) - L2

```
dataset                         n   k metric   set overlap       exact      perfect
----------------------------------------------------------------------------------------
bigann[:10k]                10000  10     l2     99.9950%   99.9750%       9,995
bigann[:10k]                10000  30     l2     99.9973%   99.9427%       9,992
bigann[:50k]                50000  10     l2     99.9960%   99.9712%      49,980
bigann[:50k]                50000  30     l2     99.9975%   99.9207%      49,962
```

Worst-case set overlap: **99.9950%**. The coordinates are integers, so distances
are often exactly equal.

---

## LAION 100M (first 50 000 rows, d=768, float16) - IP

```
dataset                         n   k metric   set overlap       exact      perfect
----------------------------------------------------------------------------------------
laion[:10k]                 10000  10     ip     99.9690%   99.6870%       9,976
laion[:10k]                 10000  30     ip     99.9850%   99.6527%       9,977
laion[:50k]                 50000  10     ip     99.9300%   99.2944%      49,755
laion[:50k]                 50000  30     ip     99.9390%   99.1804%      49,610
```

Worst-case set overlap: **99.9300%**.

---

## Summary

Set overlap is at least **99.93%** on every dataset, slice size and k under the
metric used in the thesis. Both methods run the same exact search, so they can
differ only where distances are equal or nearly equal after rounding.
