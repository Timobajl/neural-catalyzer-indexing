#!/usr/bin/env python3
"""
Check that FAISS returns the same exact k-NN as the reference code.

Under L2 it compares ``faiss.knn`` with ``utils.dist_rank``. Both are exact, so
they can differ only where distances are equal or nearly equal after rounding.

Under inner product it compares ``faiss.knn`` with a direct exact search, because
the normalized branch of ``utils.dist_rank`` returns the farthest neighbors.

The output is the table in the thesis appendix.

Usage:
    python verify_knn_equivalence.py --xb data/bigann_xb.npy --label bigann1m
    python verify_knn_equivalence.py --xb data/laion_xb.npy  --label laion
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import faiss
import numpy as np
import torch

import utils


def strip_self(I: np.ndarray, k: int) -> np.ndarray:
    """Drop the query's own ID from each row, return the first k remaining."""
    out = np.full((I.shape[0], k), -1, dtype=np.int64)
    for i in range(I.shape[0]):
        keep = I[i][I[i] != i]
        out[i, : min(len(keep), k)] = keep[:k]
    return out


def reference_ip_knn(X: np.ndarray, k: int) -> np.ndarray:
    """Direct exact inner-product top-k from the full similarity matrix."""
    X_t = torch.from_numpy(X)
    S = X_t @ X_t.t()
    return torch.topk(S, k + 1, dim=1, largest=True).indices.cpu().numpy()


def compare(name: str, X: np.ndarray, k: int, metric: str) -> dict:
    n, d = X.shape
    print(f"\n[{name}]  n={n:,}  d={d}  k={k}  metric={metric}")

    t0 = time.time()
    if metric == "l2":
        opt = SimpleNamespace(normalize_data=False, sift=False)
        Y_ref = utils.dist_rank(torch.from_numpy(X), k, opt=opt).cpu().numpy()
        ref_name = "dist_rank (paper, L2)"
    else:
        Y_ref = reference_ip_knn(X, k)
        ref_name = "direct search (IP)"
    t_ref = time.time() - t0

    t0 = time.time()
    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    _, Y_faiss = faiss.knn(X, X, k + 1, metric=faiss_metric)
    t_faiss = time.time() - t0

    P = strip_self(Y_ref, k)
    F = strip_self(Y_faiss, k)
    set_overlap = float(np.mean([len(set(P[i].tolist()) & set(F[i].tolist())) / k for i in range(n)]))
    exact_match = float(np.mean(P == F))
    perfect_rows = int(np.sum([set(P[i].tolist()) == set(F[i].tolist()) for i in range(n)]))

    print(f"  reference ({ref_name}): {t_ref:6.1f}s")
    print(f"  faiss backend (knn):       {t_faiss:6.1f}s")
    print(f"  speedup:                   {t_ref / max(t_faiss, 1e-3):6.1f}x")
    print(f"  set overlap:               {set_overlap:.6%}")
    print(f"  exact match (per-position):{exact_match:.6%}")
    print(f"  perfect-set rows:          {perfect_rows:,} / {n:,}")
    return {
        "name": name, "n": n, "d": d, "k": k, "metric": metric,
        "set_overlap": set_overlap, "exact_match": exact_match,
        "perfect_rows": perfect_rows,
        "ref_time_s": t_ref, "faiss_time_s": t_faiss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS vs dist_rank equivalence check.")
    parser.add_argument("--xb", required=True, help="Base vectors .npy")
    parser.add_argument("--label", default="dataset")
    args = parser.parse_args()

    xb = np.load(args.xb, mmap_mode="r")
    print(f"[{args.label}] {args.xb}  shape={xb.shape}  dtype={xb.dtype}")

    rows: list[dict] = []
    for n_sample in (10_000, 50_000):
        if n_sample > xb.shape[0]:
            continue
        X = np.ascontiguousarray(xb[:n_sample]).astype("float32")
        for k in (10, 30):
            for metric in ("l2", "ip"):
                rows.append(compare(f"{args.label}[:{n_sample // 1000}k]", X, k, metric))

    print("\n" + "=" * 96)
    print(
        f"{'dataset':24}  {'n':>7} {'k':>3} {'metric':>6}  "
        f"{'set overlap':>12}  {'exact':>10}  {'perfect':>11}  {'speedup':>8}"
    )
    print("-" * 96)
    for r in rows:
        speedup = r["ref_time_s"] / max(r["faiss_time_s"], 1e-3)
        print(
            f"{r['name']:24}  {r['n']:>7} {r['k']:>3} {r['metric']:>6}  "
            f"{r['set_overlap']:>11.4%}  {r['exact_match']:>9.4%}  "
            f"{r['perfect_rows']:>10,}  {speedup:>7.1f}x"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
