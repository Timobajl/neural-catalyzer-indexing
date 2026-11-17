"""
Precompute true k-NN for kNN-consistency cache.

The fractional kNN-consistency metric used across IVFFlat, LMI, NLSH and
Neural Partitioner samples the same row IDs (seed=42) and looks up exact
k-NN in the original-space vectors. That kNN computation is identical
across every (nlist / n_buckets / n_clusters / n_bins) configuration, so
compute it once and cache it.

Output is a .npz with sample_indices, true_neighbors and metadata.
"""

import argparse
import os
import time

import faiss
import numpy as np


def metric_name_to_faiss(metric_name: str) -> int:
    if metric_name == "l2":
        return faiss.METRIC_L2
    if metric_name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric_name}")


def _take_best(
    D: np.ndarray, I: np.ndarray, k: int, largest: bool
) -> tuple[np.ndarray, np.ndarray]:
    keep_count = min(k, D.shape[1])
    if largest:
        keep = np.argpartition(-D, keep_count - 1, axis=1)[:, :keep_count]
    else:
        keep = np.argpartition(D, keep_count - 1, axis=1)[:, :keep_count]
    keep_D = np.take_along_axis(D, keep, axis=1)
    keep_I = np.take_along_axis(I, keep, axis=1)
    order = np.argsort(-keep_D if largest else keep_D, axis=1)
    return (
        np.take_along_axis(keep_D, order, axis=1),
        np.take_along_axis(keep_I, order, axis=1),
    )


def compute_true_neighbors(
    xb_mmap: np.ndarray,
    sample_indices: np.ndarray,
    k: int,
    chunk_size: int,
    metric_name: str,
) -> np.ndarray:
    n = xb_mmap.shape[0]
    n_queries = len(sample_indices)
    keep_k = min(k + 1, n)
    largest = metric_name == "ip"
    fill = float("-inf") if largest else float("inf")
    best_D = np.full((n_queries, keep_k), fill, dtype=np.float32)
    best_I = np.full((n_queries, keep_k), -1, dtype=np.int64)
    metric = metric_name_to_faiss(metric_name)

    query_vectors = np.ascontiguousarray(xb_mmap[sample_indices].astype("float32"))

    print(
        f"  Exact chunked search: n={n:,}, samples={n_queries:,}, k={k}, "
        f"chunk={chunk_size:,}, metric={metric_name}"
    )
    t0 = time.time()
    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        chunk = np.ascontiguousarray(xb_mmap[start:stop].astype("float32"))
        search_k = min(keep_k, stop - start)
        D_chunk, I_chunk = faiss.knn(query_vectors, chunk, search_k, metric=metric)
        I_chunk = I_chunk.astype(np.int64) + start

        merged_D = np.concatenate([best_D, D_chunk], axis=1)
        merged_I = np.concatenate([best_I, I_chunk], axis=1)
        best_D, best_I = _take_best(merged_D, merged_I, keep_k, largest)

        elapsed = time.time() - t0
        speed = stop / max(elapsed, 1e-6)
        eta = (n - stop) / max(speed, 1e-6)
        print(
            f"  chunk {chunk_idx} | {stop:,}/{n:,}"
            f" | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            end="\r",
        )
        del chunk
    print(f"\n  Exact search done in {time.time() - t0:.1f}s")

    true_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
    for row_idx, self_id in enumerate(sample_indices):
        row = best_I[row_idx]
        row = row[(row >= 0) & (row != int(self_id))]
        true_neighbors[row_idx, : min(k, len(row))] = row[:k]
    return true_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute true k-NN for kNN-consistency cache."
    )
    parser.add_argument("--xb-orig", required=True, help="Original-space vectors (.npy).")
    parser.add_argument("--metric", choices=["l2", "ip"], required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--output", required=True, help="Output .npz cache path.")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    print(f"FAISS OMP threads: {faiss.omp_get_max_threads()}")

    print(f"\nLoading {args.xb_orig} ...")
    xb_mmap = np.load(args.xb_orig, mmap_mode="r")
    n, d = xb_mmap.shape
    print(f"  shape={xb_mmap.shape} dtype={xb_mmap.dtype}")
    print(
        f"  size on disk: {n * d * xb_mmap.dtype.itemsize / 1024**3:.1f} GB "
        f"(would be {n * d * 4 / 1024**3:.1f} GB as float32)"
    )

    if args.num_samples > n:
        parser.error(f"--num-samples {args.num_samples:,} > n={n:,}")
    if args.k >= n:
        parser.error(f"--k {args.k} must be < n={n:,}")

    print(f"\nSampling {args.num_samples:,} row IDs (seed={args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(n, size=args.num_samples, replace=False).astype(np.int64)

    true_neighbors = compute_true_neighbors(
        xb_mmap=xb_mmap,
        sample_indices=sample_indices,
        k=args.k,
        chunk_size=args.chunk_size,
        metric_name=args.metric,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        sample_indices=sample_indices,
        true_neighbors=true_neighbors,
        xb_orig_path=np.array(os.path.abspath(args.xb_orig)),
        n=np.array(n),
        d=np.array(d),
        metric=np.array(args.metric),
        num_samples=np.array(args.num_samples),
        k=np.array(args.k),
        seed=np.array(args.seed),
    )
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"\nSaved cache to {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
