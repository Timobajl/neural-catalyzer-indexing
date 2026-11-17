"""
IVFFlat benchmark.

Builds a FAISS IndexIVFFlat (random-sample k-means quantizer + flat inverted
lists), sweeps nprobe to produce a recall vs. distance-computations curve,
and reports fractional k-NN consistency on the resulting partition.
"""

import faiss
import numpy as np
import time
import argparse
import pandas as pd
import os
from math import sqrt


def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    n = len(I)
    correct = 0
    for i in range(n):
        correct += np.isin(I[i, :k], gt[i, :k]).sum()
    return round(correct / (n * k), 4)


def metric_name_to_faiss(metric_name: str) -> int:
    if metric_name == "l2":
        return faiss.METRIC_L2
    if metric_name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric_name}")


def check_index_metric(index: faiss.Index, metric_name: str, path: str) -> None:
    expected = metric_name_to_faiss(metric_name)
    actual = getattr(index, "metric_type", None)
    if actual is not None and actual != expected:
        raise ValueError(
            f"Index at {path} uses FAISS metric {actual}, "
            f"but --metric {metric_name} expects {expected}."
        )


def make_ivfflat_index(d: int, nlist: int, metric_name: str) -> faiss.IndexIVFFlat:
    metric_type = metric_name_to_faiss(metric_name)
    if metric_name == "ip":
        quantizer = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatL2(d)

    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)
    if metric_name == "ip" and hasattr(index, "cp"):
        index.cp.spherical = True
    return index


def read_assignments(index: faiss.IndexIVFFlat, n: int, nlist: int) -> np.ndarray:
    """Recover the cluster ID of every base vector from the inverted lists."""
    assignments = np.full(n, -1, dtype=np.int32)
    for list_id in range(nlist):
        size = index.invlists.list_size(list_id)
        if size == 0:
            continue
        ptr = index.invlists.get_ids(list_id)
        ids = faiss.rev_swig_ptr(ptr, size).copy()
        index.invlists.release_ids(list_id, ptr)
        assignments[ids] = list_id
    if np.any(assignments < 0):
        missing = int(np.sum(assignments < 0))
        raise ValueError(f"IVF index is missing assignments for {missing} vectors.")
    return assignments


def _take_best(D: np.ndarray, I: np.ndarray, k: int, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    keep_count = min(k, D.shape[1])
    if largest:
        keep = np.argpartition(-D, keep_count - 1, axis=1)[:, :keep_count]
        keep_D = np.take_along_axis(D, keep, axis=1)
        keep_I = np.take_along_axis(I, keep, axis=1)
        order = np.argsort(-keep_D, axis=1)
    else:
        keep = np.argpartition(D, keep_count - 1, axis=1)[:, :keep_count]
        keep_D = np.take_along_axis(D, keep, axis=1)
        keep_I = np.take_along_axis(I, keep, axis=1)
        order = np.argsort(keep_D, axis=1)
    return np.take_along_axis(keep_D, order, axis=1), np.take_along_axis(keep_I, order, axis=1)


def exact_knn_chunked(
    xb_mmap: np.ndarray,
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    k: int,
    chunk_size: int,
    metric_name: str,
) -> np.ndarray:
    """Chunked exact k-NN. Result is shape (n_q, k); -1 fills unfilled slots."""
    n = xb_mmap.shape[0]
    n_queries = query_vectors.shape[0]
    keep_k = min(k + 1, n)
    metric = metric_name_to_faiss(metric_name)
    largest = metric_name == "ip"
    fill = float("-inf") if largest else float("inf")
    best_D = np.full((n_queries, keep_k), fill, dtype=np.float32)
    best_I = np.full((n_queries, keep_k), -1, dtype=np.int64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = np.ascontiguousarray(xb_mmap[start:stop].astype("float32"))
        search_k = min(keep_k, stop - start)
        D_chunk, I_chunk = faiss.knn(query_vectors, chunk, search_k, metric=metric)
        I_chunk = I_chunk.astype(np.int64) + start

        merged_D = np.concatenate([best_D, D_chunk], axis=1)
        merged_I = np.concatenate([best_I, I_chunk], axis=1)
        best_D, best_I = _take_best(merged_D, merged_I, keep_k, largest)
        del chunk

    true_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
    for row_idx, self_id in enumerate(query_ids):
        row = best_I[row_idx]
        row = row[(row >= 0) & (row != int(self_id))]
        true_neighbors[row_idx, : min(k, len(row))] = row[:k]
    return true_neighbors


def load_consistency_cache(
    path: str, metric_name: str, num_samples: int, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load (sample_indices, true_neighbors) from a precomputed cache and
    validate metric/num_samples/k/seed against the current run.
    """
    cache = np.load(path)
    cached_metric = str(cache["metric"])
    cached_num_samples = int(cache["num_samples"])
    cached_k = int(cache["k"])
    cached_seed = int(cache["seed"])
    if cached_metric != metric_name:
        raise ValueError(
            f"Cache metric {cached_metric!r} != --metric {metric_name!r}"
        )
    if cached_num_samples != num_samples:
        raise ValueError(
            f"Cache num_samples {cached_num_samples} != --consistency-samples {num_samples}"
        )
    if cached_k != k:
        raise ValueError(f"Cache k {cached_k} != --consistency-k {k}")
    if cached_seed != seed:
        raise ValueError(f"Cache seed {cached_seed} != --seed {seed}")

    return cache["sample_indices"], cache["true_neighbors"]


def compute_knn_consistency(
    xb_mmap: np.ndarray,
    assignments: np.ndarray,
    n: int,
    nlist: int,
    k: int,
    num_samples: int,
    chunk_size: int,
    metric_name: str,
    seed: int,
    xb_orig: np.ndarray = None,
    cache_path: str = None,
) -> dict:
    """Fractional k-NN consistency: fraction of each sample's true k nearest
    neighbors that land in the sample's own cluster.

    Search is performed over `xb_orig` when supplied so the metric measures
    preservation of the original-space neighborhood structure recall is
    evaluated against. When `cache_path` is given, sample_indices and
    true_neighbors are loaded from disk and the inline kNN search is skipped.
    """
    if cache_path is not None:
        print(f"\n  [knn-consistency] loading cache from {cache_path}")
        sample_indices, true_neighbors = load_consistency_cache(
            cache_path, metric_name, num_samples, k, seed
        )
        num_samples = len(sample_indices)
        print(f"  [knn-consistency] cache loaded (samples={num_samples:,}, k={k})")
    else:
        num_samples = min(num_samples, n)
        knn_source = xb_orig if xb_orig is not None else xb_mmap
        knn_space = "original" if xb_orig is not None else "self"
        print(f"\n  [knn-consistency] samples={num_samples:,} k={k}  knn_space={knn_space}")

        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(n, size=num_samples, replace=False)
        xb_sample = knn_source[sample_indices].astype("float32")

        t0 = time.time()
        true_neighbors = exact_knn_chunked(
            xb_mmap=knn_source,
            query_vectors=xb_sample,
            query_ids=sample_indices,
            k=k,
            chunk_size=chunk_size,
            metric_name=metric_name,
        )
        print(f"  [knn-consistency] Exact search done in {time.time() - t0:.1f}s")

    sample_clusters = assignments[sample_indices]
    valid = true_neighbors >= 0
    neighbor_clusters = np.where(
        valid,
        assignments[np.where(valid, true_neighbors, 0)],
        -1,
    )
    matches = (neighbor_clusters == sample_clusters[:, None]) & valid
    per_sample = matches.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)

    avg  = float(np.mean(per_sample))
    std  = float(np.std(per_sample))
    perf = float(np.mean(per_sample == 1.0))
    zero = float(np.mean(per_sample == 0.0))
    print(
        f"  [knn-consistency] {avg:.4f} +/- {std:.4f}"
        f"  (perfect={perf:.3f}, zero={zero:.3f})"
    )

    bucket_sizes  = np.bincount(assignments, minlength=nlist).astype(np.float64)
    expected_size = n / nlist
    return {
        "knn_consistency_mean": avg,
        "knn_consistency_std":  std,
        "perfect_consistency":  perf,
        "zero_consistency":     zero,
        "bucket_mean":          float(np.mean(bucket_sizes)),
        "bucket_std":           float(np.std(bucket_sizes)),
        "bucket_min":           int(np.min(bucket_sizes)),
        "bucket_max":           int(np.max(bucket_sizes)),
        "bucket_imbalance":     float(np.max(bucket_sizes) / expected_size),
        "empty_buckets":        int(np.sum(bucket_sizes == 0)),
    }


def quantize_only(
    index: faiss.IndexIVFFlat,
    xb_mmap: np.ndarray,
    n: int,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    """Compute bucket assignments via the trained quantizer without filling
    the inverted lists. Suitable for --consistency-only runs where the
    full index.add() storage (about 314 GB for LAION 100M 768D) won't fit.

    The result is bit-identical to `read_assignments` after `index.add()`,
    because `index.add` internally uses the same quantizer.search call to
    assign each vector to its nearest centroid.
    """
    assignments = np.empty(n, dtype=np.int32)
    print(f"  Quantizing {n:,} vectors in chunks of {chunk_size:,} (no inverted lists)...")
    n_chunks = (n + chunk_size - 1) // chunk_size
    t0 = time.time()
    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        chunk = np.ascontiguousarray(xb_mmap[start:stop].astype("float32"))
        _, I = index.quantizer.search(chunk, 1)
        assignments[start:stop] = I[:, 0]
        elapsed = time.time() - t0
        speed = stop / max(elapsed, 1e-6)
        eta = (n - stop) / speed
        print(
            f"  Chunk {chunk_idx}/{n_chunks} | {stop:,}/{n:,} vectors"
            f" | {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
            end="\r",
        )
    elapsed = time.time() - t0
    print(f"\n  Quantize-only done in {elapsed:.1f}s")
    return assignments, elapsed


def train_and_add(
    index: faiss.IndexIVFFlat,
    xb_mmap: np.ndarray,
    n: int,
    train_size: int,
    chunk_size: int,
) -> tuple[float, float]:
    train_idx = np.random.choice(n, train_size, replace=False)
    train_idx.sort()
    X_train = xb_mmap[train_idx].astype("float32")

    print(f"  Training on {train_size:,} vectors...")
    t0 = time.time()
    index.train(X_train)
    del X_train
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    print(f"  Adding {n:,} vectors in chunks of {chunk_size:,}...")
    n_chunks = (n + chunk_size - 1) // chunk_size
    t0 = time.time()
    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        index.add(xb_mmap[start:stop].astype("float32"))
        elapsed = time.time() - t0
        speed = stop / max(elapsed, 1e-6)
        eta = (n - stop) / speed
        print(
            f"  Chunk {chunk_idx}/{n_chunks} | {stop:,}/{n:,} vectors"
            f" | {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
            end="\r",
        )
    add_time = time.time() - t0
    print(f"\n  Add done in {add_time:.1f}s")
    return train_time, add_time


def index_path_for(index_dir: str, label: str, metric_name: str, nlist: int) -> str:
    return os.path.join(index_dir, f"ivf_{label}_{metric_name}_nlist{nlist}.faiss")


def save_index(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)
    size_gb = os.path.getsize(path) / 1024 ** 3
    print(f"  Saved index to {path} ({size_gb:.1f} GB)")


def load_index(path: str) -> faiss.Index:
    size_gb = os.path.getsize(path) / 1024 ** 3
    print(f"  Loading index from {path} ({size_gb:.1f} GB)...")
    t0 = time.time()
    index = faiss.read_index(path)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVFFlat benchmark.")

    parser.add_argument("--xb", required=True)
    parser.add_argument(
        "--xb-orig", default=None,
        help="Original-space vectors. Used by kNN consistency on transformed-space runs.",
    )
    parser.add_argument("--xq")
    parser.add_argument("--gt")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="ivfflat_results")
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")

    parser.add_argument("--mode", choices=["build", "search", "both"], default="both")

    parser.add_argument(
        "--nlist-values", type=str, default="",
        help="Comma-separated nlist values. Default: sqrt(n), 2*sqrt(n), 4*sqrt(n).",
    )
    parser.add_argument("--nprobe-values", type=str, default="1,2,4,8,16,32,64,128")

    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--train-sample-size", type=int, default=0,
        help="0 = auto: min(n, 100*nlist, 10M).",
    )

    parser.add_argument("--consistency-samples", type=int, default=10_000)
    parser.add_argument("--consistency-k", type=int, default=10)
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument(
        "--consistency-only", action="store_true",
        help="Compute kNN consistency and skip the nprobe sweep.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--consistency-cache", default=None,
        help="Path to precomputed kNN-consistency cache (.npz from "
             "compute_true_neighbors.py). When set, skips inline kNN.",
    )

    parser.add_argument("--no-save-index", action="store_true")
    parser.add_argument("--index-dir", default="saved_indexes")
    parser.add_argument("--force-rebuild", action="store_true")

    args = parser.parse_args()
    args.save_index = not args.no_save_index
    np.random.seed(args.seed)

    if args.mode in ("search", "both") and (not args.xq or not args.gt):
        parser.error("--xq and --gt are required for --mode search/both")

    print(f"\n[{args.label}] Loading data...")
    xb_mmap = np.load(args.xb, mmap_mode="r")
    n, d = xb_mmap.shape
    print(f"  xb: {n:,} x {d}  ({n * d * 4 / 1024**3:.1f} GB as float32)")
    xb_orig_mmap = np.load(args.xb_orig, mmap_mode="r") if args.xb_orig else None
    if xb_orig_mmap is not None:
        if xb_orig_mmap.shape[0] != xb_mmap.shape[0]:
            parser.error(
                f"--xb-orig row count {xb_orig_mmap.shape[0]:,} does not match "
                f"--xb row count {xb_mmap.shape[0]:,}; row IDs must align."
            )
        print(f"  xb_orig: {xb_orig_mmap.shape[0]:,} x {xb_orig_mmap.shape[1]} dtype={xb_orig_mmap.dtype}")

    xq, gt = None, None
    nq = 0
    if args.mode in ("search", "both"):
        xq = np.load(args.xq).astype("float32")
        gt = np.load(args.gt)
        nq = xq.shape[0]
        if xq.shape[1] != d:
            parser.error(f"xq dimensionality {xq.shape[1]} does not match xb dimensionality {d}")
        if gt.shape[1] < 30:
            parser.error(f"gt must contain at least 30 neighbors per query, got shape {gt.shape}")
        print(f"  xq: {nq:,} x {xq.shape[1]}")
        print(f"  gt: {gt.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    if args.nlist_values:
        nlist_values = [int(v) for v in args.nlist_values.split(",")]
    else:
        base = int(sqrt(n))
        nlist_values = [base, base * 2, base * 4]
        print(f"  Auto nlist values: {nlist_values}  (sqrt of {n:,})")

    nprobe_values = [int(v) for v in args.nprobe_values.split(",")]
    metric_type = metric_name_to_faiss(args.metric)
    print(f"  metric: {args.metric} (FAISS metric_type={metric_type})")

    search_results      = []
    consistency_results = []

    for nlist in nlist_values:
        print(f"\n{'='*60}")
        print(f"nlist={nlist} | metric={args.metric} | mode={args.mode} | label={args.label}")
        print(f"{'='*60}")

        ipath = index_path_for(args.index_dir, args.label, args.metric, nlist)
        train_time = add_time = float("nan")
        index = None
        prebuilt_assignments = None

        if args.mode in ("build", "both"):
            if os.path.exists(ipath) and not args.force_rebuild:
                print(f"  Index already exists at {ipath}, skipping build.")
            else:
                index = make_ivfflat_index(d, nlist, args.metric)
                train_size = (
                    args.train_sample_size if args.train_sample_size > 0
                    else min(n, 100 * nlist, 10_000_000)
                )
                if args.consistency_only:
                    train_idx = np.random.choice(n, train_size, replace=False)
                    train_idx.sort()
                    X_train = xb_mmap[train_idx].astype("float32")
                    print(f"  Training quantizer on {train_size:,} vectors...")
                    t0 = time.time()
                    index.train(X_train)
                    del X_train
                    train_time = time.time() - t0
                    print(f"  Training done in {train_time:.1f}s")
                    prebuilt_assignments, add_time = quantize_only(
                        index, xb_mmap, n, args.chunk_size
                    )
                else:
                    train_time, add_time = train_and_add(
                        index, xb_mmap, n, train_size, args.chunk_size
                    )
                    if args.save_index:
                        save_index(index, ipath)

        if args.mode in ("search", "both"):
            if index is None:
                if not os.path.exists(ipath):
                    print(f"  ERROR: No index at {ipath}. Run --mode build first.")
                    continue
                index = load_index(ipath)
                check_index_metric(index, args.metric, ipath)

            if not args.skip_consistency:
                if prebuilt_assignments is not None:
                    assignments = prebuilt_assignments
                else:
                    assignments = read_assignments(index, n, nlist)
                stats = compute_knn_consistency(
                    xb_mmap=xb_mmap,
                    assignments=assignments,
                    n=n, nlist=nlist,
                    k=args.consistency_k,
                    num_samples=args.consistency_samples,
                    chunk_size=args.chunk_size,
                    metric_name=args.metric,
                    seed=args.seed,
                    xb_orig=xb_orig_mmap,
                    cache_path=args.consistency_cache,
                )
                consistency_results.append({
                    "space":           args.label,
                    "metric":          args.metric,
                    "dim":             d,
                    "n":               n,
                    "nlist":           nlist,
                    "k":               args.consistency_k,
                    "num_samples":     args.consistency_samples,
                    **stats,
                })
                cons_path = os.path.join(
                    args.output_dir, f"knn_consistency_{args.label}.csv"
                )
                pd.DataFrame(consistency_results).to_csv(cons_path, index=False)
                print(f"  Consistency flushed to {cons_path}")

            if args.consistency_only:
                continue

            print(f"\n  Sweeping nprobe: {nprobe_values}")
            for nprobe in nprobe_values:
                if nprobe > nlist:
                    print(f"  Skipping nprobe={nprobe} (> nlist={nlist})")
                    continue

                index.nprobe = nprobe
                faiss.cvar.indexIVF_stats.reset()

                t0 = time.time()
                _, I = index.search(xq, 30)
                search_time = time.time() - t0

                avg_dist_comp = faiss.cvar.indexIVF_stats.ndis / nq

                r1  = get_recall(I, gt, 1)
                r5  = get_recall(I, gt, 5)
                r10 = get_recall(I, gt, 10)
                r30 = get_recall(I, gt, 30)

                print(
                    f"  nprobe={nprobe:4d} | R@1={r1:.4f} R@10={r10:.4f} R@30={r30:.4f}"
                    f" | dist_comp={avg_dist_comp:.0f} | {search_time:.3f}s"
                )

                search_results.append({
                    "space":             args.label,
                    "metric":            args.metric,
                    "dim":               d,
                    "nlist":             nlist,
                    "nprobe":            nprobe,
                    "recall@1":          r1,
                    "recall@5":          r5,
                    "recall@10":         r10,
                    "recall@30":         r30,
                    "dist_computations": avg_dist_comp,
                    "train_time":        train_time,
                    "add_time":          add_time,
                    "search_time":       search_time,
                })

            out_path = os.path.join(args.output_dir, f"ivf_results_{args.label}.csv")
            pd.DataFrame(search_results).to_csv(out_path, index=False)
            print(f"  Results flushed to {out_path}")

    print("\nDone.")
