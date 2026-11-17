"""
Data loading, k-NN and evaluation helpers of np_index.py.
"""

import numpy as np
import torch
import os
import time
import faiss
import h5py

import utils


def load_mmap(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    print(f"  mmap {path}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def sample_subset(xb_mmap: np.ndarray, n_train: int, seed: int = 42) -> np.ndarray:
    """Draw n_train random rows from a mmap array.

    Indices are sorted before indexing so mmap reads are sequential.
    """
    n = xb_mmap.shape[0]
    if n_train >= n:
        print(f"  n_train={n_train:,} >= n={n:,}, using full dataset")
        return xb_mmap[:].astype("float32")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_train, replace=False)
    idx.sort()
    t0 = time.time()
    X = xb_mmap[idx].astype("float32")
    print(f"  Sampled {n_train:,} / {n:,} vectors in {time.time()-t0:.1f}s "
          f"({X.nbytes/1024**3:.2f} GB RAM)")
    return X


def metric_name_to_faiss(metric: str) -> int:
    if metric == "l2":
        return faiss.METRIC_L2
    if metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric}")


def build_knn_exact(
    X_f32: np.ndarray,
    k: int,
    metric: str = "l2",
    chunk_size: int = 50_000,
) -> torch.Tensor:
    """Exact k-NN via a FAISS flat index, without the point itself."""
    X = np.ascontiguousarray(X_f32.astype("float32", copy=False))
    n, d = X.shape
    k = min(k, n - 1)
    search_k = min(k + 1, n)

    use_gpu = utils.primary_device.type == "cuda"
    print(f"  Building exact kNN: {n:,} x {d}D, k={k}, metric={metric}, "
          f"backend={'faiss-gpu' if use_gpu else 'faiss-cpu'}")
    t0 = time.time()
    index = faiss.IndexFlat(d, metric_name_to_faiss(metric))
    if use_gpu:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(X)

    Y = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        _, I = index.search(X[start:stop], search_k)
        for row, global_id in enumerate(range(start, stop)):
            filtered = I[row][I[row] != global_id]
            if len(filtered) < k:
                filtered = np.pad(filtered, (0, k - len(filtered)), constant_values=global_id)
            Y[global_id] = filtered[:k]

    print(f"  kNN done in {time.time()-t0:.0f}s")
    return torch.from_numpy(Y).long()


def save_knn(Y: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = Y.cpu().numpy()
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=arr)
    print(f"  Saved kNN → {path}  ({os.path.getsize(path)/1024**2:.0f} MB)")


def load_knn(path: str, device: str = "cpu") -> torch.Tensor:
    with h5py.File(path, "r") as f:
        Y = torch.tensor(f["train"][:], dtype=torch.long, device=device)
    print(f"  Loaded kNN from {path}: shape={Y.shape}")
    return Y


def index_full_dataset(
    model_forest,
    xb_mmap: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Assign every base vector to a bin using the trained model."""
    n, d = xb_mmap.shape
    bin_assignments = np.empty(n, dtype=np.int32)

    print(f"  Indexing {n:,} vectors in chunks of {chunk_size:,}...")
    model_tree = model_forest.trees[0]
    model_tree.eval()
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            chunk = xb_mmap[start:stop].astype("float32")
            X_t = torch.from_numpy(chunk)
            bins, _ = model_tree.infer_new(X_t, batch_size=chunk_size, bin_count=1)
            bin_assignments[start:stop] = bins[:, 0].cpu().numpy().astype(np.int32)

            elapsed = time.time() - t0
            speed = stop / max(elapsed, 1e-6)
            eta = (n - stop) / max(speed, 1)
            print(f"\r  {stop:,}/{n:,}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s", end="")

    print(f"\n  Indexing done in {time.time()-t0:.1f}s")
    return bin_assignments


def build_lookup_table(bin_assignments: np.ndarray, n_bins: int) -> dict:
    """Invert bin_assignments into {bin_id: int32 array of global indices}.
    Single-pass O(n) accumulation
    """
    buckets = [[] for _ in range(n_bins)]
    for i, b in enumerate(bin_assignments):
        buckets[int(b)].append(i)
    return {b: np.array(v, dtype=np.int32) for b, v in enumerate(buckets)}


_RECALL_KS = (1, 5, 10, 30)


def compute_recall(
    lookup: dict,
    xb_mmap: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    model_tree,
    n_bins_to_search: int = 1,
    batch_size: int = 1024,
    metric: str = "l2",
) -> dict:
    """Recall@1/5/10/30 with brute-force re-ranking inside the candidate set.
    `inference_time` is the bin-routing forward pass, `search_time` is the
    brute-force rerank over the candidate set.
    """
    nq         = xq.shape[0]
    hits       = {k: 0 for k in _RECALL_KS}
    total_cand = 0
    max_k      = max(_RECALL_KS)
    t_inference = 0.0
    t_search    = 0.0

    model_tree.eval()
    with torch.no_grad():
        for start in range(0, nq, batch_size):
            stop  = min(start + batch_size, nq)
            Q_t   = torch.tensor(xq[start:stop].astype("float32"))

            _t0 = time.time()
            bins, _ = model_tree.infer_new(
                Q_t, batch_size=batch_size, bin_count=n_bins_to_search
            )
            t_inference += time.time() - _t0

            for i, qi in enumerate(range(start, stop)):
                cand_bins = bins[i, :n_bins_to_search].cpu().numpy()
                cand_idx  = np.concatenate([
                    lookup.get(int(b), np.array([], dtype=np.int32))
                    for b in cand_bins
                ])
                if len(cand_idx) == 0:
                    continue
                total_cand += len(cand_idx)

                _t0 = time.time()
                cand_vecs = xb_mmap[cand_idx].astype("float32")
                q_vec     = xq[qi : qi + 1].astype("float32")
                if metric == "l2":
                    scores = np.sum((cand_vecs - q_vec) ** 2, axis=1)
                    top_indices = np.argsort(scores)[:max_k]
                elif metric == "ip":
                    scores = cand_vecs @ q_vec[0]
                    top_indices = np.argsort(-scores)[:max_k]
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                # Slicing top-k from an ordered list, then comparing as sets.
                returned = cand_idx[top_indices].tolist()
                t_search += time.time() - _t0

                for k in _RECALL_KS:
                    gt_set   = set(gt[qi, :k].tolist())
                    returned_k = set(returned[:k])
                    hits[k]  += len(gt_set & returned_k)

    result = {
        f"recall@{k}": round(hits[k] / (nq * k), 4)
        for k in _RECALL_KS
    }
    result["dist_computations"] = round(total_cand / nq, 1)
    result["inference_time"] = round(t_inference, 4)
    result["search_time"]    = round(t_search, 4)
    result["total_time"]     = round(t_inference + t_search, 4)
    return result


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
    metric: str,
) -> np.ndarray:
    """Exact k-NN of `query_vectors` against the full xb_mmap, processed in chunks.

    Self-id of each query is excluded from the returned neighbors. Output is
    int64 shape (len(query_ids), k), filled with -1 if fewer than k results.
    """
    n = xb_mmap.shape[0]
    n_queries = query_vectors.shape[0]
    keep_k = min(k + 1, n)
    largest = metric == "ip"
    fill = float("-inf") if largest else float("inf")
    best_D = np.full((n_queries, keep_k), fill, dtype=np.float32)
    best_I = np.full((n_queries, keep_k), -1, dtype=np.int64)
    faiss_metric = metric_name_to_faiss(metric)

    print(f"  [knn-consistency] Exact chunked search "
          f"(k={k}, n={n:,}, chunk={chunk_size:,}, metric={metric})")
    t0 = time.time()
    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        chunk = np.ascontiguousarray(xb_mmap[start:stop].astype("float32"))
        search_k = min(keep_k, stop - start)
        D_chunk, I_chunk = faiss.knn(query_vectors, chunk, search_k, metric=faiss_metric)
        I_chunk = I_chunk.astype(np.int64) + start

        merged_D = np.concatenate([best_D, D_chunk], axis=1)
        merged_I = np.concatenate([best_I, I_chunk], axis=1)
        best_D, best_I = _take_best(merged_D, merged_I, keep_k, largest)

        elapsed = time.time() - t0
        eta = (n - stop) / max(stop / max(elapsed, 1e-6), 1e-6)
        print(
            f"\r  [knn-consistency] chunk {chunk_idx} | {stop:,}/{n:,}"
            f" | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            end="",
        )
        del chunk
    print(f"\n  [knn-consistency] Exact search done in {time.time() - t0:.1f}s")

    true_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
    for row_idx, self_id in enumerate(query_ids):
        row = best_I[row_idx]
        row = row[(row >= 0) & (row != int(self_id))]
        true_neighbors[row_idx, : min(k, len(row))] = row[:k]
    return true_neighbors


def load_consistency_cache(
    path: str, metric: str, num_samples: int, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load (sample_indices, true_neighbors) from a precomputed cache and
    validate metric/num_samples/k/seed against the current run.
    """
    cache = np.load(path)
    cached_metric = str(cache["metric"])
    cached_num_samples = int(cache["num_samples"])
    cached_k = int(cache["k"])
    cached_seed = int(cache["seed"])
    if cached_metric != metric:
        raise ValueError(f"Cache metric {cached_metric!r} != --metric {metric!r}")
    if cached_num_samples != num_samples:
        raise ValueError(
            f"Cache num_samples {cached_num_samples} != --consistency-samples {num_samples}"
        )
    if cached_k != k:
        raise ValueError(f"Cache k {cached_k} != --consistency-k {k}")
    if cached_seed != seed:
        raise ValueError(f"Cache seed {cached_seed} != --seed {seed}")
    return cache["sample_indices"], cache["true_neighbors"]


def compute_knn_consistency_full_db(
    bin_assignments: np.ndarray,
    xb_mmap: np.ndarray,
    k: int,
    num_samples: int,
    chunk_size: int,
    metric: str,
    seed: int = 42,
    cache_path: str = None,
) -> dict:
    """Fractional kNN consistency over the full database.

    For `num_samples` rows drawn from xb_mmap, find their exact k neighbors
    in the full database via chunked exact search and return the fraction that
    fall in the same bucket as the sample. When `cache_path` is given,
    sample_ids and true_neighbors are loaded from the cache instead.
    """
    n = len(bin_assignments)
    assert xb_mmap.shape[0] == n, (
        f"bin_assignments and xb_mmap must cover the same n "
        f"(got {n} vs {xb_mmap.shape[0]})"
    )

    if cache_path is not None:
        print(f"\n  [knn-consistency] loading cache from {cache_path}")
        sample_ids, true_neighbors = load_consistency_cache(
            cache_path, metric, num_samples, k, seed
        )
        num_samples = len(sample_ids)
        print(f"  [knn-consistency] cache loaded (samples={num_samples:,}, k={k})")
    else:
        num_samples = min(num_samples, n)
        rng = np.random.default_rng(seed)
        sample_ids = rng.choice(n, size=num_samples, replace=False).astype(np.int64)
        sample_vecs = np.ascontiguousarray(xb_mmap[sample_ids].astype("float32"))
        print(f"\n  [knn-consistency] samples={num_samples:,} k={k} (full-DB, n={n:,})")

        true_neighbors = exact_knn_chunked(
            xb_mmap=xb_mmap,
            query_vectors=sample_vecs,
            query_ids=sample_ids,
            k=k,
            chunk_size=chunk_size,
            metric=metric,
        )

    assign_arr     = np.asarray(bin_assignments)
    sample_buckets = assign_arr[sample_ids]
    valid          = true_neighbors >= 0
    # np.where guards against indexing with -1 when fewer than k valid neighbors.
    neighbor_buckets = np.where(valid, assign_arr[np.where(valid, true_neighbors, 0)], -1)
    matches    = (neighbor_buckets == sample_buckets[:, None]) & valid
    per_sample = matches.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)

    stats = {
        "knn_consistency_mean": round(float(np.mean(per_sample)), 4),
        "knn_consistency_std":  round(float(np.std(per_sample)),  4),
        "knn_consistency_p50":  round(float(np.median(per_sample)), 4),
        "knn_consistency_p90":  round(float(np.percentile(per_sample, 90)), 4),
        "perfect_consistency":  round(float(np.mean(per_sample == 1.0)), 4),
        "zero_consistency":     round(float(np.mean(per_sample == 0.0)), 4),
    }
    print(
        f"  [knn-consistency] {stats['knn_consistency_mean']:.4f}"
        f" +/- {stats['knn_consistency_std']:.4f}"
        f"  (perfect={stats['perfect_consistency']:.3f},"
        f" zero={stats['zero_consistency']:.3f})"
    )
    return stats
