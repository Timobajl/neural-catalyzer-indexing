#!/usr/bin/env python3
"""
Neural LSH thesis benchmark (Dong et al., ICLR 2020;
reference at https://github.com/twistedcubic/learn-to-hash).

Thin wrapper around the original training and partitioning code in
``kahip/kmkahip.py`` and ``model/train.py``. The changes relative to the
reference are in neural-lsh.patch and in the README.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import torch

import utils
from kahip.kmkahip import run_kmkahip
from model import train as model_train


SEED = 42
EMPTY_INT = np.empty((0,), dtype=np.int64)


def row_norm_stats(rows: np.ndarray, max_samples: int = 4096) -> dict[str, float]:
    sample_size = min(len(rows), max_samples)
    if sample_size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    sample = np.asarray(rows[:sample_size]).astype("float32", copy=False)
    norms = np.linalg.norm(sample, axis=1)
    return {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "max": float(np.max(norms)),
    }


def format_norm_stats(stats: dict[str, float]) -> str:
    return f"min={stats['min']:.4f} mean={stats['mean']:.4f} max={stats['max']:.4f}"


def to_float32_rows(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rows).astype("float32"))


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def metric_name_to_faiss(metric_name: str) -> int:
    if metric_name == "l2":
        return faiss.METRIC_L2
    if metric_name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric_name}")


def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    correct = 0
    for i in range(len(I)):
        correct += np.isin(I[i, :k], gt[i, :k]).sum()
    return round(correct / (len(I) * k), 4)


def rank_meta_path(rank_path: str) -> str:
    return rank_path + ".json"


def rank_artifact_matches(rank_path: str, expected: dict) -> bool:
    """True if the rank file and its metadata match this run, so it can be reused."""
    if not os.path.exists(rank_path) or not os.path.exists(rank_meta_path(rank_path)):
        return False
    try:
        with open(rank_meta_path(rank_path), "r") as f:
            meta = json.load(f)
        ranks = np.load(rank_path, mmap_mode="r")
    except Exception:
        return False
    return (
        ranks.shape == (expected["n"], expected["depth"])
        and str(ranks.dtype) == "int32"
        and all(meta.get(k) == v for k, v in expected.items())
    )


def make_flat_index(d: int, metric_name: str) -> faiss.Index:
    if metric_name == "ip":
        return faiss.IndexFlatIP(d)
    return faiss.IndexFlatL2(d)


def compute_exact_base_ranks(
    xb_mmap: np.ndarray,
    metric_name: str,
    depth: int,
    rank_path: str,
    chunk_size: int,
    force: bool,
) -> tuple[str, float, bool]:
    """
    Precompute exact base-base k-NN ranks once at depth
    ``max(k_graph, soft_label_neighbors, consistency_k)``. The same file feeds
    KaHIP graph construction, soft-label training, and kNN consistency.

    Ranks exclude the point itself, so ``--soft-label-neighbors N`` corresponds
    to N external neighbors and effective paper S = N+1 (DataNode appends self).
    """
    n, d = xb_mmap.shape
    depth = min(depth, n - 1)
    expected = {"n": n, "d": d, "depth": depth, "metric": metric_name}
    if not force and rank_artifact_matches(rank_path, expected):
        print(f"  Reusing exact ranks: {rank_path}")
        return rank_path, 0.0, True

    os.makedirs(os.path.dirname(rank_path) or ".", exist_ok=True)
    print(f"  Computing exact base-base ranks: n={n:,}, d={d}, depth={depth}, metric={metric_name}")

    index = make_flat_index(d, metric_name)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        index.add(to_float32_rows(xb_mmap[start:stop]))

    ranks = np.lib.format.open_memmap(rank_path, mode="w+", dtype="int32", shape=(n, depth))
    search_k = min(depth + 1, n)
    t0 = time.time()
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        queries = to_float32_rows(xb_mmap[start:stop])
        _, I = index.search(queries, search_k)
        for row, global_id in enumerate(range(start, stop)):
            filtered = I[row][I[row] != global_id]
            if len(filtered) < depth:
                filtered = np.pad(filtered, (0, depth - len(filtered)), constant_values=global_id)
            ranks[global_id] = filtered[:depth]
        elapsed = time.time() - t0
        speed = (stop - 0) / max(elapsed, 1e-6)
        eta = (n - stop) / max(speed, 1e-6)
        print(
            f"  ranks {stop:,}/{n:,} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            end="\r",
        )
    ranks.flush()
    with open(rank_meta_path(rank_path), "w") as f:
        json.dump(expected, f, indent=2)
    elapsed = time.time() - t0
    print(f"\n  Exact ranks written to {rank_path} in {elapsed:.1f}s")
    return rank_path, elapsed, False


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return device_arg


def configure_runtime(device: str) -> None:
    utils.device = device
    model_train.device = device


def model_path_for(
    model_dir: str,
    dataset_name: str,
    label: str,
    metric: str,
    height: int,
    n_clusters: int,
    soft_label_neighbors: int,
    bottom_level: str = "train",
    train_n: int = 0,
) -> str:
    train_tag = f"_t{train_n}" if train_n > 0 else ""
    bottom_tag = f"_b{bottom_level}"
    return os.path.join(
        model_dir,
        f"nlsh_mode_{dataset_name}_{label}_{metric}_h{height}_c{n_clusters}_s{soft_label_neighbors}{bottom_tag}{train_tag}.pkl",
    )


def rank_path_for(rank_dir: str, dataset_name: str, label: str, metric: str, depth: int) -> str:
    return os.path.join(rank_dir, f"nlsh_ranks_{dataset_name}_{label}_{metric}__k{depth}.npy")


def make_training_opt(
    args: argparse.Namespace,
    run_dir: str,
    model_path: str,
    rank_path: str,
    d: int,
    n_clusters: int,
) -> SimpleNamespace:
    """
    Build the ``opt`` namespace expected by the original ``run_kmkahip``.

    Fields added for this script: ``ranks_path`` (reusable exact ranks),
    ``model_path`` (deterministic artifact name), ``skip_initial_eval`` (this
    script does its own benchmark eval).
    """
    data_name = f"{args.dataset_name}_{args.label}"
    # Level numbering in kmkahip.py is leaves-first: level 0 = bottom, level
    # args.height-1 = top. --bottom-level km swaps the bottom NN for k-means
    level2action = {i: "train" for i in range(args.height)}
    if args.bottom_level == "km":
        level2action[0] = "km"
    return SimpleNamespace(
        n_clusters=n_clusters,
        n_class=n_clusters,
        kahip_config=args.kahip_config,
        parts_path_root=os.path.join(run_dir, "partition", ""),
        dsnode_path=os.path.join(run_dir, "train_dsnode"),
        k=args.k,
        k_graph=args.k_graph,
        level2action=level2action,
        subsample=1,
        nn_mult=5,
        soft_label_neighbors=args.soft_label_neighbors,
        graph_file="knn.graph",
        data_dir=run_dir,
        models_dir=os.path.dirname(model_path),
        model_path=model_path,
        model_data_name=data_name,
        ranks_path=rank_path,
        use_precomputed_soft_labels=True,
        skip_initial_eval=True,
        dataset_name=args.dataset_name,
        glove=False,
        sift=False,
        glove_c=False,
        sift_c=False,
        prefix10m=False,
        paper_architecture=True,
        fast_kmeans=False,
        itq=False,
        compute_gt_nn=False,
        write_res=False,
        normalize_data=False,
        metric_name=args.metric,
        n_input=d,
        n_hidden=args.n_hidden,
        n_epochs=args.n_epochs,
        lr=args.lr,
    )


def train_nlsh_model(args: argparse.Namespace, xb_mmap: np.ndarray, model_path: str, rank_path: str, run_dir: str, n_clusters: int, device: str) -> float:
    """Train or reuse one Neural LSH hierarchy for a fixed ``n_clusters``."""
    if os.path.exists(model_path) and not args.force_retrain:
        print(f"  Reusing model: {model_path}")
        return 0.0

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "graph"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "partition"), exist_ok=True)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    dataset = torch.from_numpy(np.array(xb_mmap, dtype=np.float32, copy=True)).to(device)
    queryset = torch.empty((0, xb_mmap.shape[1]), dtype=torch.float32, device=device)
    neighbors = torch.empty((0, args.k), dtype=torch.long, device=device)
    opt = make_training_opt(args, run_dir, model_path, rank_path, xb_mmap.shape[1], n_clusters)

    print(
        f"  Training Neural LSH: height={args.height}, n_clusters={n_clusters}, "
        f"soft-label external={args.soft_label_neighbors}, effective S={args.soft_label_neighbors + 1}"
    )
    t0 = time.time()
    run_kmkahip(args.height, opt, dataset, queryset, neighbors)
    elapsed = time.time() - t0
    print(f"  Neural LSH training/build done in {elapsed:.1f}s")
    del dataset
    if device == "cuda":
        torch.cuda.empty_cache()
    return elapsed


def load_eval_root(model_path: str, device: str):
    print(f"  Loading model: {model_path}")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    eval_root = payload["eval_root"]
    move_eval_tree(eval_root, device)
    return eval_root


def move_eval_tree(node, device: str) -> None:
    if getattr(node, "model", None) is not None:
        node.model.to(device)
        node.model.eval()
    # km leaves carry tensors built on utils.device at train time. Moving them
    # here allows training on GPU and search on CPU (or a different GPU) safely.
    km = getattr(node, "kmsolver", None)
    if km is not None:
        for attr in ("centers", "centers_norm", "codes"):
            t = getattr(km, attr, None)
            if hasattr(t, "to"):
                setattr(km, attr, t.to(device))
    for child in getattr(node, "children", []):
        move_eval_tree(child, device)


def tensor_leaf_ids(node) -> list[int]:
    """All final bucket IDs under ``node``."""
    if getattr(node, "children", []):
        ids = []
        for child in node.children:
            ids.extend(tensor_leaf_ids(child))
        return ids
    # Small-child leaves can carry the full n_class even with one real bucket,
    # so prefer the explicit leaf_idx when present.
    if getattr(node, "leaf_idx", None) is not None:
        return [int(v) for v in node.leaf_idx]
    if getattr(node, "base_idx", None) is not None:
        return [int(node.base_idx) + i for i in range(int(node.n_class))]
    raise ValueError("Eval tree leaf has no bucket ids.")


def dedupe_ordered(values: list[int]) -> list[int]:
    """Drop repeated bucket IDs and keep the order."""
    seen = set()
    out = []
    for value in values:
        bucket_id = int(value)
        if bucket_id not in seen:
            seen.add(bucket_id)
            out.append(bucket_id)
    return out


def collect_top_leaves_batch(node, queries: torch.Tensor, nprobe: int) -> list[list[int]]:
    """
    Return the buckets to scan for each query.

    At every level the model picks the ``nprobe`` most likely children and the
    query continues into each of them. At the last level it picks the ``nprobe``
    most likely buckets. With two levels, one query reaches up to ``nprobe ** 2``
    buckets. ``nprobe`` is the paper's T.
    """
    children = getattr(node, "children", [])
    batch = queries.shape[0]

    if not children:
        if getattr(node, "model", None) is not None:
            scores = node.model(queries)
            k = min(nprobe, scores.shape[1])
            _, top_pos = torch.topk(scores, k, dim=1, largest=True)
            base = int(node.base_idx)
            return [[base + int(top_pos[b, i]) for i in range(k)] for b in range(batch)]

        # km leaf (paper Section 4.1): route by k-means centroid
        # distance, not by returning every leaf ID under the subtree.
        kmsolver = getattr(node, "kmsolver", None)
        if kmsolver is not None:
            k = min(nprobe, int(node.n_class))
            idx = kmsolver.predict(queries, k=k)
            base = int(node.base_idx)
            k_actual = idx.shape[1]
            return [[base + int(idx[b, i]) for i in range(k_actual)] for b in range(batch)]

        ids = tensor_leaf_ids(node)
        return [list(ids) for _ in range(batch)]

    if getattr(node, "model", None) is None:
        raise ValueError("Hierarchical Neural LSH traversal requires neural models at internal nodes.")

    scores = node.model(queries)
    k_children = min(nprobe, scores.shape[1])
    _, top_child_pos = torch.topk(scores, k_children, dim=1, largest=True)

    per_query: list[list[int]] = [[] for _ in range(batch)]
    for child_rank in range(k_children):
        child_col = top_child_pos[:, child_rank]
        child_to_rows: dict[int, list[int]] = {}
        for row_idx, child_idx in enumerate(child_col.tolist()):
            if child_idx < len(children):
                child_to_rows.setdefault(int(child_idx), []).append(row_idx)

        for child_idx, rows in child_to_rows.items():
            child_node = children[child_idx]
            sub_queries = queries[rows]
            sub_leaves = collect_top_leaves_batch(child_node, sub_queries, nprobe)
            for local_idx, global_idx in enumerate(rows):
                per_query[global_idx].extend(sub_leaves[local_idx])

    return per_query


def predict_top_leaves(eval_root, xq: np.ndarray, nprobe: int, batch_size: int, device: str) -> tuple[list[list[int]], float]:
    """
    Ask the model which buckets to scan for each query.

    Queries go through the model in batches. Returns the bucket IDs per query and
    the time it took, which is the ``inference_time`` column of the results.
    """
    all_leaves: list[list[int]] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(xq), batch_size):
            stop = min(start + batch_size, len(xq))
            q_batch = torch.from_numpy(np.ascontiguousarray(xq[start:stop].astype("float32"))).to(device)
            batch_leaves = collect_top_leaves_batch(eval_root, q_batch, nprobe)
            all_leaves.extend(dedupe_ordered(leaves) for leaves in batch_leaves)
    inference_time = time.time() - t0
    return all_leaves, inference_time


def assignments_from_eval_root(eval_root, n: int) -> np.ndarray:
    """Bucket of every vector, read from the trained tree. Used when training covers all vectors."""
    assignments = np.full((n,), -1, dtype=np.int64)
    for idx, bucket in eval_root.idx2bin.items():
        bucket_id = int(bucket.item()) if hasattr(bucket, "item") else int(bucket)
        assignments[int(idx)] = bucket_id
    if np.any(assignments < 0):
        missing = int(np.sum(assignments < 0))
        raise ValueError(f"Eval root is missing bucket assignments for {missing} vectors.")
    return assignments


def build_bucket_index(assignments: np.ndarray, leaf_ids: list[int]) -> dict[int, np.ndarray]:
    leaf_to_indices = {}
    for bucket_id in leaf_ids:
        leaf_to_indices[int(bucket_id)] = np.where(assignments == bucket_id)[0].astype(np.int64)
    return leaf_to_indices


def nlsh_search(
    eval_root,
    xb_mmap: np.ndarray,
    xq: np.ndarray,
    nprobe: int,
    metric_name: str,
    batch_size: int,
    device: str,
    leaf_to_indices: dict[int, np.ndarray] | None = None,
    sorted_ids: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, float, int]:
    """
    Exact FAISS rescore of candidates from the predicted leaves, return top-30.

    Two lookup backends, exactly one must be provided:
    - ``leaf_to_indices`` dict for full-training runs (1M datasets).
    - ``sorted_ids`` + ``offsets`` memmapped index for ``--train-size`` runs
      (e.g. LAION 100M). ``offsets`` is indexed by raw leaf ID.
    """
    top_leaves, inference_time = predict_top_leaves(eval_root, xq, nprobe, batch_size, device)
    leaf_counts = [len(leaves) for leaves in top_leaves]
    leaf_buckets_visited_mean = float(np.mean(leaf_counts)) if leaf_counts else 0.0
    leaf_buckets_visited_max = int(max(leaf_counts)) if leaf_counts else 0
    metric = metric_name_to_faiss(metric_name)
    I = np.full((len(xq), 30), -1, dtype=np.int64)
    total_comps = 0
    use_sorted = sorted_ids is not None and offsets is not None

    t0 = time.time()
    for query_idx, leaves in enumerate(top_leaves):
        if use_sorted:
            parts = [
                sorted_ids[offsets[lid] : offsets[lid + 1]].astype(np.int64)
                for lid in leaves
                if lid < len(offsets) - 1 and offsets[lid] < offsets[lid + 1]
            ]
        else:
            parts = [leaf_to_indices.get(int(leaf), EMPTY_INT) for leaf in leaves]
        candidate_ids = np.concatenate(parts) if parts else EMPTY_INT
        if len(candidate_ids) == 0:
            continue
        total_comps += len(candidate_ids)
        candidates = to_float32_rows(xb_mmap[candidate_ids])
        query = np.ascontiguousarray(xq[query_idx : query_idx + 1].astype("float32"))
        search_k = min(30, len(candidate_ids))
        _, local_I = faiss.knn(query, candidates, search_k, metric=metric)
        I[query_idx, :search_k] = candidate_ids[local_I[0]]
    search_time = time.time() - t0
    avg_comps = total_comps / max(len(xq), 1)
    return (
        I,
        avg_comps,
        inference_time,
        search_time,
        leaf_buckets_visited_mean,
        leaf_buckets_visited_max,
    )


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


def compute_knn_consistency(assignments: np.ndarray, rank_path: str, k: int, num_samples: int, seed: int) -> dict:
    """Fractional kNN consistency from a precomputed rank file. Used on 1M runs without --xb-orig."""
    ranks = np.load(rank_path, mmap_mode="r")
    if ranks.shape[1] < k:
        raise ValueError(f"Rank artifact depth {ranks.shape[1]} is smaller than consistency k={k}.")
    n = len(assignments)
    num_samples = min(num_samples, n)
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(n, size=num_samples, replace=False)
    neighbors = ranks[sample_ids, :k]
    sample_buckets = assignments[sample_ids]
    neighbor_buckets = assignments[neighbors]
    matches = neighbor_buckets == sample_buckets[:, None]
    per_sample = matches.sum(axis=1) / k
    return {
        "knn_consistency_mean": float(np.mean(per_sample)),
        "knn_consistency_std": float(np.std(per_sample)),
        "perfect_consistency": float(np.mean(per_sample == 1.0)),
        "zero_consistency": float(np.mean(per_sample == 0.0)),
    }


# Train-subset indexing helpers (used when --train-size is set).
def index_artifact_paths(
    work_dir: str, label: str, metric: str, n: int,
    height: int, n_clusters: int, train_n: int, seed: int,
    bottom_level: str = "train",
) -> dict:
    """Paths of the saved bucket assignments of a ``--train-size`` run."""
    # train_n, seed, bottom_level in the path so any of them changing
    # invalidates the cache.
    base = os.path.join(
        work_dir, "index",
        f"{label}_{metric}_n{n}_t{train_n}_s{seed}_h{height}_c{n_clusters}_b{bottom_level}",
    )
    return {
        "assignments": base + "_assignments.npy",
        "sorted_ids": base + "_sorted_ids.npy",
        "offsets":    base + "_offsets.npy",
        "meta":       base + "_meta.json",
    }


def sample_train_ids(n: int, train_size: int, seed: int) -> np.ndarray:
    """Sample sorted train IDs from [0, n). Sorted so xb_mmap reads are sequential."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(n, size=min(train_size, n), replace=False).astype(np.int64)
    ids.sort()
    print(f"  Sampled {len(ids):,} train IDs (seed={seed})")
    return ids


def infer_all_assignments(
    eval_root,
    xb_mmap: np.ndarray,
    batch_size: int,
    device: str,
    path: str,
    force: bool,
) -> np.ndarray:
    """
    Bucket every base vector via top-1 NLSH traversal. ``nprobe=1`` matches
    how query vectors are routed at search time. Raw leaf IDs are kept
    unremapped so ``offsets[leaf_id]`` is consistent across infer and search.
    """
    if not force and os.path.exists(path):
        print(f"  Reusing full-database assignments: {path}")
        return np.load(path, mmap_mode="r")
    n = len(xb_mmap)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assignments = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(n,))
    print(f"  Inferring bucket assignments for {n:,} vectors ...")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = torch.from_numpy(
                np.ascontiguousarray(xb_mmap[start:stop].astype("float32"))
            ).to(device)
            leaves = collect_top_leaves_batch(eval_root, batch, nprobe=1)
            for i, leaf_list in enumerate(leaves):
                assignments[start + i] = int(leaf_list[0])
    assignments.flush()
    print(f"  Assignments done in {time.time() - t0:.1f}s")
    return np.load(path, mmap_mode="r")


def build_sorted_bucket_index(
    assignments: np.ndarray,
    leaf_ids: list[int],
    sorted_ids_path: str,
    offsets_path: str,
    meta_path: str,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sorted-IDs + offsets memmapped bucket index.
    ``sorted_ids[offsets[leaf_id] : offsets[leaf_id+1]]`` = IDs in that bucket.
    ``sorted_ids`` is memmapped, so search never loads it fully.
    """
    n = len(assignments)
    max_leaf_id = int(max(leaf_ids))
    meta_expected = {"n": n, "max_leaf_id": max_leaf_id}
    if not force and os.path.exists(sorted_ids_path) and os.path.exists(offsets_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if all(meta.get(k) == v for k, v in meta_expected.items()):
                print(f"  Reusing sorted bucket index")
                return np.load(sorted_ids_path, mmap_mode="r"), np.load(offsets_path)
        except Exception:
            pass

    os.makedirs(os.path.dirname(sorted_ids_path), exist_ok=True)
    print(f"  Building sorted bucket index for {n:,} vectors ...")
    t0 = time.time()

    counts = np.bincount(np.asarray(assignments).ravel(), minlength=max_leaf_id + 1).astype(np.int64)
    offsets = np.zeros(max_leaf_id + 2, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    sorted_order = np.argsort(np.asarray(assignments).ravel(), kind="stable").astype(np.int32)
    sorted_ids_mmap = np.lib.format.open_memmap(sorted_ids_path, mode="w+", dtype=np.int32, shape=(n,))
    sorted_ids_mmap[:] = sorted_order
    sorted_ids_mmap.flush()
    del sorted_order
    np.save(offsets_path, offsets)
    with open(meta_path, "w") as f:
        json.dump(meta_expected, f)
    print(f"  Sorted bucket index built in {time.time() - t0:.1f}s")
    return np.load(sorted_ids_path, mmap_mode="r"), offsets


def _take_best(D: np.ndarray, I: np.ndarray, k: int, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    if D.shape[1] <= k:
        return D, I
    ax = np.argpartition(-D if largest else D, k - 1, axis=1)[:, :k]
    D_k = np.take_along_axis(D, ax, axis=1)
    I_k = np.take_along_axis(I, ax, axis=1)
    s = np.argsort(-D_k if largest else D_k, axis=1)
    return np.take_along_axis(D_k, s, axis=1), np.take_along_axis(I_k, s, axis=1)


def exact_knn_chunked(
    xb_mmap: np.ndarray,
    sample_ids: np.ndarray,
    k: int,
    chunk_size: int,
    metric_name: str,
) -> np.ndarray:
    """
    Chunked exact k-NN for sample_ids over the full xb_mmap. Used for kNN
    consistency on ``--train-size`` runs (no full-DB rank file available).
    RAM peak: chunk_size * d * 4 + n_samples * chunk_size * 4 bytes.
    """
    n = len(xb_mmap)
    keep_k = min(k + 1, n)
    faiss_metric = metric_name_to_faiss(metric_name)
    largest = metric_name == "ip"
    fill = float("-inf") if largest else float("inf")
    best_D = np.full((len(sample_ids), keep_k), fill, dtype=np.float32)
    best_I = np.full((len(sample_ids), keep_k), -1, dtype=np.int64)
    sample_vecs = to_float32_rows(xb_mmap[sample_ids])

    print(f"  [knn-consistency] chunked exact search k={k}, n={n:,}, chunk={chunk_size:,} ...")
    t0 = time.time()
    for ci, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        chunk = to_float32_rows(xb_mmap[start:stop])
        D_c, I_c = faiss.knn(sample_vecs, chunk, min(keep_k, stop - start), metric=faiss_metric)
        best_D, best_I = _take_best(
            np.concatenate([best_D, D_c], axis=1),
            np.concatenate([best_I, I_c.astype(np.int64) + start], axis=1),
            keep_k, largest,
        )
        del chunk
        elapsed = time.time() - t0
        eta = (n - stop) / max(stop / max(elapsed, 1e-6), 1e-6)
        print(f"  [knn-consistency] chunk {ci} | {stop:,}/{n:,} | ETA {eta:.0f}s", end="\r")
    print(f"\n  [knn-consistency] exact search done in {time.time() - t0:.1f}s")

    true_neighbors = np.full((len(sample_ids), k), -1, dtype=np.int64)
    for row, self_id in enumerate(sample_ids):
        row_I = best_I[row]
        row_I = row_I[(row_I >= 0) & (row_I != self_id)]
        true_neighbors[row, : min(k, len(row_I))] = row_I[:k]
    return true_neighbors


def compute_knn_consistency_chunked(
    xb_mmap: np.ndarray,
    assignments: np.ndarray,
    k: int,
    num_samples: int,
    chunk_size: int,
    metric_name: str,
    seed: int,
    cache_path: str = None,
) -> dict:
    """Fractional kNN consistency via chunked exact search, for runs where the
    rank file cannot be used: --xb-orig or --train-size.
    When `cache_path` is given, sample_ids and true_neighbors are loaded from
    the cache and the inline chunked search is skipped.
    """
    n = len(xb_mmap)
    if cache_path is not None:
        print(f"\n  [knn-consistency] loading cache from {cache_path}")
        sample_ids, true_neighbors = load_consistency_cache(
            cache_path, metric_name, num_samples, k, seed
        )
        num_samples = len(sample_ids)
        print(f"  [knn-consistency] cache loaded (samples={num_samples:,}, k={k})")
    else:
        num_samples = min(num_samples, n)
        rng = np.random.default_rng(seed)
        sample_ids = rng.choice(n, size=num_samples, replace=False).astype(np.int64)
        print(f"\n  [knn-consistency] samples={num_samples:,} k={k} (chunked, full database)")
        true_neighbors = exact_knn_chunked(xb_mmap, sample_ids, k, chunk_size, metric_name)
    assign_arr = np.asarray(assignments)
    sample_buckets = assign_arr[sample_ids]
    valid = true_neighbors >= 0
    neighbor_buckets = np.where(valid, assign_arr[np.where(valid, true_neighbors, 0)], -1)
    matches = (neighbor_buckets == sample_buckets[:, None]) & valid
    per_sample = matches.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    stats = {
        "knn_consistency_mean": float(np.mean(per_sample)),
        "knn_consistency_std": float(np.std(per_sample)),
        "perfect_consistency": float(np.mean(per_sample == 1.0)),
        "zero_consistency": float(np.mean(per_sample == 0.0)),
    }
    print(
        f"  [knn-consistency] {stats['knn_consistency_mean']:.4f}"
        f" +/- {stats['knn_consistency_std']:.4f}"
        f"  (perfect={stats['perfect_consistency']:.3f}, zero={stats['zero_consistency']:.3f})"
    )
    return stats


def load_arrays(args: argparse.Namespace):
    xb = np.load(args.xb, mmap_mode="r")
    xq = gt = None
    if args.mode in ("search", "both"):
        xq = np.load(args.xq).astype("float32")
        gt = np.load(args.gt)
        if xq.shape[1] != xb.shape[1]:
            raise ValueError(f"xq dimensionality {xq.shape[1]} does not match xb dimensionality {xb.shape[1]}.")
        if gt.shape[1] < 30:
            raise ValueError(f"gt must contain at least 30 neighbors per query, got shape {gt.shape}.")
        args._xq_norm_stats = row_norm_stats(xq)
    return xb, xq, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural LSH benchmark for thesis experiments.")
    parser.add_argument("--xb", required=True, help="Base vectors .npy")
    parser.add_argument(
        "--xb-orig", default=None,
        help="Original-space vectors. Used by kNN consistency on transformed-space runs.",
    )
    parser.add_argument("--xq", help="Query vectors .npy")
    parser.add_argument("--gt", help="Ground-truth indices .npy")
    parser.add_argument("--dataset-name", required=True, help="Dataset family name used in artifact paths.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--mode", choices=["train", "search", "both"], default="both")
    parser.add_argument("--output-dir", default="nlsh_results")
    parser.add_argument("--work-dir", default="nlsh_work")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--rank-dir", default="")
    parser.add_argument("--n-clusters-values", default="32,45,64")
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument(
        "--nprobe-values",
        default="1,2,4,8,16,32,64,128,256",
        help="Comma-separated NLSH T values: top bins per hierarchy level.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-graph", type=int, default=10)
    parser.add_argument(
        "--soft-label-neighbors",
        type=int,
        default=15,
        help="External neighbors for soft labels. DataNode appends self, so effective paper S = N+1.",
    )
    parser.add_argument("--consistency-k", type=int, default=10)
    parser.add_argument("--consistency-samples", type=int, default=10_000)
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument(
        "--consistency-only", action="store_true",
        help="Compute kNN consistency and skip the nprobe sweep.",
    )
    parser.add_argument(
        "--consistency-cache", default=None,
        help="Path to precomputed kNN-consistency cache (.npz from "
             "compute_true_neighbors.py). Used only when consistency would "
             "otherwise run chunked exact search (--xb-orig or --train-size).",
    )
    parser.add_argument("--n-epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--n-hidden", type=int, default=512)
    parser.add_argument("--kahip-config", choices=["fast", "eco", "strong"], default="strong")
    parser.add_argument(
        "--bottom-level",
        choices=["train", "km"],
        default="train",
        help=(
            "Bottom-level classifier: 'train' = neural network, "
            "'km' = k-means (paper Section 4.1, for a small number of points "
            "per bin). Top level is always 'train'."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rank-chunk-size", type=int, default=50000)
    parser.add_argument("--search-batch-size", type=int, default=1024)
    parser.add_argument("--force-ranks", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help=(
            "Train k-NN graph, KaHIP, and neural networks on a random subset of this size. "
            "All database vectors are then bucketed via neural net inference."
        ),
    )
    parser.add_argument("--index-chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    if args.mode in ("search", "both") and (not args.xq or not args.gt):
        parser.error("--xq and --gt are required for --mode search/both")
    if args.height < 1:
        parser.error("--height must be >= 1")
    if args.k_graph < 1 or args.k < 1 or args.soft_label_neighbors < 1:
        parser.error("--k, --k-graph, and --soft-label-neighbors must be positive")

    args.model_dir = args.model_dir or os.path.join(args.work_dir, "models")
    args.rank_dir = args.rank_dir or os.path.join(args.work_dir, "ranks")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    configure_runtime(device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.rank_dir, exist_ok=True)

    print(f"\n[{args.dataset_name}:{args.label}] Loading data...")
    xb_mmap, xq, gt = load_arrays(args)
    xb_orig_mmap = np.load(args.xb_orig, mmap_mode="r") if args.xb_orig else None
    if xb_orig_mmap is not None:
        if xb_orig_mmap.shape[0] != xb_mmap.shape[0]:
            parser.error(
                f"--xb-orig row count {xb_orig_mmap.shape[0]:,} does not match "
                f"--xb row count {xb_mmap.shape[0]:,}; row IDs must align."
            )
        print(f"  xb_orig: {xb_orig_mmap.shape[0]:,} x {xb_orig_mmap.shape[1]} dtype={xb_orig_mmap.dtype}")
    n, d = xb_mmap.shape
    print(f"  xb: {n:,} x {d} dtype={xb_mmap.dtype}")
    if xq is not None:
        print(f"  xq: {xq.shape[0]:,} x {xq.shape[1]}")
        print(f"  gt: {gt.shape}")
    print(f"  metric: {args.metric}")
    print(f"  device: {device}")
    if args.metric == "ip":
        print(f"  xb norm sample: {format_norm_stats(row_norm_stats(xb_mmap))}")
        if xq is not None:
            print(f"  xq norm sample: {format_norm_stats(args._xq_norm_stats)}")
        print("  WARNING: running raw inner product without extra normalization.")
        print("  Ensure --gt was computed with raw IP on the same vector representation.")

    using_train_subset = args.train_size is not None and args.train_size < n
    if using_train_subset:
        train_ids = sample_train_ids(n, args.train_size, args.seed)
        train_n = len(train_ids)
        if args.mode in ("train", "both"):
            xb_train = np.array(xb_mmap[train_ids], dtype=np.float32)
            print(f"  Training subset: {train_n:,} / {n:,} vectors loaded into RAM ({xb_train.nbytes // 1024**2} MB)")
        else:
            xb_train = None
            print(f"  Train-subset mode (search only): train_n={train_n:,}, xb_train not loaded")
    else:
        train_n = n
        xb_train = xb_mmap

    n_clusters_values = parse_int_list(args.n_clusters_values)
    nprobe_values = parse_int_list(args.nprobe_values)
    if not n_clusters_values or any(v < 1 for v in n_clusters_values):
        parser.error("--n-clusters-values must contain positive integers")
    if not nprobe_values or any(v < 1 for v in nprobe_values):
        parser.error("--nprobe-values must contain positive integers")
    if not args.skip_consistency and args.consistency_k >= n:
        parser.error(
            f"--consistency-k {args.consistency_k} must be < n={n:,}; "
            f"use --skip-consistency or lower the value."
        )
    # --train-size runs use chunked search for consistency, so the rank file
    # only needs k_graph and soft_label_neighbors depth.
    if using_train_subset:
        rank_depth = min(max(args.k_graph, args.soft_label_neighbors), train_n - 1)
    else:
        rank_depth = min(max(args.k_graph, args.soft_label_neighbors, args.consistency_k), n - 1)

    results = []
    consistency_results = []
    out_path = os.path.join(args.output_dir, f"nlsh_results_{args.label}.csv")
    cons_path = os.path.join(args.output_dir, f"knn_consistency_{args.label}.csv")

    for n_clusters in n_clusters_values:
        print(f"\n{'=' * 60}")
        print(
            f"n_clusters={n_clusters} | height={args.height} | "
            f"target final buckets ~ {n_clusters ** args.height:,}"
        )
        print(f"{'=' * 60}")

        run_dir = os.path.join(
            args.work_dir,
            f"{args.dataset_name}_{args.label}_{args.metric}_h{args.height}_c{n_clusters}",
        )
        model_path = model_path_for(
            args.model_dir,
            args.dataset_name,
            args.label,
            args.metric,
            args.height,
            n_clusters,
            args.soft_label_neighbors,
            bottom_level=args.bottom_level,
            train_n=train_n if using_train_subset else 0,
        )
        rank_path = rank_path_for(args.rank_dir, args.dataset_name, args.label, args.metric, rank_depth)
        rank_time = float("nan")

        need_ranks = args.mode in ("train", "both") or (
            args.mode == "search" and not args.skip_consistency and not using_train_subset
        )
        if need_ranks:
            rank_path, rank_time, _ = compute_exact_base_ranks(
                xb_mmap=xb_train,
                metric_name=args.metric,
                depth=rank_depth,
                rank_path=rank_path,
                chunk_size=args.rank_chunk_size,
                force=args.force_ranks,
            )

        build_time = float("nan")
        if args.mode in ("train", "both"):
            build_time = train_nlsh_model(args, xb_train, model_path, rank_path, run_dir, n_clusters, device)

        if args.mode in ("search", "both"):
            if not os.path.exists(model_path):
                print(f"  ERROR: no model found at {model_path}; run --mode train first.")
                continue

            eval_root = load_eval_root(model_path, device)
            leaf_ids = sorted(set(tensor_leaf_ids(eval_root)))
            final_buckets = len(leaf_ids)

            if using_train_subset:
                idx_paths = index_artifact_paths(
                    args.work_dir, args.label, args.metric, n, args.height, n_clusters,
                    train_n=train_n, seed=args.seed,
                    bottom_level=args.bottom_level,
                )
                os.makedirs(os.path.join(args.work_dir, "index"), exist_ok=True)
                assignments = infer_all_assignments(
                    eval_root, xb_mmap, args.index_chunk_size, device,
                    idx_paths["assignments"], args.force_retrain,
                )
                sorted_ids, offsets = build_sorted_bucket_index(
                    assignments, leaf_ids,
                    idx_paths["sorted_ids"], idx_paths["offsets"], idx_paths["meta"],
                    args.force_retrain,
                )
                leaf_to_indices_arg: dict | None = None
                counts = np.bincount(np.asarray(assignments).ravel(),
                                     minlength=int(max(leaf_ids)) + 1)
                sizes_np = np.array([counts[lid] for lid in leaf_ids], dtype=np.float64)
            else:
                assignments = assignments_from_eval_root(eval_root, n)
                leaf_to_indices_arg = build_bucket_index(assignments, leaf_ids)
                sorted_ids = offsets = None
                sizes_np = np.array(
                    [np.sum(assignments == lid) for lid in leaf_ids], dtype=np.float64
                )

            expected_size = n / max(final_buckets, 1)
            bucket_stats = {
                "bucket_mean": float(np.mean(sizes_np)),
                "bucket_std": float(np.std(sizes_np)),
                "bucket_min": int(np.min(sizes_np)),
                "bucket_max": int(np.max(sizes_np)),
                "bucket_imbalance": float(np.max(sizes_np) / expected_size) if expected_size > 0 else 0.0,
                "empty_buckets": int(np.sum(sizes_np == 0)),
            }

            if not args.skip_consistency:
                # With --xb-orig, always use chunked exact search over xb_orig.
                # The cached rank file was built on xb (possibly transformed),
                # so reusing it would measure transformed-space neighbors.
                if xb_orig_mmap is not None or using_train_subset:
                    stats = compute_knn_consistency_chunked(
                        xb_mmap=xb_orig_mmap if xb_orig_mmap is not None else xb_mmap,
                        assignments=assignments,
                        k=args.consistency_k,
                        num_samples=args.consistency_samples,
                        chunk_size=args.index_chunk_size,
                        metric_name=args.metric,
                        seed=args.seed,
                        cache_path=args.consistency_cache,
                    )
                else:
                    stats = compute_knn_consistency(
                        assignments=assignments,
                        rank_path=rank_path,
                        k=args.consistency_k,
                        num_samples=args.consistency_samples,
                        seed=args.seed,
                    )
                consistency_results.append({
                    "space": args.label,
                    "metric": args.metric,
                    "dim": d,
                    "n": n,
                    "height": args.height,
                    "n_clusters": n_clusters,
                    "final_buckets": final_buckets,
                    "soft_label_neighbors_effective": args.soft_label_neighbors + 1,
                    "k": args.consistency_k,
                    "num_samples": min(args.consistency_samples, n),
                    **bucket_stats,
                    **stats,
                })
                pd.DataFrame(consistency_results).to_csv(cons_path, index=False)
                print(f"  Consistency flushed to {cons_path}")

            if args.consistency_only:
                continue

            for nprobe in nprobe_values:
                if nprobe > n_clusters:
                    print(f"  Skipping nprobe={nprobe} (> per-level n_clusters={n_clusters})")
                    continue

                (
                    I,
                    avg_comps,
                    inference_time,
                    search_time,
                    leaf_buckets_visited_mean,
                    leaf_buckets_visited_max,
                ) = nlsh_search(
                    eval_root=eval_root,
                    xb_mmap=xb_mmap,
                    xq=xq,
                    nprobe=nprobe,
                    metric_name=args.metric,
                    batch_size=args.search_batch_size,
                    device=device,
                    leaf_to_indices=leaf_to_indices_arg,
                    sorted_ids=sorted_ids,
                    offsets=offsets,
                )
                r1 = get_recall(I, gt, 1)
                r5 = get_recall(I, gt, 5)
                r10 = get_recall(I, gt, 10)
                r30 = get_recall(I, gt, 30)
                total_time = inference_time + search_time
                print(
                    f"  nprobe={nprobe:4d} | R@1={r1:.4f} R@10={r10:.4f} R@30={r30:.4f}"
                    f" | leaves={leaf_buckets_visited_mean:.1f}/{leaf_buckets_visited_max}"
                    f" | dist_comp={avg_comps:.1f} | time={total_time:.3f}s"
                )
                results.append({
                    "space": args.label,
                    "metric": args.metric,
                    "dim": d,
                    "n": n,
                    "height": args.height,
                    "n_clusters": n_clusters,
                    "final_buckets": final_buckets,
                    "nprobe": nprobe,
                    "leaf_buckets_visited_mean": leaf_buckets_visited_mean,
                    "leaf_buckets_visited_max": leaf_buckets_visited_max,
                    "soft_label_neighbors_effective": args.soft_label_neighbors + 1,
                    "recall@1": r1,
                    "recall@5": r5,
                    "recall@10": r10,
                    "recall@30": r30,
                    "dist_computations": avg_comps,
                    "rank_precompute_time": rank_time,
                    "nlsh_build_time": build_time,
                    "inference_time": inference_time,
                    "search_time": search_time,
                    "total_time": total_time,
                })
                pd.DataFrame(results).to_csv(out_path, index=False)
                print(f"  Results flushed to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
