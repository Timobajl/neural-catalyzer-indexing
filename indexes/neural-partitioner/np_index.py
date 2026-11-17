"""
Neural Partitioner benchmark (Fahim et al. 2023, reference at
https://github.com/abrar-fahim/Neural-Partitioner).

The changes relative to the reference are in neural-partitioner.patch and in the README.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import utils
from loss_fn import MyLoss
from modelforest import ModelForest
import np_dataset as npd


def _sentinel_path(models_path: str, label: str) -> str:
    # "000" is the root's id in upstream's filename scheme.
    return os.path.join(models_path, f"{label}-best-model-000-parameters.pt")


def _ba_path(out_dir: str, tag: str) -> str:
    return os.path.join(out_dir, "bin_assignments", f"{tag}.npy")


def train_or_load_np(
    X_input_subset: np.ndarray,
    knn_path: str,
    n_bins: int,
    epochs: int,
    batch_size: int,
    lr: float,
    models_path: str,
    label: str,
    k_prime: int,
    hierarchical_branching: list,
    metric: str,
    eta: float = 7.0,
    force_rebuild: bool = False,
) -> tuple:
    """Train a Neural Partitioner forest, or load it from disk if already done.

    Returns (forest, train_time_s, was_loaded_from_disk).
    """
    n_train, d_in = X_input_subset.shape
    os.makedirs(models_path, exist_ok=True)

    if os.path.exists(knn_path) and not force_rebuild:
        print(f"  [cache] Loading kNN from {knn_path}")
        Y = npd.load_knn(knn_path, device="cpu")
    else:
        Y = npd.build_knn_exact(X_input_subset, k=k_prime, metric=metric)
        npd.save_knn(Y, knn_path)
    Y = Y.to(utils.primary_device)

    if not hierarchical_branching:
        branching, levels = n_bins, 0
    else:
        branching = hierarchical_branching[0]
        levels    = len(hierarchical_branching) - 1

    forest = ModelForest(
        n_trees=1,
        n_input=d_in,
        branching=branching,
        levels=levels,
        n_classes=n_bins,
        n_hidden_params=128,
        model_type="neural",
        metric=metric,
    )

    sentinel = _sentinel_path(models_path, label)
    if os.path.exists(sentinel) and not force_rebuild:
        print(f"  [cache] Model exists - loading weights from {models_path}")
        forest.build_forest(load_from_file=True, data=label, models_path=models_path)
        del Y
        torch.cuda.empty_cache()
        return forest, 0.0, True

    # Train from scratch
    forest.build_forest(load_from_file=False, data=label, models_path=models_path)

    # Wrap MyLoss to inject eta
    _base_crit = MyLoss()
    class _CritWithEta:
        def __call__(self, *a, **kw):
            kw.setdefault("eta", eta)
            return _base_crit(*a, **kw)
    crit = _CritWithEta()

    X_t = torch.tensor(X_input_subset, dtype=torch.float32, device=utils.primary_device)

    print(f"  Training: n={n_train:,}  d_in={d_in}  n_bins={n_bins}  "
          f"epochs={epochs}  min batch={batch_size:,}  eta={eta}")
    t0 = time.time()
    forest.train_forest(
        X=X_t, Y=Y, crit=crit, n_epochs=epochs,
        batch_size=batch_size, lr=lr, data=label, models_path=models_path,
    )
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s ({train_time/60:.1f} min)")

    del X_t, Y
    torch.cuda.empty_cache()
    return forest, train_time, False


def index_or_load(
    forest,
    xb_mmap: np.ndarray,
    out_dir: str,
    tag: str,
    chunk_size: int,
    force_rebuild: bool = False,
) -> tuple:
    """Assign all base vectors to bins, or load cache.

    Returns (bin_assignments np.int32 shape (n,), add_time_s).
    """
    cache = _ba_path(out_dir, tag)

    if os.path.exists(cache) and not force_rebuild:
        print(f"  [cache] Loading bin assignments from {cache}")
        t0 = time.time()
        bin_assignments = np.load(cache)
        add_time = time.time() - t0
        print(f"  Loaded {len(bin_assignments):,} assignments in {add_time:.1f}s")
        return bin_assignments, add_time

    # Run full-dataset inference
    t0 = time.time()
    bin_assignments = npd.index_full_dataset(forest, xb_mmap, chunk_size=chunk_size)
    add_time = time.time() - t0

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, bin_assignments)
    print(f"  Saved bin assignments -> {cache}")
    return bin_assignments, add_time


def main():
    parser = argparse.ArgumentParser(description="Neural Partitioner benchmark.")

    parser.add_argument("--xb", required=True, help="Base vectors (.npy)")
    parser.add_argument("--xb-orig", default=None,
                        help="Original-space vectors. Used by kNN consistency on transformed-space runs.")
    parser.add_argument("--xq", required=True, help="Query vectors (.npy), same space as --xb")
    parser.add_argument("--gt", required=True, help="Ground-truth indices (.npy), original-space")
    parser.add_argument("--label", required=True, help="Written to the 'space' CSV column.")
    parser.add_argument("--output-dir", default="np_results")
    parser.add_argument("--metric", default="l2", choices=["l2", "ip"])

    parser.add_argument("--n-bins-values", default="1024",
                        help="Comma-separated final bin counts. Must equal "
                             "product of --hierarchical-branching when hierarchical.")
    parser.add_argument("--nprobe-values", default="1,2,4,8,16,32")

    parser.add_argument("--n-train", type=int, default=1_000_000,
                        help="Training subset size.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Smallest batch. Each node uses at least 4%% of its training vectors.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k-prime", type=int, default=10,
                        help="k' for training kNN matrix.")
    parser.add_argument("--eta", type=float, default=7.0,
                        help="Balance loss weight (paper Eq.5: loss = quality + eta*balance).")
    parser.add_argument("--eta-values", default=None,
                        help="Optional comma-separated eta sweep. Overrides --eta.")
    parser.add_argument("--hierarchical-branching", default="32,32",
                        help="Comma-separated branching factors, or 'none' for flat. "
                             "E.g. '100,100' = two-level 100x100=10000 bins.")

    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="Vectors per forward-pass chunk during full-dataset indexing.")

    parser.add_argument("--consistency-samples", type=int, default=10_000,
                        help="Vectors sampled for kNN consistency.")
    parser.add_argument("--consistency-k", type=int, default=10,
                        help="k used for kNN consistency (should equal --k-prime).")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--consistency-only", action="store_true",
                        help="Compute kNN consistency and skip the nprobe sweep.")
    parser.add_argument("--consistency-cache", default=None,
                        help="Path to precomputed kNN-consistency cache (.npz from "
                             "compute_true_neighbors.py). When set, skips inline kNN.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild", action="store_true")

    args = parser.parse_args()

    n_bins_values        = [int(v) for v in args.n_bins_values.split(",")]
    nprobe_values = [int(v) for v in args.nprobe_values.split(",")]
    hierarchical_branching = None if args.hierarchical_branching.lower() == "none" else (
        [int(v) for v in args.hierarchical_branching.split(",")]
        if args.hierarchical_branching else None
    )
    eta_values = (
        [float(v) for v in args.eta_values.split(",") if v.strip()]
        if args.eta_values else [args.eta]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    models_base = os.path.join(args.output_dir, "models")
    knn_cache   = os.path.join(args.output_dir, "knn_cache")
    os.makedirs(models_base, exist_ok=True)
    os.makedirs(knn_cache, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Neural Partitioner benchmark")
    print(f"Device:              {utils.primary_device}")
    print(f"label:               {args.label}")
    print(f"metric:              {args.metric}")
    print(f"xb:                  {args.xb}")
    print(f"xb_orig:             {args.xb_orig or '(same as xb)'}")
    print(f"n_train:             {args.n_train:,}")
    print(f"n_bins_values:       {n_bins_values}")
    print(f"nprobe_values:       {nprobe_values}")
    print(f"epochs:              {args.epochs}  |  lr={args.lr}  |  eta_values={eta_values}")
    print(f"hierarchical:        {hierarchical_branching or 'flat'}")
    print(f"consistency_samples: {args.consistency_samples}")
    print(f"force_rebuild:       {args.force_rebuild}")
    print(f"{'='*60}\n")

    xb_mmap = npd.load_mmap(args.xb)
    xb_orig = npd.load_mmap(args.xb_orig) if args.xb_orig else xb_mmap
    # Row IDs must match, the dimensions differ in transformed-space runs.
    if args.xb_orig and xb_orig.shape[0] != xb_mmap.shape[0]:
        parser.error(
            f"--xb-orig row count {xb_orig.shape[0]:,} does not match "
            f"--xb row count {xb_mmap.shape[0]:,}; row IDs must align."
        )
    xq      = np.load(args.xq).astype("float32")
    gt      = np.load(args.gt)
    n_full  = xb_mmap.shape[0]

    print(f"Base:    {n_full:,} x {xb_mmap.shape[1]}D")
    print(f"Queries: {xq.shape[0]:,}  |  GT: {gt.shape}")

    X_input_subset = npd.sample_subset(xb_mmap, args.n_train, seed=args.seed)
    n_train_actual = X_input_subset.shape[0]

    knn_path = os.path.join(
        knn_cache, f"{args.label}_{args.metric}_n{n_train_actual}_k{args.k_prime}.hdf5"
    )

    all_results = []
    out_path = os.path.join(args.output_dir, f"np_results_{args.label}.csv")

    for n_bins in n_bins_values:
        if hierarchical_branching:
            final_bins = int(np.prod(hierarchical_branching))
            if final_bins != n_bins:
                raise ValueError(
                    f"--hierarchical-branching {hierarchical_branching} produces "
                    f"{final_bins} bins, but current n_bins is {n_bins}."
                )
        for eta in eta_values:
            bins_tag    = f"{args.label}_{args.metric}_nbins{n_bins}_eta{int(eta)}"
            models_path = os.path.join(models_base, bins_tag)

            print(f"\n{'-'*60}")
            print(f" n_bins={n_bins} | eta={eta}")
            print(f"{'-'*60}")

            forest, train_time, loaded = train_or_load_np(
                X_input_subset=X_input_subset,
                knn_path=knn_path,
                n_bins=n_bins,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                models_path=models_path,
                label=bins_tag,
                k_prime=args.k_prime,
                hierarchical_branching=hierarchical_branching,
                metric=args.metric,
                eta=eta,
                force_rebuild=args.force_rebuild,
            )

            bin_assignments, add_time = index_or_load(
                forest=forest,
                xb_mmap=xb_mmap,
                out_dir=args.output_dir,
                tag=bins_tag,
                chunk_size=args.chunk_size,
                force_rebuild=args.force_rebuild,
            )

            lookup     = npd.build_lookup_table(bin_assignments, n_bins=n_bins)
            bin_sizes  = [len(v) for v in lookup.values()]
            model_tree = forest.trees[0]

            bin_sizes_np   = np.asarray(bin_sizes, dtype=np.float64)
            target_bin     = n_full / max(n_bins, 1)
            bucket_stats = {
                "bucket_min":       int(bin_sizes_np.min()),
                "bucket_max":       int(bin_sizes_np.max()),
                "bucket_mean":      round(float(bin_sizes_np.mean()), 1),
                "bucket_std":       round(float(bin_sizes_np.std()), 1),
                "bucket_imbalance": round(float(bin_sizes_np.max() / target_bin), 3) if target_bin > 0 else 0.0,
                "empty_buckets":    int(np.sum(bin_sizes_np == 0)),
            }
            print(f"  Bin stats: min={bucket_stats['bucket_min']:,}  max={bucket_stats['bucket_max']:,}  "
                  f"mean={bucket_stats['bucket_mean']:.0f}  std={bucket_stats['bucket_std']:.0f}  "
                  f"imbalance={bucket_stats['bucket_imbalance']:.2f}  empty={bucket_stats['empty_buckets']}  "
                  f"(target ~{int(target_bin):,})")

            consistency_stats = {}
            if not args.skip_consistency:
                n_cons = min(args.consistency_samples, n_full)
                consistency_stats = npd.compute_knn_consistency_full_db(
                    bin_assignments=bin_assignments,
                    xb_mmap=xb_orig,
                    k=args.consistency_k,
                    num_samples=n_cons,
                    chunk_size=args.chunk_size,
                    metric=args.metric,
                    seed=args.seed,
                    cache_path=args.consistency_cache,
                )
                cons_row = {
                    "space":       args.label,
                    "metric":      args.metric,
                    "dim":         xb_mmap.shape[1],
                    "n":           n_full,
                    "n_train":     n_train_actual,
                    "n_bins":      n_bins,
                    "eta":         eta,
                    "k":           args.consistency_k,
                    "num_samples": n_cons,
                    **consistency_stats,
                    **bucket_stats,
                }
                cons_path = os.path.join(args.output_dir, f"knn_consistency_{args.label}.csv")
                pd.DataFrame([cons_row]).to_csv(
                    cons_path, mode="a",
                    header=not os.path.exists(cons_path), index=False,
                )
                print(f"  Consistency flushed -> {cons_path}")

            if args.consistency_only:
                del forest, bin_assignments, lookup
                torch.cuda.empty_cache()
                continue

            for nprobe in nprobe_values:
                if nprobe > n_bins:
                    print(f"  Skipping nprobe={nprobe} (> n_bins={n_bins})")
                    continue

                metrics = npd.compute_recall(
                    lookup=lookup,
                    xb_mmap=xb_mmap,
                    xq=xq,
                    gt=gt,
                    model_tree=model_tree,
                    n_bins_to_search=nprobe,
                    metric=args.metric,
                )

                print(f"  nprobe={nprobe:4d} | "
                      f"R@1={metrics['recall@1']:.4f}  "
                      f"R@10={metrics['recall@10']:.4f}  "
                      f"R@30={metrics['recall@30']:.4f} | "
                      f"dist_comp={metrics['dist_computations']:.0f} | "
                      f"infer={metrics['inference_time']:.3f}s  "
                      f"search={metrics['search_time']:.3f}s")

                row = {
                    "space":              args.label,
                    "metric":             args.metric,
                    "dim":                xb_mmap.shape[1],
                    "n":                  n_full,
                    "n_train":            n_train_actual,
                    "n_bins":             n_bins,
                    "nprobe":             nprobe,
                    "eta":                eta,
                    "recall@1":           metrics["recall@1"],
                    "recall@5":           metrics["recall@5"],
                    "recall@10":          metrics["recall@10"],
                    "recall@30":          metrics["recall@30"],
                    "dist_computations":  metrics["dist_computations"],
                    "inference_time":     metrics["inference_time"],
                    "search_time":        metrics["search_time"],
                    "total_time":         metrics["total_time"],
                    "train_time":         round(train_time, 1),
                    "add_time":           round(add_time, 1),
                    "knn_consistency_mean": consistency_stats.get("knn_consistency_mean", float("nan")),
                    "knn_consistency_std":  consistency_stats.get("knn_consistency_std",  float("nan")),
                    "model_loaded":       loaded,
                }
                all_results.append(row)

            pd.DataFrame(all_results).to_csv(out_path, index=False)
            print(f"  Results flushed -> {out_path}")

            del forest, bin_assignments, lookup
            torch.cuda.empty_cache()

    print(f"\nDone. Final results -> {out_path}")


if __name__ == "__main__":
    main()
