"""HNSW benchmark. Builds a FAISS IndexHNSWFlat and sweeps efSearch."""

import faiss
import numpy as np
import time
import argparse
import pandas as pd
import os


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


def estimate_ram_gb(n: int, d: int, M: int) -> float:
    vectors_gb = n * d * 4 / 1024 ** 3
    graph_gb = n * M * 2 * 4 / 1024 ** 3
    return vectors_gb + graph_gb


def build_index(
    xb_mmap: np.ndarray,
    n: int,
    d: int,
    M: int,
    efConstruction: int,
    chunk_size: int,
    metric_type: int,
) -> tuple[faiss.IndexHNSWFlat, float]:
    index = faiss.IndexHNSWFlat(d, M, metric_type)
    index.hnsw.efConstruction = efConstruction

    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"  Adding {n:,} vectors in {n_chunks} chunks of {chunk_size:,}...")

    t0 = time.time()
    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        index.add(xb_mmap[start:stop].astype("float32"))
        elapsed = time.time() - t0
        speed = stop / max(elapsed, 1e-6)
        eta = (n - stop) / speed
        print(
            f"  Chunk {chunk_idx}/{n_chunks} | {stop:,}/{n:,} vectors"
            f" | {elapsed/3600:.2f}h elapsed | ETA {eta/3600:.2f}h",
            end="\r",
        )
    add_time = time.time() - t0
    print(f"\n  Build done in {add_time/3600:.2f}h ({add_time:.0f}s)")
    return index, add_time


def index_path_for(index_dir: str, label: str, metric_name: str, M: int, efC: int) -> str:
    return os.path.join(index_dir, f"hnsw_{label}_{metric_name}_M{M}_efC{efC}.faiss")


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
    parser = argparse.ArgumentParser(description="HNSW benchmark.")

    parser.add_argument("--xb", required=True, help="Base vectors (.npy)")
    parser.add_argument("--xq", help="Query vectors (.npy)")
    parser.add_argument("--gt", help="Ground-truth indices (.npy)")
    parser.add_argument("--label", required=True, help="Tag written into the CSV")
    parser.add_argument("--output-dir", default="hnsw_results")
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")

    parser.add_argument(
        "--mode",
        choices=["build", "search", "both"],
        default="both",
        help="build=add only | search=load+sweep only | both=full pipeline",
    )

    parser.add_argument("--M-values", type=str, default="16,32,48")
    parser.add_argument("--efConstruction", type=int, default=500)
    parser.add_argument(
        "--efSearch-values",
        type=str,
        default="30,40,50,60,80,100,150,200,300,400,500,600,800,1000",
    )

    parser.add_argument("--chunk-size", type=int, default=500_000)

    parser.add_argument("--index-dir", default="saved_indexes")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-save-index", action="store_true")

    args = parser.parse_args()
    args.save_index = not args.no_save_index

    if args.mode in ("search", "both") and (not args.xq or not args.gt):
        parser.error("--xq and --gt are required for --mode search/both")

    print(f"\n[{args.label}] Loading data...")
    xb_mmap = np.load(args.xb, mmap_mode="r")
    n, d = xb_mmap.shape
    print(f"  xb: {n:,} x {d}  ({n * d * 4 / 1024**3:.1f} GB as float32)")

    xq, gt = None, None
    if args.mode in ("search", "both"):
        xq = np.load(args.xq).astype("float32")
        gt = np.load(args.gt)
        if xq.shape[1] != d:
            parser.error(f"xq dimensionality {xq.shape[1]} does not match xb dimensionality {d}")
        if gt.shape[1] < 30:
            parser.error(f"gt must contain at least 30 neighbors per query, got shape {gt.shape}")
        print(f"  xq: {xq.shape[0]:,} x {xq.shape[1]}")
        print(f"  gt: {gt.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    M_values = [int(v) for v in args.M_values.split(",")]
    efSearch_values = [int(v) for v in args.efSearch_values.split(",")]
    efConstruction = args.efConstruction
    metric_type = metric_name_to_faiss(args.metric)
    print(f"  metric: {args.metric} (FAISS metric_type={metric_type})")

    for M in M_values:
        ram_gb = estimate_ram_gb(n, d, M)
        print(f"  Estimated RAM for M={M}: {ram_gb:.1f} GB")

    results = []

    for M in M_values:
        print(f"\n{'='*60}")
        print(
            f"M={M} | efConstruction={efConstruction} | metric={args.metric} "
            f"| mode={args.mode} | label={args.label}"
        )
        print(f"{'='*60}")

        ipath = index_path_for(args.index_dir, args.label, args.metric, M, efConstruction)
        add_time = float("nan")
        index = None

        if args.mode in ("build", "both"):
            if os.path.exists(ipath) and not args.force_rebuild:
                print(f"  Index already exists at {ipath}, skipping build.")
                print(f"  Pass --force-rebuild to rebuild from scratch.")
            else:
                index, add_time = build_index(
                    xb_mmap, n, d, M, efConstruction, args.chunk_size, metric_type
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

            print(f"\n  Sweeping efSearch: {efSearch_values}")
            for efSearch in efSearch_values:
                index.hnsw.efSearch = efSearch

                faiss.cvar.hnsw_stats.reset()
                t0 = time.time()
                _, I = index.search(xq, 30)
                search_time = time.time() - t0

                ndis = faiss.cvar.hnsw_stats.ndis
                avg_dist_comp = ndis / len(xq)

                r1  = get_recall(I, gt, 1)
                r5  = get_recall(I, gt, 5)
                r10 = get_recall(I, gt, 10)
                r30 = get_recall(I, gt, 30)

                print(
                    f"  efSearch={efSearch:4d} | R@1={r1:.4f} R@10={r10:.4f} R@30={r30:.4f}"
                    f" | DistComps={avg_dist_comp:.0f} | {search_time:.3f}s"
                )

                results.append({
                    "space":           args.label,
                    "metric":          args.metric,
                    "dim":             d,
                    "M":               M,
                    "efConstruction":  efConstruction,
                    "efSearch":        efSearch,
                    "recall@1":        r1,
                    "recall@5":        r5,
                    "recall@10":       r10,
                    "recall@30":       r30,
                    "add_time":        add_time,
                    "search_time":     search_time,
                    "dist_computations": avg_dist_comp,
                })

            out_path = os.path.join(args.output_dir, f"hnsw_results_{args.label}.csv")
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"  Results flushed to {out_path}")

    print("\nDone.")
