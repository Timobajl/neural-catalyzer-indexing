"""
Exhaustive k-NN search that reads the base vectors in chunks, scored against
the ground truth of the original space. Gives the ceiling of a space that does
not fit in memory, where knn.py would hold the base vectors twice.
"""

import argparse
import time

import faiss
import numpy as np

from knn import append_result, get_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact k-NN in chunks (R_max ceiling)")
    parser.add_argument("--xb", required=True, help="Path to base vectors (xb.npy or xbt.npy)")
    parser.add_argument("--xq", required=True, help="Path to query vectors (xq.npy or xqt.npy)")
    parser.add_argument("--gt", required=True, help="Path to ground truth (gt.npy)")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--output-dir", default="knn_results")
    parser.add_argument("--output-name", default="knn_ceiling_full.csv")
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--k-list", default="1,5,10,30")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    k_list = [int(k) for k in args.k_list.split(",")]
    k_max = max(k_list)
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2

    xq = np.load(args.xq).astype("float32")
    gt = np.load(args.gt)
    n, d = np.load(args.xb, mmap_mode="r").shape
    nq = xq.shape[0]
    if xq.shape[1] != d:
        parser.error(f"xq dimensionality {xq.shape[1]} does not match xb dimensionality {d}")
    if gt.shape[0] != nq or gt.shape[1] < k_max:
        parser.error(f"gt shape {gt.shape} does not fit {nq} queries and k={k_max}")

    heap = faiss.ResultHeap(nq, k_max, keep_max=args.metric == "ip")
    t0 = time.time()
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        # The memory map is opened per chunk, so its pages do not pile up in RAM.
        chunk = np.load(args.xb, mmap_mode="r")[start:stop].astype("float32")
        D, I = faiss.knn(xq, chunk, min(k_max, stop - start), metric=metric)
        heap.add_result(D, I + start)
    heap.finalize()
    total_time = time.time() - t0

    recalls = {f"recall@{k}": get_recall(heap.I, gt, k) for k in k_list}
    print(f"[{args.label}] " + " | ".join(f"{name}: {v:.4f}" for name, v in recalls.items())
          + f" | Total: {total_time:.2f}s")

    row = {
        "space": args.label, "metric": args.metric, "dim": d, "n": n, "nq": nq,
        **recalls,
        "dist_computations": float(n), "total_time": total_time,
    }
    append_result(row, args.output_dir, args.output_name)
