"""
Buckets and routing from one vector space, candidates ranked by their distance
in the original space. Removes the ceiling of the transformation, so recall
depends only on whether the true neighbors reach the visited buckets.
"""

import argparse
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch
from faiss.contrib.ivf_tools import add_preassigned, search_preassigned

from ivfflat_index import get_recall, make_ivfflat_index, read_assignments, train_and_add
from lmi_index import LMI


def build_ivfflat(xb: np.ndarray, n_buckets: int, chunk_size: int):
    n, d = xb.shape
    index = make_ivfflat_index(d, n_buckets, "l2")
    train_and_add(index, xb, n, min(n, 100 * n_buckets), chunk_size)
    assignments = read_assignments(index, n, n_buckets).astype(np.int64)
    return assignments, lambda xq, nprobe: index.quantizer.search(xq, nprobe)[1]


def build_lmi(xb: np.ndarray, n_buckets: int, chunk_size: int):
    lmi = LMI.create(xb, 15, 0.00098, min(xb.shape[0], 1_000_000), n_buckets, chunk_size, "l2", 8, 4, assignments_only=True)
    assignments = lmi.bucket_assignments.numpy().astype(np.int64)
    return assignments, lambda xq, nprobe: lmi._predict(torch.from_numpy(xq), nprobe).numpy().astype(np.int64)


BUILDERS = {"ivfflat": build_ivfflat, "lmi": build_lmi}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route in one space, rank in the original space")
    parser.add_argument("--router", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--xb", required=True, help="Base vectors that define the buckets")
    parser.add_argument("--xq", required=True, help="Query vectors in the same space as --xb")
    parser.add_argument("--xb-orig", required=True, help="Original base vectors")
    parser.add_argument("--xq-orig", required=True, help="Original query vectors")
    parser.add_argument("--gt", required=True, help="Ground truth in the original space")
    parser.add_argument("--label", required=True, help="Written to the 'space' CSV column.")
    parser.add_argument("--output-dir", default="rerank_results")
    parser.add_argument("--n-buckets-values", default="1000,2000,4000")
    parser.add_argument("--nprobe-values", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)

    xb = np.load(args.xb, mmap_mode="r")
    xq = np.load(args.xq).astype("float32")
    xb_orig = np.array(np.load(args.xb_orig, mmap_mode="r"), dtype="float32")
    xq_orig = np.load(args.xq_orig).astype("float32")
    gt = np.load(args.gt)
    if xb.shape[0] != xb_orig.shape[0] or xq.shape[0] != xq_orig.shape[0]:
        parser.error("--xb/--xq and --xb-orig/--xq-orig must have the same rows")

    n, d = xb_orig.shape
    nprobe_values = [int(v) for v in args.nprobe_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"rerank_{args.router}_{args.label}.csv")

    rows = []
    for n_buckets in [int(v) for v in args.n_buckets_values.split(",")]:
        print(f"\n--- {args.label}: {args.router}, {n_buckets} buckets ---", flush=True)
        assignments, route = BUILDERS[args.router](xb, n_buckets, args.chunk_size)
        visited = route(xq, max(nprobe_values))

        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, n_buckets, faiss.METRIC_L2)
        index.is_trained = True
        add_preassigned(index, xb_orig, assignments)

        for nprobe in nprobe_values:
            index.nprobe = nprobe
            faiss.cvar.indexIVF_stats.reset()
            t0 = time.time()
            _, I = search_preassigned(index, xq_orig, 30, visited[:, :nprobe])
            search_time = time.time() - t0
            dist_comps = faiss.cvar.indexIVF_stats.ndis / xq_orig.shape[0]
            recalls = {f"recall@{k}": get_recall(I, gt, k) for k in (1, 5, 10, 30)}
            print(f"  nprobe={nprobe:4d} | recall@10={recalls['recall@10']:.4f} | dist_comps={dist_comps:.1f}", flush=True)
            rows.append({
                "space": args.label, "router": args.router, "dim": xb.shape[1], "n": n, "n_buckets": n_buckets,
                "nprobe": nprobe, **recalls, "dist_computations": dist_comps, "search_time": search_time,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")
