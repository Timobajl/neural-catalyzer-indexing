"""
Exhaustive k-NN search in one vector space, scored against the ground truth of
the original space. Selects the catalyzer lambda and measures the PCA baseline.
"""

import faiss
import numpy as np
import time
import argparse
import pandas as pd
import os


def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    n = len(I)
    correct = sum(np.isin(I[i, :k], gt[i, :k]).sum() for i in range(n))
    return round(correct / (n * k), 4)


def load_subset(xb_path: str, n_samples: int | None, seed: int) -> np.ndarray:
    xb_full = np.load(xb_path, mmap_mode="r")
    total = xb_full.shape[0]

    if n_samples is None or n_samples >= total:
        return np.array(xb_full, dtype="float32")

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n_samples, replace=False)
    idx.sort()
    return np.array(xb_full[idx], dtype="float32")


def apply_pca(xt: np.ndarray, xb: np.ndarray, xq: np.ndarray, dim: int, normalize: bool = True):
    if dim > xt.shape[1]:
        raise ValueError(f"--pca-dim ({dim}) cannot exceed the input dimension ({xt.shape[1]})")

    mat = faiss.PCAMatrix(xt.shape[1], dim)
    mat.train(xt)
    xb_p = mat.apply_py(xb).astype("float32")
    xq_p = mat.apply_py(xq).astype("float32")

    if normalize:
        faiss.normalize_L2(xb_p)
        faiss.normalize_L2(xq_p)
    return xb_p, xq_p


def append_result(row: dict, output_dir: str, output_name: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_name)

    write_header = not os.path.exists(csv_path)
    new_row = pd.DataFrame([row])
    new_row.to_csv(csv_path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact k-NN Benchmark (R_max ceiling)")
    parser.add_argument("--xb", required=True, help="Path to base vectors (xb.npy or xbt.npy)")
    parser.add_argument("--xq", required=True, help="Path to query vectors (xq.npy or xqt.npy)")
    parser.add_argument("--gt", default=None, help="Path to ground truth (gt.npy)")
    parser.add_argument("--gt-out", default=None, help="Where to save newly computed gt")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--output-dir", default="knn_results")
    parser.add_argument("--output-name", default="knn_ceiling_results.csv")
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--k-list", default="1,5,10,30")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xt", default=None, help="Path to training/learn vectors (required with --pca-dim)")
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--pca-normalize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    k_list = [int(k) for k in args.k_list.split(",")]

    xb = load_subset(args.xb, args.n_samples, args.seed)
    xq = np.load(args.xq).astype("float32")

    if args.pca_dim is not None:
        if args.xt is None:
            raise ValueError("--xt is required with --pca-dim (PCA is fitted on the learn set, not on xb)")
        xt = np.array(np.load(args.xt, mmap_mode="r"), dtype="float32")
        xb, xq = apply_pca(xt, xb, xq, args.pca_dim, normalize=args.pca_normalize)

    n, d = xb.shape
    nq = xq.shape[0]

    index = faiss.IndexFlatIP(d) if args.metric == "ip" else faiss.IndexFlatL2(d)

    t0 = time.time()
    index.add(xb)
    add_time = time.time() - t0

    t0 = time.time()
    D, I = index.search(xq, max(k_list))
    search_time = time.time() - t0

    if args.gt is None:
        gt = I
        if args.gt_out:
            os.makedirs(os.path.dirname(args.gt_out) or ".", exist_ok=True)
            np.save(args.gt_out, gt)
    else:
        gt = np.load(args.gt)

    recalls = {f"recall@{k}": get_recall(I, gt, k) for k in k_list}

    print(f"[{args.label}] " + " | ".join(f"{k_name}: {v:.4f}" for k_name, v in recalls.items())
          + f" | Search: {search_time:.2f}s")

    row = {
        "space": args.label, "metric": args.metric, "dim": d, "n": n, "nq": nq, "seed": args.seed,
        "pca_dim": args.pca_dim,
        "pca_normalize": args.pca_normalize if args.pca_dim is not None else None,
        **recalls,
        "dist_computations": float(n), "add_time": add_time, "search_time": search_time,
    }
    append_result(row, args.output_dir, args.output_name)
