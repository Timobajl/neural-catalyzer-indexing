"""
Exhaustive k-NN search in the continuous output of one catalyzer checkpoint,
scored against the ground truth of the original space. Compares the authors'
checkpoints with the checkpoints trained for this thesis.
"""

import argparse
import time

import faiss
import numpy as np
import torch
from torch import nn

from knn import append_result, get_recall
from lib.net import Normalize, forward_pass


def one_recall(I: np.ndarray, gt: np.ndarray, rank: int) -> float:
    return round(float((I[:, :rank] == gt[:, :1]).any(axis=1).mean()), 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact k-NN in the output of a catalyzer checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint")
    parser.add_argument("--xb", required=True, help="Original base vectors (xb.npy)")
    parser.add_argument("--xq", required=True, help="Original query vectors (xq.npy)")
    parser.add_argument("--gt", required=True, help="Ground truth in the original space (gt.npy)")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--output-dir", default="knn_results")
    parser.add_argument("--output-name", default="knn_ceiling_authors.csv")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    xb = np.load(args.xb).astype("float32")
    xq = np.load(args.xq).astype("float32")
    gt = np.load(args.gt)

    net = nn.Sequential(
        nn.Linear(xb.shape[1], a.dint), nn.BatchNorm1d(a.dint), nn.ReLU(),
        nn.Linear(a.dint, a.dint), nn.BatchNorm1d(a.dint), nn.ReLU(),
        nn.Linear(a.dint, a.dout), Normalize(),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    t0 = time.time()
    xbt = forward_pass(net, xb, bs=4096, device="cpu").astype("float32")
    xqt = forward_pass(net, xq, bs=4096, device="cpu").astype("float32")
    forward_time = time.time() - t0

    index = faiss.IndexFlatL2(a.dout)
    index.add(xbt)
    _, I = index.search(xqt, 100)

    row = {
        "space": args.label, "dim": a.dout, "lambda": a.lambda_uniform, "epochs": a.epochs,
        "saved_epoch": ckpt.get("epoch"), "save_best_criterion": a.save_best_criterion,
        "n": xb.shape[0], "nq": xq.shape[0],
        **{f"recall@{k}": get_recall(I, gt, k) for k in (1, 5, 10, 30)},
        **{f"1-recall@{r}": one_recall(I, gt, r) for r in (1, 10, 100)},
    }
    print(f"[{args.label}] recall@10: {row['recall@10']:.4f} | 1-recall@10: {row['1-recall@10']:.4f} | forward: {forward_time:.1f}s")
    append_result(row, args.output_dir, args.output_name)
