"""
Time of the catalyzer forward pass on the query vectors, without checkpoint
loading and file I/O. Measured per batch size, with the threads of the job.
"""

import argparse
import statistics
import time

import numpy as np
import torch
from torch import nn

from knn import append_result
from lib.net import Normalize, forward_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward pass time of a catalyzer checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint")
    parser.add_argument("--xq", required=True, help="Original query vectors (xq.npy)")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--output-dir", default="inference_results")
    parser.add_argument("--output-name", default="inference_times.csv")
    parser.add_argument("--batch-sizes", default="10000,1")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    xq = np.load(args.xq).astype("float32")

    net = nn.Sequential(
        nn.Linear(xq.shape[1], a.dint), nn.BatchNorm1d(a.dint), nn.ReLU(),
        nn.Linear(a.dint, a.dint), nn.BatchNorm1d(a.dint), nn.ReLU(),
        nn.Linear(a.dint, a.dout), Normalize(),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
        times = []
        with torch.no_grad():
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                forward_pass(net, xq, bs=batch_size, device="cpu")
                times.append(time.perf_counter() - t0)
        median_time = statistics.median(times)
        per_query = median_time / xq.shape[0]

        print(f"[{args.label}] batch {batch_size} | threads {torch.get_num_threads()} | {per_query * 1e6:.1f} us per query")
        row = {
            "space": args.label, "dim": a.dout, "nq": xq.shape[0],
            "batch_size": batch_size, "threads": torch.get_num_threads(), "repeats": args.repeats,
            "median_time": median_time, "min_time": min(times), "max_time": max(times),
            "per_query_us": per_query * 1e6, "qps": 1 / per_query,
        }
        append_result(row, args.output_dir, args.output_name)
