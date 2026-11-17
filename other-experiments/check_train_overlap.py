"""
Counts how many rows of the LAION 10M file also occur in the indexed 100M base,
for the whole file and for the slices that training reads from it.
"""

import argparse
import hashlib
import os

import h5py
import numpy as np
import pandas as pd

from count_duplicates import hash_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlap of the LAION training rows with the indexed base")
    parser.add_argument("--xb", required=True, help="Path to the indexed base vectors (xb.npy)")
    parser.add_argument("--h5", required=True, help="Path to the LAION 10M file (.h5)")
    parser.add_argument("--output-dir", default="duplicates_results")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--val-base", type=int, default=1_000_000)
    parser.add_argument("--val-queries", type=int, default=100_000)
    parser.add_argument("--num-learn", type=int, default=500_000)
    args = parser.parse_args()

    xb_hashes = hash_rows(args.xb, args.chunk_size)

    with h5py.File(args.h5, "r") as f:
        emb = f["emb"]
        n = emb.shape[0]
        if emb.dtype != np.load(args.xb, mmap_mode="r").dtype:
            parser.error(f"dtype {emb.dtype} of --h5 does not match --xb")
        h5_hashes = np.empty(n, dtype=np.uint64)
        for start in range(0, n, args.chunk_size):
            chunk = emb[start:start + args.chunk_size]
            digests = b"".join(hashlib.blake2b(row, digest_size=8).digest() for row in chunk)
            h5_hashes[start:start + len(chunk)] = np.frombuffer(digests, dtype=np.uint64)

    in_base = np.isin(h5_hashes, xb_hashes)

    val_stop = args.val_base + args.val_queries
    train_stop = val_stop + args.num_learn
    slices = {
        "whole_file": (0, n), "validation_base": (0, args.val_base),
        "validation_queries": (args.val_base, val_stop), "training": (val_stop, train_stop),
    }
    rows = []
    for name, (start, stop) in slices.items():
        found = int(in_base[start:stop].sum())
        rows.append({"slice": name, "start": start, "stop": stop, "rows": stop - start,
                     "rows_in_base": found, "share_in_base": found / (stop - start)})
        print(f"[{name}] {found:,}/{stop - start:,} rows also in the base")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "train_overlap.csv"), index=False)
