"""
Counts exact duplicate rows in a base vector file.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd


def hash_rows(path: str, chunk_size: int) -> np.ndarray:
    n = np.load(path, mmap_mode="r").shape[0]
    hashes = np.empty(n, dtype=np.uint64)
    for start in range(0, n, chunk_size):
        chunk = np.array(np.load(path, mmap_mode="r")[start:start + chunk_size])
        digests = b"".join(hashlib.blake2b(row, digest_size=8).digest() for row in chunk)
        hashes[start:start + len(chunk)] = np.frombuffer(digests, dtype=np.uint64)
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count exact duplicate rows")
    parser.add_argument("--xb", required=True, help="Path to base vectors (xb.npy)")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--output-dir", default="duplicates_results")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    n, d = np.load(args.xb, mmap_mode="r").shape

    hashes = hash_rows(args.xb, args.chunk_size)
    values, counts = np.unique(hashes, return_counts=True)

    largest_ids = np.flatnonzero(hashes == values[counts.argmax()])
    largest_rows = np.load(args.xb, mmap_mode="r")[largest_ids]

    row = {
        "label": args.label, "n": n, "dim": d,
        "distinct_rows": len(values),
        "rows_with_a_copy": int(counts[counts > 1].sum()),
        "duplicate_groups": int((counts > 1).sum()),
        "largest_group": int(counts.max()),
        "largest_group_identical": bool(np.all(largest_rows == largest_rows[0])),
        "largest_group_norm": float(np.linalg.norm(largest_rows[0].astype("float64"))),
        "largest_groups": ";".join(str(size) for size in np.sort(counts)[::-1][:20]),
    }

    print(f"[{args.label}] {row}")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame([row]).to_csv(os.path.join(args.output_dir, f"duplicates_{args.label}.csv"), index=False)
