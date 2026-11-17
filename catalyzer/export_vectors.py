"""
Export original and catalyzer-transformed vectors for the index experiments.

Loads a trained catalyzer checkpoint and the dataset in test mode and writes
  <output-dir>/<db>/original/                      xb.npy, xq.npy, gt.npy, xt.npy
  <output-dir>/<db>/catalyzer_d<dout>_l<lambda>/   xbt.npy, xqt.npy
The original vectors are shared by all catalyzer variants and are written
only once.
"""
import argparse
import os
from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn

from lib.data import load_dataset, LaionLazyLoader
from lib.net import Normalize, forward_pass


def base_shape(xb) -> tuple[int, int]:
    return tuple(int(v) for v in xb.shape)


def validate_loaded_dataset(db: str, xb, xq: np.ndarray, gt: np.ndarray) -> None:
    n_base, dim_base = base_shape(xb)
    print(f"  loaded xb shape=({n_base:,}, {dim_base})")
    print(f"  loaded xq shape={xq.shape}")
    print(f"  loaded gt shape={gt.shape}  min={int(gt.min())}  max={int(gt.max())}")

    if xq.shape[1] != dim_base:
        raise ValueError(f"xq dimension {xq.shape[1]} does not match xb dimension {dim_base}")
    if gt.min() < 0:
        raise ValueError("Ground truth must be 0-based for FAISS, but contains negative IDs.")
    if gt.max() >= n_base:
        raise ValueError(
            f"Ground truth max ID {int(gt.max())} is outside xb length {n_base}."
        )
    if db == "laion":
        print("  LAION GT validated as 0-based and within exported base-vector range.")


def validate_existing_array(path: str, expected: np.ndarray, name: str) -> None:
    existing = np.load(path, mmap_mode='r')
    if existing.shape != expected.shape:
        raise ValueError(
            f"Existing {name} at {path} has shape {existing.shape}, expected {expected.shape}."
        )
    if existing.dtype != expected.dtype:
        raise ValueError(
            f"Existing {name} at {path} has dtype {existing.dtype}, expected {expected.dtype}."
        )
    if not np.array_equal(np.asarray(existing), expected):
        raise ValueError(
            f"Existing {name} at {path} differs from freshly loaded data."
        )


def save_chunked_original(xb_lazy: LaionLazyLoader, out_path: str) -> None:
    """
    Saves the original float16 vectors from the lazy loader directly to a
    float16 .npy file.  No float32 cast — the source dtype is kept to
    halve disk usage (about 157 GB instead of 314 GB for the 102 M × 768 LAION file).
    """
    n_total, dim_in = xb_lazy.shape
    print(f"Saving {n_total:,} original vectors (float16) -> {out_path}")

    mm = np.lib.format.open_memmap(
        out_path, mode='w+', dtype='float16', shape=(n_total, dim_in)
    )
    for chunk, start, stop in xb_lazy.iter_chunks():
        mm[start:stop] = chunk   # chunk is already float16
        # Write the memory-mapped output to disk every 10 chunks, not only at the end.
        if ((stop // args.chunk_size) % 10) == 0:
            mm.flush()
    mm.flush()
    del mm
    print(f"  -> saved  shape=({n_total}, {dim_in})  dtype=float16")


def save_chunked_transformed(xb_lazy: LaionLazyLoader, net, out_path: str) -> None:
    """
    Runs the network over the lazy-loaded base vectors chunk by chunk and
    saves the float16 output to a .npy file via memory-mapped writes.
    """
    n_total, _ = xb_lazy.shape
    print(f"Transforming {n_total:,} base vectors in chunks of {args.chunk_size:,} ...")
    print(f"  output: {out_path}  shape=({n_total}, {dout})  dtype=float16")

    mm = np.lib.format.open_memmap(
        out_path, mode='w+', dtype='float16', shape=(n_total, dout)
    )
    for chunk, start, stop in xb_lazy.iter_chunks():
        chunk_f32 = chunk.astype('float32')
        out = forward_pass(net, chunk_f32, bs=1024, device=args.device)
        mm[start:stop] = out.astype('float16')
        # Write the memory-mapped output to disk every 10 chunks, not only at the end.
        if ((stop // args.chunk_size) % 10) == 0:
            mm.flush()
    mm.flush()
    del mm
    print(f"  -> saved  ({n_total:,} vectors)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckpt-path",  type=str, required=True,
                    help="Path to the .pth checkpoint (e.g. checkpoints/bigann64/0.0002/checkpoint.pth.best)")
parser.add_argument("--database",   type=str, default="bigann",
                    choices=["bigann", "deep1b", "laion"],
                    help="Dataset to transform")
parser.add_argument("--device",     default="auto",
                    help="Device for inference: cpu, cuda, or auto")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Base output directory. Both originals and transformed "
                         "vectors are written under this path. "
                         "Default: ./data (relative to cwd). "
                         "On Metacentrum use absolute path, e.g. "
                         "/storage/brno2/home/user/data")

# LAION-specific
parser.add_argument("--size-base",  type=int, default=int(1e8),
                    help="Number of base vectors to export for laion "
                         "(default: 100 M). Use 1_000_000 or 10_000_000 for smaller runs.")
parser.add_argument("--chunk-size", type=int, default=500_000,
                    help="Vectors per chunk for chunked processing "
                         "(500 K × 768 × 2 B ≈ 0.77 GB as float16, "
                         "1.5 GB after the float32 cast for the network)")

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {args.device}")

print(f"Loading checkpoint: {args.ckpt_path}")
ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=False)

# Copy training args that are NOT already set by the CLI (e.g. dint, dout)
for k, v in vars(ckpt['args']).items():
    vars(args).setdefault(k, v)

# ---------------------------------------------------------------------------
# Derive output paths from checkpoint metadata
# ---------------------------------------------------------------------------

dout = ckpt['args'].dout
lam = format(Decimal(str(ckpt['args'].lambda_uniform)), "f")
db = args.database

base_dir = args.output_dir or "data"
orig_dir = os.path.join(base_dir, db, "original")
catalyzer_dir = os.path.join(base_dir, db, f"catalyzer_d{dout}_l{lam}")

print(f"  Originals   : {orig_dir}/")
print(f"  Transformed : {catalyzer_dir}/")

os.makedirs(orig_dir, exist_ok=True)
os.makedirs(catalyzer_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

load_kwargs = {}
if args.database == "laion":
    load_kwargs["chunk_size"] = args.chunk_size
    size = args.size_base
else:
    size = int(1e6)

print(f"Loading {args.database} (size={size:,}) ...")
xt, xb, xq, gt = load_dataset(args.database, args.device, size=size, test=True, **load_kwargs)
validate_loaded_dataset(args.database, xb, xq, gt)

dim  = xq.shape[1]
dint = args.dint

# ---------------------------------------------------------------------------
# Rebuild network
# ---------------------------------------------------------------------------

net = nn.Sequential(
    nn.Linear(in_features=dim, out_features=dint, bias=True),
    nn.BatchNorm1d(dint),
    nn.ReLU(),
    nn.Linear(in_features=dint, out_features=dint, bias=True),
    nn.BatchNorm1d(dint),
    nn.ReLU(),
    nn.Linear(in_features=dint, out_features=dout, bias=True),
    Normalize()
)

net.load_state_dict(ckpt['state_dict'])
net.to(args.device)
net.eval()

# ---------------------------------------------------------------------------
# Save original vectors  (shared across all catalyzer variants)
# ---------------------------------------------------------------------------

xb_orig_path = os.path.join(orig_dir, "xb.npy")
xq_orig_path = os.path.join(orig_dir, "xq.npy")
gt_orig_path = os.path.join(orig_dir, "gt.npy")
xt_orig_path = os.path.join(orig_dir, "xt.npy")

if not os.path.exists(xq_orig_path):
    np.save(xq_orig_path, xq)
    print(f"  -> saved xq.npy  shape={xq.shape}")
else:
    validate_existing_array(xq_orig_path, xq, "xq.npy")
    print(f"  xq.npy already exists and matches freshly loaded queries")

if xt is None:
    print(f"  xt is None, skipping xt.npy")
elif not os.path.exists(xt_orig_path):
    np.save(xt_orig_path, xt.astype('float32'))
    print(f"  -> saved xt.npy  shape={xt.shape}")
else:
    existing_xt = np.load(xt_orig_path, mmap_mode='r')
    if existing_xt.shape != base_shape(xt):
        raise ValueError(
            f"Existing xt.npy at {xt_orig_path} has shape {existing_xt.shape}, expected {base_shape(xt)}."
        )
    print(f"  xt.npy already exists with expected shape, skipping")

if not os.path.exists(gt_orig_path):
    np.save(gt_orig_path, gt)
    print(f"  -> saved gt.npy")
else:
    validate_existing_array(gt_orig_path, gt, "gt.npy")
    print(f"  gt.npy already exists and matches freshly loaded ground truth")

if not os.path.exists(xb_orig_path):
    if isinstance(xb, LaionLazyLoader):
        # LAION 100M: stream float16 chunks directly, the full array does not fit in RAM
        save_chunked_original(xb, xb_orig_path)
    else:
        # BigANN and Deep1B fit in RAM. LAION ≤ 10 M is stored as float16 like the 100 M export.
        dtype_out = 'float16' if args.database == "laion" else 'float32'
        np.save(xb_orig_path, xb.astype(dtype_out))
        print(f"  -> saved xb.npy  shape={xb.shape}  dtype={dtype_out}")
else:
    existing_xb = np.load(xb_orig_path, mmap_mode='r')
    expected_shape = base_shape(xb)
    if existing_xb.shape != expected_shape:
        raise ValueError(
            f"Existing xb.npy at {xb_orig_path} has shape {existing_xb.shape}, expected {expected_shape}."
        )
    print(f"  xb.npy already exists with expected shape, skipping")

# ---------------------------------------------------------------------------
# Save transformed vectors  (catalyzer-specific)
# ---------------------------------------------------------------------------

xqt = forward_pass(net, xq, bs=1024, device=args.device)
np.save(os.path.join(catalyzer_dir, "xqt.npy"), xqt)
print(f"  -> saved xqt.npy  shape=({len(xq)}, {dout})")

xbt_path = os.path.join(catalyzer_dir, "xbt.npy")

if isinstance(xb, LaionLazyLoader):
    save_chunked_transformed(xb, net, xbt_path)
else:
    xbt = forward_pass(net, xb, bs=1024, device=args.device)
    print(f"  avg L2 norm (sanity): {np.linalg.norm(xbt, axis=1).mean():.4f}")

    dtype_out = 'float16' if args.database == "laion" else 'float32'
    np.save(xbt_path, xbt.astype(dtype_out))
    print(f"  -> saved xbt.npy  shape={xbt.shape}  dtype={dtype_out}")

print(f"\nDone.")
print(f"  Transformed : {catalyzer_dir}/")
print(f"  Originals   : {orig_dir}/")
