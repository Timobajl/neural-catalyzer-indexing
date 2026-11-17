# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from os.path import join
import numpy as np
from .metrics import get_nearestneighbors, sanitize
import os
import h5py


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def mmap_fbin(fname):
    with open(fname, "rb") as f:
        _, dim = np.fromfile(f, count=2, dtype=np.int32)
    file_size = os.path.getsize(fname)
    actual_n_vecs = (file_size - 8) // (dim * 4)
    x = np.memmap(fname, dtype='float32', mode='r', offset=8)
    return x[:actual_n_vecs * dim].reshape(actual_n_vecs, dim).copy()


def mmap_ibin(fname):
    with open(fname, "rb") as f:
        _, dim = np.fromfile(f, count=2, dtype=np.int32)
    file_size = os.path.getsize(fname)
    actual_n_vecs = (file_size - 8) // (dim * 4)
    x = np.memmap(fname, dtype='int32', mode='r', offset=8)
    return x[:actual_n_vecs * dim].reshape(actual_n_vecs, dim)

def getBasedir(s):
    if "BIGANN_PATH" in os.environ and s == "bigann":
        return os.environ["BIGANN_PATH"]
    if "DEEP1B_PATH" in os.environ and s == "deep1b":
        return os.environ["DEEP1B_PATH"]
    if "LAION_PATH" in os.environ and s == "laion":
        return os.environ["LAION_PATH"]

    paths = {
        "bigann": "../datasets",
    }

    return paths[s]

# Original Deep1B loader of the reference implementation (.fvecs files),
# kept for reference. load_deep1b below reads the .fbin files used in this
# thesis and keeps the same train/test split.
# def load_deep1b(device, size = 10 ** 6, test=True, qsize=10 ** 5):
#     basedir = getBasedir("deep1b") + '/'
#     xt = mmap_fvecs(basedir + 'learn.fvecs')
#     if test:
#         xb = mmap_fvecs(basedir + 'base.fvecs')
#         xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
#         xb = xb[:size]

#         gt = ivecs_read(basedir + 'deep%s_groundtruth.ivecs' % (
#             '1M' if size == 10 ** 6 else
#             '10M' if size == 10 ** 7 else
#             '100M' if size == 10 ** 8 else
#             '1B' if size == 10 ** 9 else 1/0
#         ))
#     else:
#         xb = xt[:size]
#         xq = xt[size:size+qsize]
#         xt = xt[size+qsize:]

#     xb, xq = sanitize(xb), sanitize(xq)
#     if not test:
#         gt = get_nearestneighbors(xq, xb, 100, device)

#     return xt, xb, xq, gt

def load_deep1b(device, size = 10 ** 6, test=True, qsize=10 ** 5):
    basedir = getBasedir("deep1b")

    xt = mmap_fbin(join(basedir, 'learn.350M.fbin'))
    if test:
        xb = mmap_fbin(join(basedir, 'base.1B.fbin'))
        xq = mmap_fbin(join(basedir, 'query.public.10K.fbin'))
        xb = xb[:size]

        if size == 10 ** 9:
            gt = mmap_ibin(join(basedir, 'deep_new_groundtruth.public.10K.bin'))
        else:
            # The official ground truth covers the full 1B base, so a subset
            # needs its own, computed below.
            gt = None
    else:
        xb = xt[:size]
        xq = xt[size:size+qsize]
        xt = xt[size+qsize:]
        gt = None

    xb, xq = sanitize(xb), sanitize(xq)
    if gt is None:
        print("Computing groundtruth on the fly for subset...")
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


def load_bigann(device, size = 10 ** 6, test=True, qsize=10 ** 5):
    basedir = getBasedir("bigann")

    dbsize = int(size / 10 ** 6)
    xt = mmap_bvecs(join(basedir, 'bigann_learn.bvecs'))
    if test:
        xb = mmap_bvecs(join(basedir, 'bigann_base.bvecs'))
        xq = mmap_bvecs(join(basedir, 'bigann_query.bvecs'))
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        gt = ivecs_read(join(basedir, 'gnd/idx_%dM.ivecs' % dbsize))
    else:
        xb = xt[:size]
        xq = xt[size:size+qsize]
        xt = xt[size+qsize:]

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


class LaionLazyLoader:
    """
    Chunk-based loader for laion base vectors stored in an HDF5 file.

    The 100 M × 768-D float16 database is too large to materialize fully in RAM.
    Instead this object exposes an `iter_chunks()` generator that yields
    ``(chunk_array, start_idx, stop_idx)`` tuples so callers can process
    the data incrementally and write results directly to a memory-mapped
    output file.

    The object also exposes `.shape` and `__len__` so that downstream code
    can query sizes without loading data.
    """

    def __init__(self, path: str, n_vectors: int, chunk_size: int = 500_000):
        """
        Args:
            path:       Path to the HDF5 file (must contain an 'emb' dataset).
            n_vectors:  Number of vectors to expose (may be < file length).
            chunk_size: Vectors per iteration step.  500 K × 768 × 2 B ≈ 0.77 GB
        """
        self.path = path
        self.n_vectors = n_vectors
        self.chunk_size = chunk_size
        self._dim: int | None = None


    def _get_dim(self) -> int:
        if self._dim is None:
            with h5py.File(self.path, 'r') as f:
                self._dim = int(f['emb'].shape[1])
        return self._dim

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_vectors, self._get_dim())

    def __len__(self) -> int:
        return self.n_vectors

    def iter_chunks(self):
        """
        Yields ``(chunk, start, stop)`` where:
          - ``chunk``  is a contiguous float16 ndarray of shape ``(stop-start, dim)``
          - ``start``  is the inclusive global index of the first vector in the chunk
          - ``stop``   is the exclusive global end index (one past the last vector)
        Yields raw float16 — caller casts to float32 for forward_pass if needed
        """
        with h5py.File(self.path, 'r') as f:
            emb = f['emb']
            for start in range(0, self.n_vectors, self.chunk_size):
                stop = min(start + self.chunk_size, self.n_vectors)
                chunk = np.array(emb[start:stop])  # native float16, contiguous copy
                print(f"  [LaionLazyLoader] processed {stop:,} / {self.n_vectors:,} vectors")
                yield chunk, start, stop


# Maps an upper-bound size to (GT-size-string, HDF5 filename).
# Sizes above 10 M are served from the 100 M file. Everything up to 10 M
# is served from the 10 M file, which is fast to read and fits in ~32 GB RAM.
SISAP24_DB_FILES = {
      300_000: ("300K", "laion2B-en-clip768v2-n=10M.h5"),
    1_000_000: ("1M",   "laion2B-en-clip768v2-n=10M.h5"),
   10_000_000: ("10M",  "laion2B-en-clip768v2-n=10M.h5"),
  100_000_000: ("100M", "laion2B-en-clip768v2-n=100M.h5"),
}

QUERIES_FILE = "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
GT_FILE_TMPL  = "gold-standard-dbsize={size}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"

def h5_emb_count(path: str) -> int:
    with h5py.File(path, 'r') as f:
        return int(f['emb'].shape[0])


def laion_resolve(size: int, basedir: str) -> tuple[str, str, int]:
    """
    Given a requested size, return
    ``(gt_size_str, db_filename, n_vectors)`` for the closest catalogue entry.

    Important: the official SISAP "100M" HDF5 file contains slightly more than
    100,000,000 rows. The official 100M ground truth is defined against that
    full file, so this object must expose the actual HDF5 length instead of truncating
    to exactly 100,000,000.
    """
    for threshold, (gt_str, db_file) in sorted(SISAP24_DB_FILES.items()):
        if size <= threshold:
            if gt_str == "100M":
                return gt_str, db_file, h5_emb_count(join(basedir, db_file))
            return gt_str, db_file, size
    # Larger than anything in the map -> use the full official 100M file.
    db_file = "laion2B-en-clip768v2-n=100M.h5"
    return "100M", db_file, h5_emb_count(join(basedir, db_file))

def load_laion_queries(path: str) -> np.ndarray:
    """Loads 10k LAION queries fully into RAM """
    with h5py.File(path, 'r') as f:
        return f['emb'][:]

def load_laion_ground_truth(path: str, k = 100) -> np.ndarray:
    """Loads 0-based ground-truth knns[:, :k] from a gold-standard .h5 file."""
    with h5py.File(path, 'r') as f:
        gt = f['knns'][:, :k].astype('int64') - 1
    if gt.min() < 0:
        raise ValueError(
            f"LAION ground truth at {path} contains IDs below 1. "
            "expected official SISAP 1-based IDs."
        )
    return gt

def load_laion_database(path: str, n_load = 2_000_000) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f['emb'][:n_load]

def load_laion(device, size = 10 ** 6, test=True, qsize= 10 ** 5, chunk_size = 500_000):
    """
    Load the SISAP 2024 LAION dataset.

    Returns ``(xt, xb, xq, gt)`` — the same tuple contract as
    ``load_bigann`` / ``load_deep1b``.

    test=True  (eval / export mode)
    ────────────────────────────────
    ``xb`` contains (or lazily exposes) all ``size`` base vectors.
    When size > 10 M, ``xb`` is a :class:`LaionLazyLoader` instead of a
    plain ndarray — callers that cannot handle this should pass size ≤ 10 M.
    ``xt`` is a small slice of the base data, and None for the 100 M file.

    test=False  (training mode)
    ───────────────────────────
    Unlike BigANN and Deep1B, LAION has no separate learn split, only the
    base vectors, the public queries and their ground truth. Training
    therefore splits the first 2 M vectors of the 10 M file, and the 100 M
    file is not needed.
    • ``xb``  — first 1 M vectors, the validation base
    • ``xq``  — next 100 K vectors, the validation queries
    • ``xt``  — remaining 900 K vectors, of which train.py keeps num_learn (500 K)
    • ``gt``  — exact ground truth of ``xq`` against ``xb``

    Environment variables
    ─────────────────────
    Set ``LAION_PATH`` to the directory that contains the HDF5 files.
    """
    basedir = getBasedir("laion")


    if test:
        gt_size_str, db_filename, n_vectors = laion_resolve(int(size), basedir)
        db_path      = join(basedir, db_filename)
        queries_path = join(basedir, QUERIES_FILE)
        gt_path      = join(basedir, GT_FILE_TMPL.format(size=gt_size_str))

        print(f"Loading queries from {queries_path}")
        xq = load_laion_queries(queries_path)
        print(f"Loading ground truth from {gt_path}")
        gt = load_laion_ground_truth(gt_path)

        if n_vectors > 10_000_000:
            # ── 100 M path: use lazy loader, never load everything at once ──
            print(f"{n_vectors:,} vectors requested — "
                  "returning LaionLazyLoader (chunked access)")
            xb = LaionLazyLoader(db_path, n_vectors, chunk_size=chunk_size)
            xt = None   # not consumed in export_vectors.py
        else:
            # ── ≤ 10 M path: fits in RAM ──
            print(f"Loading {n_vectors:,} base vectors from {db_path}")
            xb = load_laion_database(db_path, n_vectors)
            xt = xb[:min(50_000, n_vectors)]

    else:
        # ── Training mode ──────────────────────────────────────────────
        # Training reads only the first 2 M vectors of the 10 M file.
        train_db_path = join(basedir, "laion2B-en-clip768v2-n=10M.h5")
        n_load = 2_000_000

        print(f"Training mode — loading {n_load:,} vectors from {train_db_path}")
        data = load_laion_database(train_db_path, n_load)

        # Define split sizes
        n_xb = min(size, 1_000_000)

        xb = data[:n_xb]
        xq = data[n_xb : n_xb + qsize]
        xt = data[n_xb + qsize :]

    if not isinstance(xb, LaionLazyLoader):
        xb = sanitize(xb)
    xq = sanitize(xq)

    if not test:
        print(f"Computing synthetic ground truth ({qsize} queries vs {n_xb} base) …")
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt

def load_dataset(name, device, size=10**6, test=True, **kwargs):
    if name == "bigann":
        return load_bigann(device, size, test)
    elif name == "deep1b":
        return load_deep1b(device, size, test)
    elif name == "laion":
        return load_laion(device, size, test, **kwargs)
