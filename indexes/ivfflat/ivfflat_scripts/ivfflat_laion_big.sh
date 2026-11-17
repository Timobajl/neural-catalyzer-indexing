#!/bin/bash
#PBS -l select=1:ncpus=32:mem=440gb:scratch_ssd=200gb
#PBS -l walltime=24:00:00
# Usage: qsub ivfflat_scripts/ivfflat_laion_big.sh

set -euo pipefail

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
LAION768="/storage/brno2/home/martykant/data/laion/catalyzer_d768_l0.0005/"
LAION_N=102144212

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"

SCRIPT_DIR="$REPO_DIR/indexes/ivfflat"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/ivfflat/laion"}

TARGET_ORIG="$SCRATCHDIR/original"
TARGET768="$SCRATCHDIR/768_0_0005"
GT_DIR="$SCRATCHDIR/gt"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_final_orig_768.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

check_rows () {
    python -c "import numpy as np, sys; n = np.load(sys.argv[1], mmap_mode='r').shape[0]; print(f'rows: {n:,}'); sys.exit(0 if n == int(sys.argv[2]) else 1)" \
      "$1" "$LAION_N"
}

# gt.npy is kept apart so it survives deleting the original from scratch.
mkdir -p "$GT_DIR"
cp "$LAION_ORIG/gt.npy" "$GT_DIR/gt.npy"

# ---------------------------------------------------------------
# 1. Original (not transformed)
# ---------------------------------------------------------------
check_rows "$LAION_ORIG/xb.npy"

START=$(date +"%s")
echo "Copy laion original"
cp -r "$LAION_ORIG" "$TARGET_ORIG"
echo "Copy time: $(($(date +"%s") - START))s"

echo "Running IVFFlat Index with laion original"
/usr/bin/time -v python -u ivfflat_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$GT_DIR/gt.npy" \
  --label original \
  --mode both \
  --no-save-index \
  --output-dir "$OUTPUT_DIR" \
  --metric ip \
  --consistency-cache "$CONSISTENCY_CACHE"

echo "Cleaning up scratch for original"
rm -rf "$TARGET_ORIG"

# ---------------------------------------------------------------
# 2. Catalyzer 768D, lambda* = 0.0005
# ---------------------------------------------------------------
check_rows "$LAION768/xbt.npy"

START=$(date +"%s")
echo "Copy laion 768 for lambda 0.0005"
cp -r "$LAION768" "$TARGET768"
echo "Copy time: $(($(date +"%s") - START))s"

echo "Running IVFFlat Index with laion768 (lambda 0.0005)"
/usr/bin/time -v python -u ivfflat_index.py \
  --xb "$TARGET768/xbt.npy" \
  --xq "$TARGET768/xqt.npy" \
  --gt "$GT_DIR/gt.npy" \
  --label catalyzer768_0_0005 \
  --mode both \
  --no-save-index \
  --output-dir "$OUTPUT_DIR" \
  --metric ip \
  --consistency-cache "$CONSISTENCY_CACHE"

echo "Cleaning up scratch for 768"
rm -rf "$TARGET768"

clean_scratch
