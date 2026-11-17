#!/bin/bash
#PBS -l select=1:ncpus=32:mem=240gb:scratch_ssd=200gb:cluster=halmir
#PBS -l walltime=16:00:00
# Usage: qsub np_scripts/np_laion_orig.sh

set -euo pipefail

ETA=120

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-partitioner"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/np/laion"}

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_orig.log"
echo "Logging output to $LOG"
exec > "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy laion original"
cp -r "$LAION_ORIG" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

/usr/bin/time -v python -u np_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label "original" \
  --output-dir "$OUTPUT_DIR" \
  --metric ip \
  --n-bins-values 10000 \
  --hierarchical-branching 100,100 \
  --eta "$ETA" \
  --consistency-cache "$CONSISTENCY_CACHE"

clean_scratch
