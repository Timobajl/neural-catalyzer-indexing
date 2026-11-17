#!/bin/bash
#PBS -l select=1:ncpus=2:mem=28gb:ngpus=1:gpu_mem=20gb:scratch_ssd=180gb
#PBS -l walltime=06:00:00
# Usage: qsub -v ETA=120 np_scripts/eta/np_laion_eta.sh
# The thesis sweeps ETA over 40, 50, 60, 80, 100, 120, 140, 160, 170, 200, 220.

set -euo pipefail

ETA=${ETA:-120}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-partitioner"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/np/eta"}

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_laion_eta_${ETA}.log"
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
  --label "laion_eta${ETA}" \
  --eta-values "$ETA" \
  --consistency-only \
  --output-dir "$OUTPUT_DIR" \
  --n-bins-values 10000 \
  --hierarchical-branching 100,100 \
  --metric ip \
  --chunk-size 100000 \
  --consistency-cache "$CONSISTENCY_CACHE"

clean_scratch
