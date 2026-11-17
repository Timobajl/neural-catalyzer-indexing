#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:scratch_ssd=200gb
#PBS -l walltime=02:00:00
# Usage: qsub duplicates_scripts/train_overlap_laion.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR="$REPO_DIR/results/duplicates/laion"
XB="/storage/brno2/home/martykant/data/laion/original/xb.npy"
H5="/storage/brno2/home/martykant/datasets/laion/laion2B-en-clip768v2-n=10M.h5"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_overlap.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy laion base and 10M file"
cp "$XB" "$SCRATCHDIR/xb.npy"
cp "$H5" "$SCRATCHDIR/10M.h5"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

/usr/bin/time -v python -u check_train_overlap.py \
  --xb "$SCRATCHDIR/xb.npy" \
  --h5 "$SCRATCHDIR/10M.h5" \
  --output-dir "$OUTPUT_DIR"

clean_scratch
