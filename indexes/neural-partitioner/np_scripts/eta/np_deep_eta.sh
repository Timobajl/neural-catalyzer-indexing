#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb:ngpus=1:scratch_local=20gb
#PBS -l walltime=02:00:00
# Usage: qsub np_scripts/eta/np_deep_eta.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-partitioner"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/np/eta"}

ORIG="/storage/brno2/home/martykant/data/deep1b/original/"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_deep_eta.log"
echo "Logging output to $LOG"
exec > "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy deep original"
cp -r "$ORIG" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

/usr/bin/time -v python -u np_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label deep_eta \
  --eta-values 7,10,15,20,22,24,26,30 \
  --consistency-only \
  --output-dir "$OUTPUT_DIR"

clean_scratch
