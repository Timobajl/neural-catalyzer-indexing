#!/bin/bash
#PBS -l select=1:ncpus=32:mem=480gb:scratch_ssd=200gb
#PBS -l walltime=24:00:00
# Usage: qsub lmi_scripts/lmi_laion_orig.sh

set -euo pipefail

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"

SCRIPT_DIR="$REPO_DIR/indexes/lmi"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/lmi/laion"}

TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_final_original.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

START=$(date +"%s")
echo "Copy laion original"
cp -r "$LAION_ORIG" "$TARGET_ORIG"
echo "Copy time: $(($(date +"%s") - START))s"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

echo "Running LMI with laion original"
/usr/bin/time -v python -u lmi_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label original \
  --output-dir "$OUTPUT_DIR" \
  --metric ip \
  --consistency-cache "$CONSISTENCY_CACHE" \
  --chunk-size 100000

clean_scratch