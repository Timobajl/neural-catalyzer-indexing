#!/bin/bash
#PBS -l select=1:ncpus=32:mem=64gb:scratch_ssd=50gb
#PBS -l walltime=12:00:00
# Usage: qsub lmi_scripts/lmi_laion_32.sh

set -euo pipefail

DIM=32
LAMBDA="0.001"
LAMBDA_LABEL="${LAMBDA//./_}"

LAION="/storage/brno2/home/martykant/data/laion/catalyzer_d${DIM}_l${LAMBDA}/"
TARGET="$SCRATCHDIR/${DIM}_${LAMBDA_LABEL}"

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

LOG="$OUTPUT_DIR/stdout_final_32.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

START=$(date +"%s")
echo "Copy laion ${DIM} for lambda ${LAMBDA}"
cp -r "$LAION" "$TARGET"
echo "Copy time: $(($(date +"%s") - START))s"

# Only gt.npy from the original: consistency reads the cached original-space neighbors.
echo "Copy original gt"
mkdir -p "$TARGET_ORIG"
cp "$LAION_ORIG/gt.npy" "$TARGET_ORIG/"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

echo "Running LMI with laion${DIM} (lambda ${LAMBDA})"
/usr/bin/time -v python -u lmi_index.py \
  --xb "$TARGET/xbt.npy" \
  --xq "$TARGET/xqt.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
  --output-dir "$OUTPUT_DIR" \
  --metric ip \
  --consistency-cache "$CONSISTENCY_CACHE" \
  --chunk-size 200000

clean_scratch
