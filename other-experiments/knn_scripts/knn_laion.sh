#!/bin/bash
#PBS -l select=1:ncpus=16:mem=44gb:scratch_ssd=10gb
#PBS -l walltime=10:00:00
# Usage: qsub knn_scripts/knn_laion.sh

set -euo pipefail

# Scores every swept pair. Uncomment all pairs in export_scripts/export_run_laion.sh and export them first.
PAIRS=(
    # --- 16D ---
    "16 0.05"
    "16 0.02"
    "16 0.01"
    "16 0.08"
    "16 0.1"
    "16 0.2"

    # --- 32D ---
    "32 0.005"
    "32 0.001"
    "32 0.0007"
    "32 0.0005"

    # --- 64D ---
    "64 0.001"
    "64 0.0005"
    "64 0.0002"
    "64 0.0001"

    # --- 128D ---
    "128 0.01"
    "128 0.0005"
    "128 0.0002"
    "128 0.0001"
    "128 0.00008"
    "128 0.001"
    "128 0.005"

    # --- 768D ---
    "768 0.0005"
    "768 0.0001"
    "768 0.00005"
    "768 0.00001"
    "768 0.000005"
    "768 0.001"
    "768 0.002"
    "768 0.005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/knn/laion"}
LAION_ORIG="/storage/brno2/home/martykant/data/laion/original"
GT_FILE="$SCRATCHDIR/gt_laion_1m.npy"
SCRIPT_DIR="$REPO_DIR/other-experiments"


export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"
module add mambaforge

LOG="$OUTPUT_DIR/stdout_knn.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

cd $SCRIPT_DIR

mamba activate /storage/brno2/home/martykant/conda_envs/bc

# 1. Original space, which also gives the ground truth of the 1M subset
echo "Running Exact k-NN with laion_original"
/usr/bin/time -v python -u knn.py \
  --xb "$LAION_ORIG/xb.npy" \
  --xq "$LAION_ORIG/xq.npy" \
  --label original \
  --output-dir "$OUTPUT_DIR" \
  --n-samples 1000000 \
  --gt-out "$GT_FILE" \
  --metric ip

# 2. Every (dout, lambda) pair, memory-mapped from storage
for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = ${DIM}, LAMBDA = ${LAMBDA}"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    CAT="/storage/brno2/home/martykant/data/laion/catalyzer_d${DIM}_l${LAMBDA}"

    /usr/bin/time -v python -u knn.py \
      --xb "${CAT}/xbt.npy" \
      --xq "${CAT}/xqt.npy" \
      --gt "$GT_FILE" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
      --output-dir "${OUTPUT_DIR}" \
      --n-samples 1000000 \
      --metric ip
done

clean_scratch
