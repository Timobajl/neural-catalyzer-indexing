#!/bin/bash
#PBS -l select=1:ncpus=32:mem=64gb:scratch_ssd=200gb
#PBS -l walltime=08:00:00
# Usage: qsub knn_scripts/knn_laion_full.sh

set -euo pipefail

PAIRS=(
    "16 0.1"
    "32 0.001"
    "64 0.0002"
    "128 0.001"
    "768 0.0005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR="$REPO_DIR/results/knn/laion"
DATA="/storage/brno2/home/martykant/data/laion"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR" "$TARGET_ORIG"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_full.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

cp "$DATA/original/gt.npy" "$DATA/original/xq.npy" "$TARGET_ORIG/"

run_knn () {
    local XB="$1"
    local XQ="$2"
    local LABEL="$3"

    /usr/bin/time -v python -u knn_chunked.py \
      --xb "$XB" \
      --xq "$XQ" \
      --gt "$TARGET_ORIG/gt.npy" \
      --label "$LABEL" \
      --metric ip \
      --output-dir "$OUTPUT_DIR"
}

for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"
    LAMBDA_LABEL="${LAMBDA//./_}"
    TARGET="$SCRATCHDIR/${DIM}_${LAMBDA_LABEL}"

    echo "Copy laion $DIM for lambda $LAMBDA"
    cp -r "$DATA/catalyzer_d${DIM}_l${LAMBDA}" "$TARGET"

    run_knn "$TARGET/xbt.npy" "$TARGET/xqt.npy" "catalyzer${DIM}_${LAMBDA_LABEL}"

    rm -rf "$TARGET"
done

echo "Copy laion original"
cp "$DATA/original/xb.npy" "$TARGET_ORIG/"
run_knn "$TARGET_ORIG/xb.npy" "$TARGET_ORIG/xq.npy" original

clean_scratch
