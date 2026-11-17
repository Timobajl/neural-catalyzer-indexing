#!/bin/bash
#PBS -l select=1:ncpus=32:mem=20gb:scratch_ssd=5gb:cluster=halmir
#PBS -l walltime=05:00:00
# Usage: qsub rerank_scripts/rerank_bigann.sh

set -euo pipefail

PAIRS=(
    "16 0.01"
    "32 0.0007"
    "64 0.0002"
    "128 0.0001"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR="$REPO_DIR/results/rerank/bigann"
DATA="/storage/brno2/home/martykant/data/bigann"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy bigann original"
cp -r "$DATA/original" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_DIR/indexes/ivfflat:$REPO_DIR/indexes/lmi"

run_rerank () {
    local XB="$1"
    local XQ="$2"
    local LABEL="$3"

    for ROUTER in ivfflat lmi; do
        echo "Running $ROUTER with $LABEL"
        /usr/bin/time -v python -u rerank_original.py \
          --router "$ROUTER" \
          --xb "$XB" \
          --xq "$XQ" \
          --xb-orig "$TARGET_ORIG/xb.npy" \
          --xq-orig "$TARGET_ORIG/xq.npy" \
          --gt "$TARGET_ORIG/gt.npy" \
          --label "$LABEL" \
          --output-dir "$OUTPUT_DIR"
    done
}

run_rerank "$TARGET_ORIG/xb.npy" "$TARGET_ORIG/xq.npy" original

for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"
    LAMBDA_LABEL="${LAMBDA//./_}"
    TARGET="$SCRATCHDIR/${DIM}_${LAMBDA_LABEL}"

    echo "Copy bigann $DIM for lambda $LAMBDA"
    cp -r "$DATA/catalyzer_d${DIM}_l${LAMBDA}" "$TARGET"

    run_rerank "$TARGET/xbt.npy" "$TARGET/xqt.npy" "catalyzer${DIM}_${LAMBDA_LABEL}"

    rm -rf "$TARGET"
done

clean_scratch
