#!/bin/bash
#PBS -l select=1:ncpus=32:mem=28gb:scratch_ssd=10gb:cluster=halmir
#PBS -l walltime=12:00:00
# Usage: qsub np_scripts/np_deep.sh

set -euo pipefail

PAIRS=(
    "16 0.02"
    "32 0.005"
    "64 0.0002"
    "96 0.0002"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-partitioner"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/np/deep1b"}

ORIG="/storage/brno2/home/martykant/data/deep1b/original/"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout.log"
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

run_np () {
    local LABEL="$1"; shift
    local XB="$1"; shift
    local XQ="$1"; shift

    echo "Running NP with $LABEL"
    /usr/bin/time -v python -u np_index.py \
      --xb "$XB" \
      --xq "$XQ" \
      --gt "$TARGET_ORIG/gt.npy" \
      --label "$LABEL" \
      --output-dir "$OUTPUT_DIR" \
      "$@"
}

run_np "deep_orig_16x16" "$TARGET_ORIG/xb.npy" "$TARGET_ORIG/xq.npy" \
  --n-bins-values 256 --hierarchical-branching 16,16 --eta 10

run_np "deep_orig_32x32" "$TARGET_ORIG/xb.npy" "$TARGET_ORIG/xq.npy" \
  --eta 24

for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = $DIM, LAMBDA = $LAMBDA"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    DATA="/storage/brno2/home/martykant/data/deep1b/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="$SCRATCHDIR/${DIM}_${LAMBDA_LABEL}"

    echo "Copy deep $DIM for lambda $LAMBDA"
    cp -r "$DATA" "$TARGET"

    run_np "catalyzer${DIM}_${LAMBDA_LABEL}_16x16" "$TARGET/xbt.npy" "$TARGET/xqt.npy" \
      --n-bins-values 256 --hierarchical-branching 16,16 --eta 10 \
      --xb-orig "$TARGET_ORIG/xb.npy"

    run_np "catalyzer${DIM}_${LAMBDA_LABEL}_32x32" "$TARGET/xbt.npy" "$TARGET/xqt.npy" \
      --eta 24 \
      --xb-orig "$TARGET_ORIG/xb.npy"

    rm -rf "$TARGET"
done

clean_scratch
