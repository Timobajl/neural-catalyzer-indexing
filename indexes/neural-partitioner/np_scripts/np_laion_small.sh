#!/bin/bash
#PBS -l select=1:ncpus=32:mem=200gb:scratch_ssd=50gb:cluster=halmir
#PBS -l walltime=24:00:00
# Usage: qsub np_scripts/np_laion_small.sh

set -euo pipefail

PAIRS=(
    "16 0.1"
    "32 0.001"
    "64 0.0002"
    "128 0.001"
)
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

LOG="$OUTPUT_DIR/stdout.log"
echo "Logging output to $LOG"
exec > "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

# Only gt.npy from the original: consistency reads the cached original-space neighbors.
echo "Copy laion original gt"
mkdir -p "$TARGET_ORIG"
cp "$LAION_ORIG/gt.npy" "$TARGET_ORIG/gt.npy"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = $DIM, LAMBDA = $LAMBDA"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    LAION="/storage/brno2/home/martykant/data/laion/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="$SCRATCHDIR/${DIM}_${LAMBDA_LABEL}"

    echo "Copy laion $DIM for lambda $LAMBDA"
    cp -r "$LAION" "$TARGET"

    /usr/bin/time -v python -u np_index.py \
      --xb "$TARGET/xbt.npy" \
      --xq "$TARGET/xqt.npy" \
      --gt "$TARGET_ORIG/gt.npy" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}_100x100" \
      --output-dir "$OUTPUT_DIR" \
      --metric ip \
      --n-bins-values 10000 \
      --hierarchical-branching 100,100 \
      --eta "$ETA" \
      --consistency-cache "$CONSISTENCY_CACHE" \
      --chunk-size 100000

    rm -rf "$TARGET"
done

clean_scratch
