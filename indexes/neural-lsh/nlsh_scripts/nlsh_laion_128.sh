#!/bin/bash
#PBS -l select=1:ncpus=16:mem=64gb:scratch_ssd=50gb
#PBS -l walltime=24:00:00
# Usage: qsub nlsh_scripts/nlsh_laion_128.sh

set -euo pipefail

DIM=128
LAMBDA="0.001"
LAMBDA_LABEL="${LAMBDA//./_}"

N_CLUSTERS_VALUES=${N_CLUSTERS_VALUES:-100}
NPROBE_VALUES=${NPROBE_VALUES:-1,2,3,4,5,6,7,8}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-lsh"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/nlsh/laion"}

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
LAION="/storage/brno2/home/martykant/data/laion/catalyzer_d${DIM}_l${LAMBDA}/"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"

TARGET_ORIG="$SCRATCHDIR/original"
TARGET="$SCRATCHDIR/laion${DIM}"
WORK_DIR="$SCRATCHDIR/nlsh_work"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_${DIM}.log"
echo "Logging output to $LOG"
exec > "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy laion ${DIM} for lambda ${LAMBDA}"
cp -r "$LAION" "$TARGET"

# Only gt.npy from the original: consistency reads the cached original-space neighbors.
echo "Copy original gt"
mkdir -p "$TARGET_ORIG"
cp "$LAION_ORIG/gt.npy" "$TARGET_ORIG/"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

/usr/bin/time -v python -u nlsh_index.py \
  --xb "$TARGET/xbt.npy" \
  --xq "$TARGET/xqt.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --dataset-name "laion" \
  --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
  --metric "ip" \
  --output-dir "$OUTPUT_DIR" \
  --work-dir "$WORK_DIR" \
  --n-clusters-values "$N_CLUSTERS_VALUES" \
  --nprobe-values "$NPROBE_VALUES" \
  --train-size 1000000 \
  --bottom-level km \
  --consistency-cache "$CONSISTENCY_CACHE"

cp -r "$WORK_DIR/models" "$OUTPUT_DIR/"
cp -r "$WORK_DIR/ranks" "$OUTPUT_DIR/"

clean_scratch
