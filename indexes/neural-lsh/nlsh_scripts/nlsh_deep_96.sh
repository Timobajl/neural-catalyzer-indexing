#!/bin/bash
#PBS -l select=1:ncpus=16:mem=36gb:scratch_ssd=15gb:cluster=halmir
#PBS -l walltime=16:00:00
# Usage: qsub nlsh_scripts/nlsh_deep_96.sh

set -euo pipefail

DIM=96
LAMBDA="0.0002"
LAMBDA_LABEL="${LAMBDA//./_}"

N_CLUSTERS_LIST=(16 32)
NPROBE_VALUES=${NPROBE_VALUES:-1,2,3,4,5,6,7,8}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-lsh"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/nlsh/deep1b"}

ORIG="/storage/brno2/home/martykant/data/deep1b/original/"
DATA="/storage/brno2/home/martykant/data/deep1b/catalyzer_d${DIM}_l${LAMBDA}/"

TARGET_ORIG="$SCRATCHDIR/original"
TARGET="$SCRATCHDIR/deep${DIM}"
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

echo "Copy deep ${DIM} for lambda ${LAMBDA}"
cp -r "$DATA" "$TARGET"

echo "Copy deep original"
cp -r "$ORIG" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

for NC in "${N_CLUSTERS_LIST[@]}"; do
    echo "Running NeuralLSH with deep${DIM} (lambda ${LAMBDA}), ${NC}x${NC}"
    /usr/bin/time -v python -u nlsh_index.py \
      --xb "$TARGET/xbt.npy" \
      --xq "$TARGET/xqt.npy" \
      --gt "$TARGET_ORIG/gt.npy" \
      --dataset-name "deep" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}_${NC}x${NC}" \
      --metric "l2" \
      --output-dir "$OUTPUT_DIR" \
      --work-dir "$WORK_DIR" \
      --n-clusters-values "$NC" \
      --nprobe-values "$NPROBE_VALUES" \
      --xb-orig "$TARGET_ORIG/xb.npy"
done

cp -r "$WORK_DIR/models" "$OUTPUT_DIR/"
cp -r "$WORK_DIR/ranks" "$OUTPUT_DIR/"

clean_scratch
