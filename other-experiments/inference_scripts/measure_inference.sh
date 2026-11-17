#!/bin/bash
#PBS -l select=1:ncpus=32:mem=8gb:cluster=halmir
#PBS -l walltime=00:20:00
# Usage: qsub inference_scripts/measure_inference.sh

set -euo pipefail

MODELS=(
    "bigann 16 0.01"
    "bigann 32 0.0007"
    "bigann 64 0.0002"
    "bigann 128 0.0001"
    "deep1b 16 0.02"
    "deep1b 32 0.005"
    "deep1b 64 0.0002"
    "deep1b 96 0.0002"
    "laion 16 0.1"
    "laion 32 0.001"
    "laion 64 0.0002"
    "laion 128 0.001"
    "laion 768 0.0005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR="$REPO_DIR/results/inference"
DATA="/storage/brno2/home/martykant/data"

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

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_DIR/catalyzer"

for model in "${MODELS[@]}"; do
    read -r DB DIM LAMBDA <<< "$model"

    /usr/bin/time -v python -u measure_inference.py \
      --ckpt "$REPO_DIR/catalyzer/checkpoints/$DB$DIM/$LAMBDA/checkpoint.pth.best" \
      --xq "$DATA/$DB/original/xq.npy" \
      --label "${DB}_d${DIM}_l${LAMBDA}" \
      --output-dir "$OUTPUT_DIR"
done
