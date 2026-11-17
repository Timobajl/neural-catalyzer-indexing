#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=02:00:00
# Usage: qsub verify_scripts/verify_knn.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/neural-lsh"
OUTPUT_DIR="$REPO_DIR/results/knn_equivalence"
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
# utils.py of NeuralLSH reads 'config' from the working directory
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

for DB in bigann deep1b laion; do
    /usr/bin/time -v python -u "$REPO_DIR/other-experiments/verify_knn_equivalence.py" \
      --xb "$DATA/$DB/original/xb.npy" \
      --label "$DB"
done
