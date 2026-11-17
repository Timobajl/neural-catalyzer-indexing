#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:scratch_ssd=200gb
#PBS -l walltime=06:00:00
# Usage: qsub duplicates_scripts/duplicates_laion.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR="$REPO_DIR/results/duplicates/laion"
DATA="/storage/brno2/home/martykant/data/laion"

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

echo "Copy laion original"
cp "$DATA/original/xb.npy" "$SCRATCHDIR/xb.npy"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

/usr/bin/time -v python -u count_duplicates.py \
  --xb "$SCRATCHDIR/xb.npy" \
  --label original \
  --output-dir "$OUTPUT_DIR"

clean_scratch
