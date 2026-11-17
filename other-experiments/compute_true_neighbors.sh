#!/bin/bash
#PBS -l select=1:ncpus=32:mem=12gb:scratch_ssd=180gb
#PBS -l walltime=6:00:00
# Usage: qsub compute_true_neighbors.sh

set -euo pipefail

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/cache"}

TARGET_ORIG="$SCRATCHDIR/original"


export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p $OUTPUT_DIR

module add mambaforge

LOG="$OUTPUT_DIR/stdout.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1


echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

START=$(date +"%s")
echo "Copy laion original"
cp -r "$LAION_ORIG" "$TARGET_ORIG"
echo "Copy time: $(($(date +"%s") - START))s"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd $SCRIPT_DIR


/usr/bin/time -v python -u compute_true_neighbors.py \
    --xb-orig $TARGET_ORIG/xb.npy \
    --metric ip \
    --num-samples 10000 \
    --output $OUTPUT_DIR/laion_true_neighbors.npz

