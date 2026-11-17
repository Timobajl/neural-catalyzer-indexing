#!/bin/bash
#PBS -l select=1:ncpus=32:mem=480gb:scratch_ssd=200gb
#PBS -l walltime=48:00:00
# Usage: qsub hnsw_scripts/hnsw_laion_orig.sh

set -euo pipefail

LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/hnsw"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/hnsw/laion"}

TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_final_orig.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1


echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy laion original"
cp -r "$LAION_ORIG" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

echo "Running HNSW with laion original"
/usr/bin/time -v python -u hnsw_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label original \
  --mode both \
  --no-save-index \
  --efSearch-values 30,40,50,60,80,100,150,200,300,400,500 \
  --output-dir "$OUTPUT_DIR" \
  --metric ip

clean_scratch
