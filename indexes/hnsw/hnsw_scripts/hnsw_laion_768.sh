#!/bin/bash
#PBS -l select=1:ncpus=32:mem=480gb:scratch_ssd=200gb
#PBS -l walltime=48:00:00
# Usage: qsub hnsw_scripts/hnsw_laion_768.sh

set -euo pipefail

LAION768="/storage/brno2/home/martykant/data/laion/catalyzer_d768_l0.0005/"
LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/hnsw"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/hnsw/laion"}

TARGET768="$SCRATCHDIR/768"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_final_768.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1


echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy laion catalyzer 768D"
cp -r "$LAION768" "$TARGET768"

echo "Copy original gt"
mkdir -p "$TARGET_ORIG"
cp "$LAION_ORIG/gt.npy" "$TARGET_ORIG/"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

echo "Running HNSW with laion 768D"
/usr/bin/time -v python -u hnsw_index.py \
  --xb "$TARGET768/xbt.npy" \
  --xq "$TARGET768/xqt.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label catalyzer768_0_0005 \
  --mode both \
  --no-save-index \
  --efSearch-values 30,40,50,60,80,100,150,200,300,400,500 \
  --output-dir "$OUTPUT_DIR" \
  --metric ip

clean_scratch
