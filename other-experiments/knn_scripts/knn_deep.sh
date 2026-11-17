#!/bin/bash
#PBS -l select=1:ncpus=16:mem=20gb:scratch_ssd=10gb
#PBS -l walltime=00:30:00
# Usage: qsub knn_scripts/knn_deep.sh

set -euo pipefail

# Scores every swept pair. Uncomment all pairs in export_scripts/export_run_deep.sh and export them first.
PAIRS=(
    # --- 16D ---
    "16 0.05"
    "16 0.02"
    "16 0.01"
    "16 0.005"
    "16 0.001"

    # --- 32D ---
    "32 0.005"
    "32 0.001"
    "32 0.0007"
    "32 0.0005"
    "32 0.0001"
    "32 0.008"
    "32 0.01"
    "32 0.02"

    # --- 64D ---
    "64 0.001"
    "64 0.0005"
    "64 0.0002"
    "64 0.0001"
    "64 0.00005"

    # --- 96D ---
    "96 0.01"
    "96 0.001"
    "96 0.0005"
    "96 0.0002"
    "96 0.00015"
    "96 0.0001"
    "96 0.00005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/knn/deep1b"}

DEEP1B_ORIG="/storage/brno2/home/martykant/data/deep1b/original/"
TARGET_ORIG="$SCRATCHDIR/original"
SCRIPT_DIR="$REPO_DIR/other-experiments"


export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_knn_deep1b.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

echo "Copy deep1b original"
cp -r "$DEEP1B_ORIG" "$TARGET_ORIG"

cd $SCRIPT_DIR

mamba activate /storage/brno2/home/martykant/conda_envs/bc

echo "Running Exact k-NN with deep1b_original"
/usr/bin/time -v python -u knn.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label original \
  --metric "l2" \
  --output-dir "$OUTPUT_DIR"

# Every (dout, lambda) pair
for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = ${DIM}, LAMBDA = ${LAMBDA}"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    CAT="/storage/brno2/home/martykant/data/deep1b/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="${SCRATCHDIR}/${DIM}_${LAMBDA_LABEL}"

    echo "Copy deep1b ${DIM} for lambda ${LAMBDA}"
    cp -r "${CAT}" "${TARGET}"

    echo "Running Exact k-NN with deep1b_${DIM} (lambda ${LAMBDA})"
    /usr/bin/time -v python -u knn.py \
      --xb "${TARGET}/xbt.npy" \
      --xq "${TARGET}/xqt.npy" \
      --gt "${TARGET_ORIG}/gt.npy" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
      --metric "l2" \
      --output-dir "${OUTPUT_DIR}"

    rm -rf "${TARGET}"
done

clean_scratch
