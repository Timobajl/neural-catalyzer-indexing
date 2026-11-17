#!/bin/bash
#PBS -l select=1:ncpus=32:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=1:00:00
# Usage: qsub knn_scripts/knn_bigann.sh

set -euo pipefail

# Scores every swept pair. Uncomment all pairs in export_scripts/export_run.sh and export them first.
PAIRS=(
    # --- 16D ---
    "16 0.05"
    "16 0.02"
    "16 0.01"
    "16 0.005"
    "16 0.001"
    "16 0.0005"
    "16 0.0001"
    "16 0.00005"

    # --- 32D ---
    "32 0.01"
    "32 0.005"
    "32 0.001"
    "32 0.0007"
    "32 0.0005"
    "32 0.0001"
    "32 0.00005"
    "32 0.00001"
    "32 0.000005"

    # --- 64D ---
    "64 0.01"
    "64 0.005"
    "64 0.001"
    "64 0.0005"
    "64 0.0002"
    "64 0.0001"
    "64 0.00008"
    "64 0.00005"
    "64 0.00001"
    "64 0.000005"

    # --- 128D ---
    "128 0.01"
    "128 0.005"
    "128 0.001"
    "128 0.0005"
    "128 0.0002"
    "128 0.0001"
    "128 0.00008"
    "128 0.00005"
    "128 0.00001"
    "128 0.000005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/knn/bigann"}

BIGANN_ORIG="/storage/brno2/home/martykant/data/bigann/original/"
TARGET_ORIG="$SCRATCHDIR/original"
SCRIPT_DIR="$REPO_DIR/other-experiments"


export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_knn.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

# 1. Original space
echo "Copy bigann original"
cp -r "$BIGANN_ORIG" "$TARGET_ORIG"

cd $SCRIPT_DIR

mamba activate /storage/brno2/home/martykant/conda_envs/bc

echo "Running Exact k-NN with bigann_original"
/usr/bin/time -v python -u knn.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label original \
  --output-dir "$OUTPUT_DIR"


# 2. PCA baseline, with and without l2-normalization
for dim in 16 32 64 128; do
  /usr/bin/time -v python -u knn.py \
    --xb "$TARGET_ORIG/xb.npy" \
    --xq "$TARGET_ORIG/xq.npy" \
    --gt "$TARGET_ORIG/gt.npy" \
    --xt "$TARGET_ORIG/xt.npy" \
    --label pca${dim}_norm \
    --pca-dim ${dim} \
    --output-dir "$OUTPUT_DIR" \
    --output-name knn_ceiling_results_pca.csv
 
  /usr/bin/time -v python -u knn.py \
    --xb "$TARGET_ORIG/xb.npy" \
    --xq "$TARGET_ORIG/xq.npy" \
    --gt "$TARGET_ORIG/gt.npy" \
    --xt "$TARGET_ORIG/xt.npy" \
    --label pca${dim} \
    --pca-dim ${dim} \
    --no-pca-normalize \
    --output-dir "$OUTPUT_DIR" \
    --output-name knn_ceiling_results_pca.csv
done

# 3. Every (dout, lambda) pair
for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = ${DIM}, LAMBDA = ${LAMBDA}"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    CAT="/storage/brno2/home/martykant/data/bigann/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="${SCRATCHDIR}/${DIM}_${LAMBDA_LABEL}"

    echo "Copy bigann ${DIM} for lambda ${LAMBDA}"
    cp -r "${CAT}" "${TARGET}"

    echo "Running Exact k-NN with bigann${DIM} (lambda ${LAMBDA})"
    /usr/bin/time -v python -u knn.py \
      --xb "${TARGET}/xbt.npy" \
      --xq "${TARGET}/xqt.npy" \
      --gt "${TARGET_ORIG}/gt.npy" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
      --output-dir "${OUTPUT_DIR}"

    rm -rf "${TARGET}"
done

clean_scratch