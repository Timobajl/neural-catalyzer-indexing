#!/bin/bash
#PBS -l select=1:ncpus=32:mem=16gb:scratch_ssd=10gb:cluster=halmir
#PBS -l walltime=1:00:00
# Usage: qsub ivfflat_scripts/ivfflat_deep.sh

set -euo pipefail

PAIRS=(
    "16 0.02"
    "32 0.005"
    "64 0.0002"
    "96 0.0002"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/indexes/ivfflat"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/ivfflat/deep1b"}

DEEP_ORIG="/storage/brno2/home/martykant/data/deep1b/original/"
TARGET_ORIG="$SCRATCHDIR/original"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR"

module add mambaforge

LOG="$OUTPUT_DIR/stdout_final.log"
echo "Logging output to $LOG"
exec >> "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

# Original space
echo "Copy deep1b original"
cp -r "$DEEP_ORIG" "$TARGET_ORIG"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

echo "Running IVFFlat Index with deep1b_original"
/usr/bin/time -v python -u ivfflat_index.py \
  --xb "$TARGET_ORIG/xb.npy" \
  --xq "$TARGET_ORIG/xq.npy" \
  --gt "$TARGET_ORIG/gt.npy" \
  --label original \
  --mode both \
  --no-save-index \
  --output-dir "$OUTPUT_DIR"

# Every (dout, lambda) pair
for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = ${DIM}, LAMBDA = ${LAMBDA}"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    DEEP="/storage/brno2/home/martykant/data/deep1b/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="${SCRATCHDIR}/${DIM}_${LAMBDA_LABEL}"

    echo "Copy deep1b ${DIM} for lambda ${LAMBDA}"
    cp -r "${DEEP}" "${TARGET}"

    echo "Running IVFFlat Index with deep1b${DIM} (lambda ${LAMBDA})"
    /usr/bin/time -v python -u ivfflat_index.py \
      --xb "${TARGET}/xbt.npy" \
      --xq "${TARGET}/xqt.npy" \
      --gt "${TARGET_ORIG}/gt.npy" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
      --mode both \
      --no-save-index \
      --output-dir "${OUTPUT_DIR}" \
      --xb-orig "${TARGET_ORIG}/xb.npy"

    rm -rf "${TARGET}"
done

clean_scratch
