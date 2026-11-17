#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:scratch_ssd=50gb
#PBS -l walltime=24:00:00
# Usage: qsub ivfflat_scripts/ivfflat_laion_small.sh

set -euo pipefail

PAIRS=(
    "16 0.1"
    "32 0.001"
    "64 0.0002"
    "128 0.001"
)

# Full 100M exports at lambda*.
LAION_DATA="/storage/brno2/home/martykant/data/laion"
LAION_ORIG="/storage/brno2/home/martykant/data/laion/original/"
LAION_N=102144212

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
CONSISTENCY_CACHE="$REPO_DIR/cache/laion_true_neighbors.npz"

SCRIPT_DIR="$REPO_DIR/indexes/ivfflat"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/ivfflat/laion"}

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

# Only gt.npy is needed from the original: consistency uses the cached
# original-space neighbors, so the 150+ GB xb.npy is not copied.
echo "Copy laion original gt"
mkdir -p "$TARGET_ORIG"
cp "$LAION_ORIG/gt.npy" "$TARGET_ORIG/gt.npy"

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"

for pair in "${PAIRS[@]}"; do
    read -r DIM LAMBDA <<< "$pair"

    echo "====================================================="
    echo "DIM = ${DIM}, LAMBDA = ${LAMBDA}"
    echo "====================================================="

    LAMBDA_LABEL="${LAMBDA//./_}"
    LAION="${LAION_DATA}/catalyzer_d${DIM}_l${LAMBDA}/"
    TARGET="${SCRATCHDIR}/${DIM}_${LAMBDA_LABEL}"

    # Fail fast if this is not a full 100M export.
    python -c "import numpy as np, sys; n = np.load(sys.argv[1], mmap_mode='r').shape[0]; print(f'xbt rows: {n:,}'); sys.exit(0 if n == int(sys.argv[2]) else 1)" \
      "${LAION}/xbt.npy" "${LAION_N}"

    START=$(date +"%s")
    echo "Copy laion ${DIM} for lambda ${LAMBDA}"
    cp -r "${LAION}" "${TARGET}"
    echo "Copy time: $(($(date +"%s") - START))s"

    echo "Running IVFFlat Index with laion${DIM} (lambda ${LAMBDA})"
    /usr/bin/time -v python -u ivfflat_index.py \
      --xb "${TARGET}/xbt.npy" \
      --xq "${TARGET}/xqt.npy" \
      --gt "${TARGET_ORIG}/gt.npy" \
      --label "catalyzer${DIM}_${LAMBDA_LABEL}" \
      --mode both \
      --no-save-index \
      --output-dir "${OUTPUT_DIR}" \
      --metric ip \
      --consistency-cache "${CONSISTENCY_CACHE}"

    echo "Cleaning up scratch for DIM ${DIM} lambda ${LAMBDA}"
    rm -rf "${TARGET}"
done

clean_scratch
