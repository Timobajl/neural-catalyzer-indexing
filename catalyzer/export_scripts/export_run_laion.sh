#!/bin/bash
#PBS -l select=1:ncpus=10:mem=38gb:scratch_ssd=370gb
#PBS -l walltime=10:00:00
# Usage: qsub export_scripts/export_run_laion.sh

set -euo pipefail

DATABASE="laion"


# Pairs used in the thesis are active. The others complete the lambda sweep.
# The full sweep of 29 pairs takes about 12 h, so raise the walltime to 24 h for it.
PAIRS=(
    # --- 16D ---
    # "16 0.05"
    # "16 0.02"
    # "16 0.01"
    # "16 0.08"
    "16 0.1"
    # "16 0.2"

    # --- 32D ---
    # "32 0.005"
    "32 0.001"
    # "32 0.0007"
    # "32 0.0005"

    # --- 64D ---
    # "64 0.001"
    # "64 0.0005"
    "64 0.0002"
    # "64 0.0001"

    # --- 128D ---
    # "128 0.01"
    # "128 0.0005"
    # "128 0.0002"
    # "128 0.0001"
    # "128 0.00008"
    "128 0.001"
    # "128 0.005"

    # --- 768D ---
    "768 0.0005"
    # "768 0.0001"
    # "768 0.00005"
    # "768 0.00001"
    # "768 0.000005"
    # "768 0.001"
    # "768 0.002"
    # "768 0.005"
)

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
CKPT_BASE="$REPO_DIR/catalyzer/checkpoints"
LAION_DIR="/storage/brno2/home/martykant/datasets/laion"
SCRIPT_DIR="$REPO_DIR/catalyzer"
DEVICE="auto"
OUTPUT_DIR=${OUTPUT_DIR:-"/storage/brno2/home/martykant/data"}

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

mkdir -p "$OUTPUT_DIR/$DATABASE"

MASTER_LOG="$OUTPUT_DIR/${DATABASE}_master_setup.log"
echo "Logging setup output to $MASTER_LOG"
exec >> "$MASTER_LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

module add mambaforge

export LAION_PATH="$SCRATCHDIR"

echo "Starting data copy to SCRATCHDIR..."
COPY_START=$(date +%s)

cp $LAION_DIR/laion2B-en-clip768v2-n=100M.h5 $SCRATCHDIR/
cp $LAION_DIR/public-queries-2024-laion2B-en-clip768v2-n=10k.h5 $SCRATCHDIR/
cp $LAION_DIR/gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 $SCRATCHDIR/

COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))
echo "Data copying finished in $COPY_TIME seconds."

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd $SCRIPT_DIR

for pair in "${PAIRS[@]}"; do
    read -r dout lam <<< "$pair"

    ckpt="${CKPT_BASE}/${DATABASE}${dout}/${lam}/checkpoint.pth.best"
    TRANSFORMED_DIR="$SCRATCHDIR/$DATABASE/catalyzer_d${dout}_l${lam}"
    DEST_DIR="$OUTPUT_DIR/$DATABASE/catalyzer_d${dout}_l${lam}"

    echo ""
    echo "===================================================================="
    echo "  Dataset : $DATABASE   Dim : ${dout}D   Lambda : $lam"
    echo "  Ckpt    : $ckpt"
    echo "===================================================================="

    start=$(date +%s)

    /usr/bin/time -v python -u export_vectors.py \
        --ckpt-path  "$ckpt" \
        --database   "$DATABASE" \
        --device     "$DEVICE" \
        --output-dir "$SCRATCHDIR"

    end=$(date +%s)
    runtime=$((end - start))
    echo "Run time: $runtime seconds."

    echo "Copying exported data to $DEST_DIR ..."
    mkdir -p "$DEST_DIR"
    cp -a "$TRANSFORMED_DIR/." "$DEST_DIR/"
    echo "Data copied."

    rm -rf "$TRANSFORMED_DIR"
done

if [ -d "$SCRATCHDIR/$DATABASE/original/" ]; then
    mkdir -p "$OUTPUT_DIR/$DATABASE/original"
    cp -a "$SCRATCHDIR/$DATABASE/original/." "$OUTPUT_DIR/$DATABASE/original/"
    echo "Original data copied."
fi

echo "Export completed."
clean_scratch
