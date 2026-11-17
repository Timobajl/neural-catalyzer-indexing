#!/bin/bash
#PBS -l select=1:ncpus=8:mem=20gb:scratch_local=20gb:cluster=galdor
#PBS -l walltime=04:00:00
# Usage: qsub export_scripts/export_run_deep.sh

set -euo pipefail

DATABASE="deep1b"

# Pairs used in the thesis are active. The others complete the lambda sweep.
PAIRS=(
    # --- 16D ---
    # "16 0.05"
    "16 0.02"
    # "16 0.01"
    # "16 0.005"
    # "16 0.001"

    # --- 32D ---
    "32 0.005"
    # "32 0.001"
    # "32 0.0007"
    # "32 0.0005"
    # "32 0.0001"
    # "32 0.008"
    # "32 0.01"
    # "32 0.02"

    # --- 64D ---
    # "64 0.001"
    # "64 0.0005"
    "64 0.0002"
    # "64 0.0001"
    # "64 0.00005"

    # --- 96D ---
    # "96 0.01"
    # "96 0.001"
    # "96 0.0005"
    "96 0.0002"
    # "96 0.00015"
    # "96 0.0001"
    # "96 0.00005"
)

CKPT_BASE="checkpoints"
DEEP1B_DIR="/storage/brno2/home/martykant/datasets/deep1b"
REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
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

BYTES=3840000008  # 8 B header + first 10 M vectors (384 B each)
export DEEP1B_PATH="$SCRATCHDIR"

echo "Starting data copy to SCRATCHDIR..."
COPY_START=$(date +%s)

cp $DEEP1B_DIR/query.public.10K.fbin $SCRATCHDIR/
cp $DEEP1B_DIR/deep_new_groundtruth.public.10K.bin $SCRATCHDIR/

echo "Copying first $BYTES bytes (~10M vectors) from learn and base files..."
head -c $BYTES $DEEP1B_DIR/learn.350M.fbin > $SCRATCHDIR/learn.350M.fbin
head -c $BYTES $DEEP1B_DIR/base.1B.fbin > $SCRATCHDIR/base.1B.fbin

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