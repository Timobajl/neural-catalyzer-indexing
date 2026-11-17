#!/bin/bash
#PBS -l select=1:ncpus=8:mem=12gb:scratch_local=20gb:cluster=galdor
#PBS -l walltime=01:00:00
# Usage: qsub export_scripts/export_run.sh

set -euo pipefail

DATABASE="bigann"
# Pairs used in the thesis are active. The others complete the lambda sweep.
PAIRS=(
    # --- 16D ---
    # "16 0.05"
    # "16 0.02"
    "16 0.01"
    # "16 0.005"
    # "16 0.001"
    # "16 0.0005"
    # "16 0.0001"
    # "16 0.00005"
    # --- 32D ---
    # "32 0.01"
    # "32 0.005"
    # "32 0.001"
    "32 0.0007"
    # "32 0.0005"
    # "32 0.0001"
    # "32 0.00005"
    # "32 0.00001"
    # "32 0.000005"
    # --- 64D ---
    # "64 0.01"
    # "64 0.005"
    # "64 0.001"
    # "64 0.0005"
    "64 0.0002"
    # "64 0.0001"
    # "64 0.00008"
    # "64 0.00005"
    # "64 0.00001"
    # "64 0.000005"
    # --- 128D ---
    # "128 0.01"
    # "128 0.005"
    # "128 0.001"
    # "128 0.0005"
    # "128 0.0002"
    "128 0.0001"
    # "128 0.00008"
    # "128 0.00005"
    # "128 0.00001"
    # "128 0.000005"
)

CKPT_BASE="checkpoints"

BIGANN_DIR="/storage/brno2/home/martykant/datasets/bigann"
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

BYTES=264000000  # first 2 M vectors (132 B each)
export BIGANN_PATH="$SCRATCHDIR"

echo "copying bigann query to scratch"
cp $BIGANN_DIR/bigann_query.bvecs $SCRATCHDIR/

echo "copying gnd to scratch"
mkdir -p $SCRATCHDIR/gnd
cp $BIGANN_DIR/gnd/*.ivecs $SCRATCHDIR/gnd/

echo "copying learn to scratch"
head -c $BYTES $BIGANN_DIR/bigann_learn.bvecs > $SCRATCHDIR/bigann_learn.bvecs

echo "copying base to scratch"
head -c $BYTES $BIGANN_DIR/bigann_base.bvecs > $SCRATCHDIR/bigann_base.bvecs


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
