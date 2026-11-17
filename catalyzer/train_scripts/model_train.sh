#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=14gb:scratch_ssd=25gb:cluster=galdor
#PBS -l walltime=02:00:00
# Usage: qsub -v DOUT=64,LAMBDA=0.0002 train_scripts/model_train.sh

DOUT=${DOUT:-16}
LAMBDA=${LAMBDA:-0.01}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/catalyzer"
BIGANN_DIR="/storage/brno2/home/martykant/datasets/bigann"
TARGET_DIR="$SCRIPT_DIR/checkpoints/bigann${DOUT}/${LAMBDA}"

mkdir -p "$TARGET_DIR"

BYTES=1320000000  # first 10 M vectors (132 B each)
export BIGANN_PATH="$SCRATCHDIR"

module add mambaforge

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN


LOG_FILE="$TARGET_DIR/stdout.log"
exec >> "$LOG_FILE" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "Threads: $PBS_NUM_PPN, Lambda: $LAMBDA, Dout: $DOUT"
echo "====================================================="

mkdir -p $SCRATCHDIR/gnd
echo "Starting data copy to SCRATCHDIR..."
COPY_START=$(date +%s)
cp $BIGANN_DIR/bigann_query.bvecs $SCRATCHDIR/
cp $BIGANN_DIR/gnd/*.ivecs $SCRATCHDIR/gnd/
head -c $BYTES $BIGANN_DIR/bigann_learn.bvecs > $SCRATCHDIR/bigann_learn.bvecs
head -c $BYTES $BIGANN_DIR/bigann_base.bvecs > $SCRATCHDIR/bigann_base.bvecs
COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))
echo "Data copying finished in $COPY_TIME seconds."

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd $SCRIPT_DIR


echo "Start training time: $(date)"
TRAIN_START=$(date +%s)

/usr/bin/time -v python -u train.py \
  --dout $DOUT \
  --save_best_criterion none,rank=10 \
  --validation_quantizers none \
  --database bigann \
  --lambda_uniform $LAMBDA \
  --checkpoint_dir "$TARGET_DIR"

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

echo "End training time: $(date)"
echo "Training finished in $TRAIN_TIME seconds."
echo "====================================================="

clean_scratch
