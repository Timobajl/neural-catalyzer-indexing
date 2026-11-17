#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=14gb:scratch_ssd=25gb:cluster=galdor
#PBS -l walltime=04:00:00
# Usage: qsub train_scripts/model_train_300.sh

set -euo pipefail

DOUT=128
LAMBDA=0.0001
EPOCHS=300
# Learning rate of the paper: 0.1, lowered to 0.05 at epoch 80 and to 0.01 at epoch 120.
# train.py repeats every value EPOCHS / 15 = 20 times.
LR_SCHEDULE="0.1,0.1,0.1,0.1,0.05,0.05,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01"

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/catalyzer"
BIGANN_DIR="/storage/brno2/home/martykant/datasets/bigann"
TARGET_DIR="$SCRIPT_DIR/checkpoints_300/bigann${DOUT}/${LAMBDA}"
DATA="/storage/brno2/home/martykant/data/bigann/original"

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
export PYTHONPATH="$REPO_DIR/catalyzer"


echo "Start training time: $(date)"
TRAIN_START=$(date +%s)

/usr/bin/time -v python -u train.py \
  --dout $DOUT \
  --epochs $EPOCHS \
  --lr_schedule "$LR_SCHEDULE" \
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

/usr/bin/time -v python -u "$REPO_DIR/other-experiments/knn_authors.py" \
  --ckpt "$TARGET_DIR/checkpoint.pth.best" \
  --xb "$DATA/xb.npy" \
  --xq "$DATA/xq.npy" \
  --gt "$DATA/gt.npy" \
  --label "thesis_d${DOUT}_l${LAMBDA}_e${EPOCHS}" \
  --output-dir "$REPO_DIR/results/knn/bigann" \
  --output-name knn_ceiling_300.csv

clean_scratch
