#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=24gb:scratch_ssd=25gb:cluster=galdor
#PBS -l walltime=04:00:00
# Usage: qsub -v DOUT=64,LAMBDA=0.0002 train_scripts/model_train_deep.sh

DOUT=${DOUT:-16}
LAMBDA=${LAMBDA:-0.01}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/catalyzer"
DEEP1B_DIR="/storage/brno2/home/martykant/datasets/deep1b"
TARGET_DIR="$SCRIPT_DIR/checkpoints/deep1b${DOUT}/${LAMBDA}"

mkdir -p "$TARGET_DIR"

BYTES=3840000008  # 8 B header + first 10 M vectors (384 B each)
export DEEP1B_PATH="$SCRATCHDIR"

module add mambaforge

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

LOG_FILE="$TARGET_DIR/stdout.log"
echo "Logging into $LOG_FILE"
exec >> "$LOG_FILE" 2>&1


echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "Threads: $PBS_NUM_PPN, Lambda: $LAMBDA, Dout: $DOUT"
echo "====================================================="

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


echo "Start training time: $(date)"
TRAIN_START=$(date +%s)

/usr/bin/time -v python -u train.py \
  --dout $DOUT \
  --save_best_criterion none,rank=10 \
  --validation_quantizers none \
  --database deep1b \
  --size_base 1000000 \
  --lambda_uniform $LAMBDA \
  --checkpoint_dir "$TARGET_DIR"

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

echo "End training time: $(date)"
echo "Training finished in $TRAIN_TIME seconds."
echo "====================================================="

clean_scratch
