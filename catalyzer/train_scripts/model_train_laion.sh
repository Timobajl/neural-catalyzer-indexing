#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=36gb:scratch_ssd=80gb:cluster=galdor
#PBS -l walltime=08:00:00
# Usage: qsub -v DOUT=64,LAMBDA=0.0002 train_scripts/model_train_laion.sh

DOUT=${DOUT:-16}
LAMBDA=${LAMBDA:-0.01}

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/catalyzer"
LAION_DIR="/storage/brno2/home/martykant/datasets/laion"
TARGET_DIR="$SCRIPT_DIR/checkpoints/laion${DOUT}/${LAMBDA}"

mkdir -p "$TARGET_DIR"

export LAION_PATH="$SCRATCHDIR"

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

cp $LAION_DIR/laion2B-en-clip768v2-n=10M.h5 $SCRATCHDIR/
cp $LAION_DIR/public-queries-2024-laion2B-en-clip768v2-n=10k.h5 $SCRATCHDIR/

cp $LAION_DIR/gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 $SCRATCHDIR/

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
  --database laion \
  --size_base 2000000 \
  --lambda_uniform $LAMBDA \
  --checkpoint_dir "$TARGET_DIR"

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

echo "End training time: $(date)"
echo "Training finished in $TRAIN_TIME seconds."
echo "====================================================="

clean_scratch
