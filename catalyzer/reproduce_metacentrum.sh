#!/bin/bash
#PBS -l select=1:ncpus=32:mem=20gb:scratch_ssd=30gb:cluster=halmir
#PBS -l walltime=1:00:00
# Usage: qsub reproduce_metacentrum.sh

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/catalyzer"
BIGANN_DIR="/storage/brno2/home/martykant/datasets/bigann"
DEEP1B_DIR="/storage/brno2/home/martykant/datasets/deep1b"
LOG_FILE="$SCRIPT_DIR/reproduce_interactive.log"
BYTES=1320000000  # first 10 M vectors (132 B each)
export BIGANN_PATH="$SCRATCHDIR/bigann"
export DEEP1B_PATH="$SCRATCHDIR/deep"


module add mambaforge


export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

echo "======================================"
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "Threads: $OMP_NUM_THREADS"
echo "======================================"

mkdir -p $BIGANN_PATH/gnd
echo "Copying the query and gnd files..."
cp $BIGANN_DIR/bigann_query.bvecs $BIGANN_PATH/
cp $BIGANN_DIR/gnd/*.ivecs $BIGANN_PATH/gnd/

echo "Copying bigann_learn.bvecs (10M vectors)..."
head -c $BYTES $BIGANN_DIR/bigann_learn.bvecs > $BIGANN_PATH/bigann_learn.bvecs

echo "Copying bigann_base.bvecs (10M vectors)..."
head -c $BYTES $BIGANN_DIR/bigann_base.bvecs > $BIGANN_PATH/bigann_base.bvecs


BYTES=3840000008  # 8 B header + first 10 M vectors (384 B each)
echo "Starting data copy to SCRATCHDIR..."

mkdir -p $DEEP1B_PATH
COPY_START=$(date +%s)

cp $DEEP1B_DIR/query.public.10K.fbin $DEEP1B_PATH/
cp $DEEP1B_DIR/deep_new_groundtruth.public.10K.bin $DEEP1B_PATH/

echo "Copying first $BYTES bytes (~10M vectors) from learn and base files..."
head -c $BYTES $DEEP1B_DIR/learn.350M.fbin > $DEEP1B_PATH/learn.350M.fbin
head -c $BYTES $DEEP1B_DIR/base.1B.fbin > $DEEP1B_PATH/base.1B.fbin

COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))
echo "Data copying finished in $COPY_TIME seconds."

mamba activate /storage/brno2/home/martykant/conda_envs/bc


echo "Writing the reproduction output to $LOG_FILE"
cd $SCRIPT_DIR

bash reproduce.sh 2>&1 | tee -a $LOG_FILE


echo "======================================"
echo "Done: $(date)"
echo "Log saved to $LOG_FILE"
echo "======================================"

clean_scratch
