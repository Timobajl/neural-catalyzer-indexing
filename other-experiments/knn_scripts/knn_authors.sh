#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb:scratch_ssd=10gb:cluster=halmir
#PBS -l walltime=1:00:00
# Usage: qsub knn_scripts/knn_authors.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/brno2/home/martykant/neural-catalyzer-indexing}"
SCRIPT_DIR="$REPO_DIR/other-experiments"
DATA="/storage/brno2/home/martykant/data"
RESULTS="$REPO_DIR/results/knn"

export OMP_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN

module add mambaforge

mkdir -p "$RESULTS/bigann" "$RESULTS/deep1b"
LOG="$RESULTS/stdout_authors.log"
echo "Logging output to $LOG"
exec > "$LOG" 2>&1

echo "====================================================="
echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "====================================================="

mamba activate /storage/brno2/home/martykant/conda_envs/bc
cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_DIR/catalyzer"

run_knn () {
    local DB="$1"; shift
    local LABEL="$1"; shift
    local CKPT="$1"; shift

    /usr/bin/time -v python -u knn_authors.py \
      --ckpt "$CKPT" \
      --xb "$SCRATCHDIR/$DB/xb.npy" \
      --xq "$SCRATCHDIR/$DB/xq.npy" \
      --gt "$SCRATCHDIR/$DB/gt.npy" \
      --label "$LABEL" \
      --output-dir "$RESULTS/$DB"
}

for DB in bigann deep1b; do
    echo "Copy $DB original"
    cp -r "$DATA/$DB/original" "$SCRATCHDIR/$DB"

    for DIM in 16 32 64 128; do
        run_knn "$DB" "author_binary_d$DIM" "$REPO_DIR/catalyzer/ckpts/binary/$DB/ckpt_$DIM.pth"
    done
    run_knn "$DB" "author_lattice_d24" "$REPO_DIR/catalyzer/ckpts/lattice/$DB/ckpt.pth"
    run_knn "$DB" "author_lattice_end2end_d24" "$REPO_DIR/catalyzer/ckpts/lattice/$DB/ckpt_end2end.pth"

    for DIM in 16 32 64; do
        run_knn "$DB" "thesis_d${DIM}_l0.05" "$REPO_DIR/catalyzer/checkpoints/$DB$DIM/0.05/checkpoint.pth.best"
    done
done

run_knn bigann "thesis_d128_l0.01" "$REPO_DIR/catalyzer/checkpoints/bigann128/0.01/checkpoint.pth.best"
run_knn deep1b "thesis_d96_l0.01" "$REPO_DIR/catalyzer/checkpoints/deep1b96/0.01/checkpoint.pth.best"

clean_scratch
