#!/bin/bash
# Quick example: the whole pipeline on a small part of Deep1B, run locally.
# Usage (from the repository root, conda environment active): bash example/quick_example.sh

set -euo pipefail

DOUT=${DOUT:-64}
LAMBDA=${LAMBDA:-0.0002}

REPO_DIR="${REPO_DIR:-$PWD}"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/example/output"}
DEEP1B_URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"
DEEP1B_DIR="$OUTPUT_DIR/deep1b"
CKPT_DIR="$OUTPUT_DIR/checkpoints/deep1b${DOUT}/${LAMBDA}"
DATA_DIR="$OUTPUT_DIR/data"
RESULTS_DIR="$OUTPUT_DIR/results"

BASE_BYTES=38400008    # 8 B header + first 100 000 vectors (384 B each)
LEARN_BYTES=76800008   # 8 B header + first 200 000 vectors (384 B each)

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$DEEP1B_DIR" "$CKPT_DIR" "$DATA_DIR" "$RESULTS_DIR"
export DEEP1B_PATH="$DEEP1B_DIR"

echo "====================================================="
echo "Start: $(date)"
echo "Lambda: $LAMBDA, Dout: $DOUT"
echo "====================================================="

echo "Downloading the first part of the Deep1B files..."
COPY_START=$(date +%s)

curl -fL -o "$DEEP1B_DIR/query.public.10K.fbin" "$DEEP1B_URL/query.public.10K.fbin"
curl -fL --range 0-$((BASE_BYTES - 1)) -o "$DEEP1B_DIR/base.1B.fbin" "$DEEP1B_URL/base.1B.fbin"
curl -fL --range 0-$((LEARN_BYTES - 1)) -o "$DEEP1B_DIR/learn.350M.fbin" "$DEEP1B_URL/learn.350M.fbin"

COPY_END=$(date +%s)
echo "Download finished in $((COPY_END - COPY_START)) seconds."

cd "$REPO_DIR/catalyzer"

echo "Start training time: $(date)"
TRAIN_START=$(date +%s)

python -u train.py \
  --dout $DOUT \
  --save_best_criterion none,rank=10 \
  --validation_quantizers none \
  --database deep1b \
  --size_base 50000 \
  --num_learn 50000 \
  --lambda_uniform $LAMBDA \
  --epochs 10 \
  --lr_schedule 0.1 \
  --checkpoint_dir "$CKPT_DIR"

TRAIN_END=$(date +%s)
echo "Training finished in $((TRAIN_END - TRAIN_START)) seconds."

echo "Exporting vectors..."
EXPORT_START=$(date +%s)

python -u export_vectors.py \
  --ckpt-path "$CKPT_DIR/checkpoint.pth.best" \
  --database deep1b \
  --output-dir "$DATA_DIR"

EXPORT_END=$(date +%s)
echo "Export finished in $((EXPORT_END - EXPORT_START)) seconds."

ORIG="$DATA_DIR/deep1b/original"
TRANSFORMED="$DATA_DIR/deep1b/catalyzer_d${DOUT}_l${LAMBDA}"

cd "$REPO_DIR/indexes/ivfflat"

echo "Running IVFFlat Index with deep1b_original"
python -u ivfflat_index.py \
  --xb "$ORIG/xb.npy" \
  --xq "$ORIG/xq.npy" \
  --gt "$ORIG/gt.npy" \
  --label original \
  --mode both \
  --no-save-index \
  --output-dir "$RESULTS_DIR"

echo "Running IVFFlat Index with deep1b${DOUT} (lambda ${LAMBDA})"
python -u ivfflat_index.py \
  --xb "$TRANSFORMED/xbt.npy" \
  --xq "$TRANSFORMED/xqt.npy" \
  --gt "$ORIG/gt.npy" \
  --label "catalyzer${DOUT}_${LAMBDA//./_}" \
  --mode both \
  --no-save-index \
  --output-dir "$RESULTS_DIR" \
  --xb-orig "$ORIG/xb.npy"

echo "====================================================="
echo "End: $(date)"
echo "Results: $RESULTS_DIR"
echo "====================================================="
