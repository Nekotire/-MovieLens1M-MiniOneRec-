#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET="${DATASET:-MovieLens1M}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_ROOT/datasets/MovieLens1M/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/output/MovieLens1M/rqvae}"

python rqvae.py \
  --data_path "${EMBEDDING_PATH:-$PROCESSED_DIR/$DATASET.emb.npy}" \
  --ckpt_dir "$OUTPUT_DIR" \
  "$@"
