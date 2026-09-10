#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET="${DATASET:-MovieLens1M}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_ROOT/datasets/MovieLens1M/processed}"

python rqkmeans_plus.py \
  --data_path "${EMBEDDING_PATH:-$PROCESSED_DIR/$DATASET.emb.npy}" \
  --pretrained_codebook_path "${CODEBOOK_PATH:-$PROCESSED_DIR/$DATASET.codebooks_constrained.npz}" \
  --num_emb_list 256 256 256 \
  "$@"
