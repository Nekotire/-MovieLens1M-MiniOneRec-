#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET="${DATASET:-MovieLens1M}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_ROOT/datasets/MovieLens1M/processed}"
: "${RQKMEANS_PLUS_CHECKPOINT:?Set RQKMEANS_PLUS_CHECKPOINT to the trained checkpoint path}"

python generate_indices_plus.py \
  --data_path "${EMBEDDING_PATH:-$PROCESSED_DIR/$DATASET.emb.npy}" \
  --ckpt_path "$RQKMEANS_PLUS_CHECKPOINT" \
  --num_emb_list 256 256 256 \
  "$@"
