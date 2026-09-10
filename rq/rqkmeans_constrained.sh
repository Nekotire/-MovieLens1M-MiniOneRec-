#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET="${DATASET:-MovieLens1M}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_ROOT/datasets/MovieLens1M/processed}"
EMBEDDING_PATH="${EMBEDDING_PATH:-$PROCESSED_DIR/$DATASET.emb.npy}"

python rqkmeans_constrained.py \
  --dataset "$DATASET" \
  --root "$PROCESSED_DIR" \
  --data_path "$EMBEDDING_PATH" \
  "$@"
