#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET="${DATASET:-MovieLens1M}"
MINIONEREC_DIR="${MINIONEREC_DIR:-$DATA_ROOT/datasets/MovieLens1M/minionerec}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_ROOT/datasets/MovieLens1M/processed}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
: "${TS_DESCRIPTION_PATH:?Set TS_DESCRIPTION_PATH to a MovieLens SID keyword JSON file}"

python ts_rec_sft.py \
  --base_model "$MODEL" \
  --train_file "$MINIONEREC_DIR/train/$DATASET.csv" \
  --eval_file "$MINIONEREC_DIR/valid/$DATASET.csv" \
  --output_dir "$DATA_ROOT/output/MovieLens1M/ts_rec_sft" \
  --category "$DATASET" \
  --sid_index_path "$PROCESSED_DIR/$DATASET.index.json" \
  --item_meta_path "$PROCESSED_DIR/$DATASET.item.json" \
  --description_path "$TS_DESCRIPTION_PATH" \
  "$@"
