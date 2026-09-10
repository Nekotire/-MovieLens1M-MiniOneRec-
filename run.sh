#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ " ${*:-} " == *" --smoke "* ]]; then
  exec bash scripts/smoke_test.sh
fi

DATA_ROOT="$(python3 scripts/select_data_root.py)"
export DATA_ROOT
export HF_HOME="$DATA_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$DATA_ROOT/cache/conda/pkgs"
export XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"
export TMPDIR="$DATA_ROOT/tmp"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$TMPDIR"

ENV_PREFIX="$DATA_ROOT/envs/minionerec"
if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  PYTHON="$ENV_PREFIX/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
exec "$PYTHON" scripts/pipeline.py "$@"
