#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: setup.sh targets Ubuntu/Linux GPU servers." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi is not available." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

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
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$TMPDIR" "$DATA_ROOT/envs"
bash scripts/check_disk.sh "$DATA_ROOT" 20

if ! command -v unzip >/dev/null 2>&1 || { ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; }; then
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$EUID" -eq 0 ]]; then
      apt-get update && apt-get install -y unzip curl
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y unzip curl
    else
      echo "ERROR: unzip and curl/wget are required, and setup cannot invoke apt-get without root/sudo." >&2
      exit 1
    fi
  else
    echo "ERROR: unzip and curl or wget are required." >&2
    exit 1
  fi
fi

ENV_PREFIX="$DATA_ROOT/envs/minionerec"
if command -v conda >/dev/null 2>&1; then
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
  fi
else
  MINICONDA_DIR="$DATA_ROOT/miniconda3"
  if [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
    installer="$DATA_ROOT/downloads/Miniconda3-latest-Linux-x86_64.sh"
    mkdir -p "$DATA_ROOT/downloads"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --retry 3 --output "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif command -v wget >/dev/null 2>&1; then
      wget --tries=3 --output-document="$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    else
      echo "ERROR: curl or wget is required to install Miniconda." >&2; exit 1
    fi
    bash "$installer" -b -p "$MINICONDA_DIR"
    rm -f "$installer"
  fi
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$MINICONDA_DIR/bin/conda" create --prefix "$ENV_PREFIX" python=3.11 -y
  fi
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
"$PYTHON" -m pip install -r requirements_server.txt
"$PYTHON" - <<'PY'
import torch, transformers, trl, accelerate, datasets, deepspeed, bitsandbytes
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__, "trl:", trl.__version__)
PY
"$PYTHON" -m pip cache purge || true
if [[ -x "$DATA_ROOT/miniconda3/bin/conda" ]]; then
  "$DATA_ROOT/miniconda3/bin/conda" clean --all -y || true
elif command -v conda >/dev/null 2>&1; then
  conda clean --all -y || true
fi
python3 scripts/select_data_root.py --path "$DATA_ROOT" --persist >/dev/null
echo "Setup complete. Run: bash run.sh"
