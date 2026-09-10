#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SMOKE_PYTHON="${PYTHON:-python3}"
if [[ -n "${DATA_ROOT:-}" && -x "$DATA_ROOT/envs/minionerec/bin/python" ]]; then
  SMOKE_PYTHON="$DATA_ROOT/envs/minionerec/bin/python"
fi
"$SMOKE_PYTHON" -m pytest -q tests/test_movielens_downloader.py tests/test_movielens_pipeline.py tests/test_constrained_decoding.py tests/test_movielens_entrypoints.py tests/test_training_hardening.py
echo "Smoke test PASS. No formal model training was executed."
