#!/usr/bin/env bash
set -euo pipefail
target="${1:-${DATA_ROOT:-/data}}"
minimum_gb="${2:-8}"
mkdir -p "$target"
available_kb="$(df -Pk "$target" | awk 'NR==2 {print $4}')"
required_kb=$((minimum_gb * 1024 * 1024))
if [[ "$available_kb" -lt "$required_kb" ]]; then
  echo "ERROR: insufficient disk space at $target: require at least ${minimum_gb} GB free." >&2
  exit 1
fi
echo "Disk check PASS: $target has $((available_kb / 1024 / 1024)) GB free."
