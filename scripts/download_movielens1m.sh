#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
CONFIG_PATH="${MINIONEREC_CONFIG:-config/movielens1m.yaml}"
OFFICIAL_URL="${MOVIELENS_URL:-https://files.grouplens.org/datasets/movielens/ml-1m.zip}"
RETRIES="${MOVIELENS_DOWNLOAD_RETRIES:-3}"
RETRY_DELAY="${MOVIELENS_RETRY_DELAY_SECONDS:-3}"
MIN_BYTES="${MOVIELENS_MIN_ARCHIVE_BYTES:-1000000}"
EXPECTED_SHA256="${MOVIELENS_SHA256:-}"
RAW_PARENT="${MOVIELENS_RAW_PARENT:-$DATA_ROOT/datasets/MovieLens1M/raw}"
DATASET_DIR="$RAW_PARENT/ml-1m"
DOWNLOAD_DIR="${MOVIELENS_DOWNLOAD_DIR:-$DATA_ROOT/downloads}"
ARCHIVE="$DOWNLOAD_DIR/ml-1m.zip"
DELETE_ARCHIVE="${DELETE_DATASET_ARCHIVE_AFTER_EXTRACT:-}"

if [[ -z "$DELETE_ARCHIVE" && -f "$CONFIG_PATH" ]] && command -v python3 >/dev/null 2>&1; then
  DELETE_ARCHIVE="$(python3 - "$CONFIG_PATH" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    print(str(yaml.safe_load(f).get("storage", {}).get("delete_dataset_archive_after_extract", True)).lower())
PY
)"
fi
DELETE_ARCHIVE="${DELETE_ARCHIVE:-true}"

dataset_complete() {
  local name
  for name in ratings.dat movies.dat users.dat; do
    [[ -s "$DATASET_DIR/$name" ]] || return 1
  done
}

fail_download() {
  printf '%s\n' \
    "MovieLens 1M automatic download failed." \
    "Please check network access to the official GroupLens dataset server." >&2
  exit 1
}

if dataset_complete; then
  echo "MovieLens 1M dataset already exists. SKIP DOWNLOAD."
  exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: unzip is required to validate and extract MovieLens 1M." >&2
  exit 1
fi
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
else
  echo "ERROR: neither curl nor wget is installed; cannot download MovieLens 1M." >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$RAW_PARENT"

archive_valid() {
  [[ -f "$ARCHIVE" ]] || return 1
  local size
  size="$(wc -c < "$ARCHIVE" | tr -d '[:space:]')"
  [[ "$size" =~ ^[0-9]+$ && "$size" -gt "$MIN_BYTES" ]] || return 1
  unzip -tqq "$ARCHIVE" >/dev/null 2>&1 || return 1
  if [[ -n "$EXPECTED_SHA256" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      actual_sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      actual_sha256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
    else
      echo "ERROR: SHA-256 is configured but neither sha256sum nor shasum is installed." >&2
      return 1
    fi
    [[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || return 1
  fi
}

if ! archive_valid; then
  rm -f "$ARCHIVE"
  attempt=1
  while [[ "$attempt" -le "$RETRIES" ]]; do
    echo "Downloading MovieLens 1M from GroupLens (attempt $attempt/$RETRIES)..."
    tmp_archive="$ARCHIVE.part"
    rm -f "$tmp_archive"
    if [[ "$DOWNLOADER" == "curl" ]]; then
      curl --fail --location --show-error --connect-timeout 30 --output "$tmp_archive" "$OFFICIAL_URL" || true
    else
      wget --timeout=30 --output-document="$tmp_archive" "$OFFICIAL_URL" || true
    fi
    if [[ -f "$tmp_archive" ]]; then
      mv "$tmp_archive" "$ARCHIVE"
    fi
    if archive_valid; then
      break
    fi
    rm -f "$ARCHIVE" "$tmp_archive"
    if [[ "$attempt" -lt "$RETRIES" ]]; then sleep "$RETRY_DELAY"; fi
    attempt=$((attempt + 1))
  done
fi

archive_valid || fail_download

extract_dir="$(mktemp -d "$RAW_PARENT/.ml-1m.extract.XXXXXX")"
cleanup() { rm -rf "$extract_dir"; }
trap cleanup EXIT
if ! unzip -q "$ARCHIVE" -d "$extract_dir"; then
  rm -f "$ARCHIVE"
  echo "ERROR: MovieLens archive extraction failed; the damaged archive was removed." >&2
  exit 1
fi

source_dir="$extract_dir/ml-1m"
if [[ ! -d "$source_dir" ]]; then
  echo "ERROR: archive does not contain the expected ml-1m directory." >&2
  rm -f "$ARCHIVE"
  exit 1
fi
for name in ratings.dat movies.dat users.dat; do
  if [[ ! -s "$source_dir/$name" ]]; then
    echo "ERROR: extracted MovieLens dataset is missing non-empty $name." >&2
    rm -f "$ARCHIVE"
    exit 1
  fi
done

rm -rf "$DATASET_DIR"
mv "$source_dir" "$DATASET_DIR"
dataset_complete || { echo "ERROR: final MovieLens dataset validation failed." >&2; exit 1; }

if [[ "$DELETE_ARCHIVE" == "true" ]]; then
  rm -f "$ARCHIVE"
fi
echo "MovieLens 1M dataset ready: $DATASET_DIR"
