from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DOWNLOADER = REPO / "scripts/download_movielens1m.sh"


def make_zip(path: Path, missing: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in {"ratings.dat": "1::1::5::1\n", "movies.dat": "1::Movie::Drama\n",
                           "users.dat": "1::F::1::1::00000\n", "README": "test\n"}.items():
            if name != missing:
                archive.writestr(f"ml-1m/{name}", body)


def fake_curl(bin_dir: Path) -> None:
    script = bin_dir / "curl"
    script.write_text("""#!/bin/sh
dest=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then shift; dest="$1"; fi
  shift
done
printf 'download\\n' >> "$MOCK_COUNTER"
cp "$MOCK_ZIP" "$dest"
""", encoding="utf-8")
    script.chmod(0o755)


def run_downloader(tmp_path: Path, archive: Path, *, preset: dict[str, str] | None = None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_curl(bin_dir)
    counter = tmp_path / "counter"
    env = os.environ.copy()
    env.update({"DATA_ROOT": str(tmp_path / "data"), "MOCK_ZIP": str(archive),
                "MOCK_COUNTER": str(counter), "PATH": f"{bin_dir}:{env['PATH']}",
                "MOVIELENS_MIN_ARCHIVE_BYTES": "1", "MOVIELENS_DOWNLOAD_RETRIES": "1",
                "MOVIELENS_RETRY_DELAY_SECONDS": "0", "DELETE_DATASET_ARCHIVE_AFTER_EXTRACT": "true"})
    if preset:
        env.update(preset)
    result = subprocess.run(["bash", str(DOWNLOADER)], cwd=REPO, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    count = len(counter.read_text().splitlines()) if counter.exists() else 0
    return result, count, Path(env["DATA_ROOT"])


def test_missing_dataset_downloads_and_extracts_to_canonical_path(tmp_path):
    archive = tmp_path / "source.zip"
    make_zip(archive)
    result, count, root = run_downloader(tmp_path, archive)
    assert result.returncode == 0, result.stdout
    assert count == 1
    final = root / "datasets/MovieLens1M/raw/ml-1m"
    assert all((final / name).stat().st_size for name in ("ratings.dat", "movies.dat", "users.dat"))
    assert (final / "README").is_file()
    assert not (final / "ml-1m").exists()
    assert not (root / "downloads/ml-1m.zip").exists()


def test_existing_dataset_is_idempotent_and_skips_download(tmp_path):
    archive = tmp_path / "source.zip"
    make_zip(archive)
    first, first_count, root = run_downloader(tmp_path, archive)
    second, second_count, _ = run_downloader(tmp_path, archive)
    assert first.returncode == second.returncode == 0
    assert first_count == second_count == 1
    assert "MovieLens 1M dataset already exists. SKIP DOWNLOAD." in second.stdout


def test_missing_ratings_is_incomplete_and_triggers_download(tmp_path):
    archive = tmp_path / "source.zip"
    make_zip(archive)
    final = tmp_path / "data/datasets/MovieLens1M/raw/ml-1m"
    final.mkdir(parents=True)
    (final / "movies.dat").write_text("x")
    (final / "users.dat").write_text("x")
    result, count, _ = run_downloader(tmp_path, archive)
    assert result.returncode == 0
    assert count == 1
    assert (final / "ratings.dat").stat().st_size > 0


def test_corrupt_zip_stops_pipeline_and_is_deleted(tmp_path):
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    result, count, root = run_downloader(tmp_path, archive)
    assert result.returncode != 0
    assert count == 1
    assert "automatic download failed" in result.stdout
    assert not (root / "downloads/ml-1m.zip").exists()
    assert not (root / "datasets/MovieLens1M/raw/ml-1m/ratings.dat").exists()


def test_archive_missing_core_file_is_rejected(tmp_path):
    archive = tmp_path / "missing.zip"
    make_zip(archive, missing="ratings.dat")
    result, _, root = run_downloader(tmp_path, archive)
    assert result.returncode != 0
    assert "missing non-empty ratings.dat" in result.stdout
    assert not (root / "datasets/MovieLens1M/raw/ml-1m").exists()
