#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PERSISTED_PATH = REPO_ROOT / ".data_root"
DEFAULT_CANDIDATES = (Path("/data"), Path("/workspace"), Path("/root/autodl-tmp"))


def usable(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def select_data_root(
    explicit: str | None = None,
    persisted_path: Path = PERSISTED_PATH,
    candidates: tuple[Path, ...] = DEFAULT_CANDIDATES,
) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not usable(path):
            raise RuntimeError(f"DATA_ROOT is not a writable directory: {path}")
        return path

    if persisted_path.is_file():
        saved = Path(persisted_path.read_text(encoding="utf-8").strip()).expanduser().resolve()
        if usable(saved):
            return saved

    available = [path.resolve() for path in candidates if usable(path)]
    if not available:
        raise RuntimeError(
            "No writable data disk found. Set DATA_ROOT to a writable mounted data-disk path."
        )
    return max(available, key=lambda path: shutil.disk_usage(path).free)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the writable data mount with the most free space.")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--path", help="Validate and select this explicit path")
    args = parser.parse_args()
    selected = select_data_root(args.path or os.environ.get("DATA_ROOT"))
    if args.persist:
        PERSISTED_PATH.write_text(str(selected) + "\n", encoding="utf-8")
    print(selected)


if __name__ == "__main__":
    main()
