from __future__ import annotations

import re
import shutil
from pathlib import Path


_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_directories(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            match = _CHECKPOINT.fullmatch(path.name)
            if path.is_dir() and match:
                checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoints = checkpoint_directories(output_dir)
    return checkpoints[-1] if checkpoints else None


def cleanup_checkpoints(output_dir: str | Path, keep_latest: bool = True) -> list[Path]:
    checkpoints = checkpoint_directories(output_dir)
    removed = checkpoints[:-1] if keep_latest and checkpoints else checkpoints
    for checkpoint in removed:
        shutil.rmtree(checkpoint)
    return removed
