from __future__ import annotations

import csv
import random
from pathlib import Path


def deterministic_sample_csv(
    source: str | Path,
    destination: str | Path,
    max_samples: int,
    seed: int,
) -> tuple[Path, int, int]:
    """Reservoir-sample CSV rows before expensive prompt construction."""
    source_path = Path(source)
    destination_path = Path(destination)
    if max_samples <= 0:
        return source_path, -1, -1

    rng = random.Random(seed)
    reservoir: list[tuple[int, list[str]]] = []
    total = 0
    with source_path.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.reader(source_handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty RL training CSV: {source_path}") from exc
        for total, row in enumerate(reader, start=1):
            entry = (total, row)
            if len(reservoir) < max_samples:
                reservoir.append(entry)
            else:
                position = rng.randrange(total)
                if position < max_samples:
                    reservoir[position] = entry

    reservoir.sort(key=lambda value: value[0])
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination_handle:
        writer = csv.writer(destination_handle)
        writer.writerow(header)
        writer.writerows(row for _, row in reservoir)
    temporary.replace(destination_path)
    return destination_path, len(reservoir), total
