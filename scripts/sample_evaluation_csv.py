#!/usr/bin/env python3
"""Create a deterministic subset from an already separated evaluation CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_csv(source: Path, destination: Path, size: int, seed: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("size must be positive; use the source CSV directly for full evaluation")

    frame = pd.read_csv(source)
    sampled_size = min(size, len(frame))
    sampled = frame.sample(n=sampled_size, random_state=seed)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(destination, index=False)
    return len(frame), len(sampled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    total, sampled = sample_csv(args.input, args.output, args.size, args.seed)
    print(f"Evaluation sample: {sampled}/{total} rows, seed={args.seed}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
