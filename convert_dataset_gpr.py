#!/usr/bin/env python3
"""Create the optional GPR view of the MovieLens1M MiniOneRec dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from convert_dataset import convert


def convert_gpr(data_dir: Path, output_dir: Path, dataset_name: str = "MovieLens1M") -> dict[str, int]:
    """Run the canonical converter and append deterministic GPR context columns.

    MovieLens has no browsing/search event types or heterogeneous user-value labels,
    so the optional GPR branch uses explicit neutral tokens instead of inventing data.
    """
    counts = convert(data_dir, output_dir, dataset_name)
    for split in ("train", "valid", "test"):
        csv_path = output_dir / split / f"{dataset_name}.csv"
        frame = pd.read_csv(csv_path)
        frame["user_id_original_str"] = frame["user_id"].astype(str)
        frame["u_token"] = "[USER_UNKNOWN]"
        frame["e_token"] = "[CTX_HOMEPAGE]"
        frame["final_value"] = 1.0
        frame.to_csv(csv_path, index=False)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MovieLens1M for the optional GPR branch")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="MovieLens1M")
    args = parser.parse_args()
    print(convert_gpr(Path(args.data_dir), Path(args.output_dir), args.dataset_name))


if __name__ == "__main__":
    main()
