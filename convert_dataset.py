#!/usr/bin/env python3
"""Convert generic processed interactions and SIDs to MiniOneRec CSV files."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def load_interactions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            history_value = row.get("history_item_ids", row.get("item_sequence", ""))
            rows.append({"user_id": row["user_id"],
                         "history": [int(value) for value in history_value.split() if value],
                         "target": int(row.get("target_item_id", row.get("target_item", "-1"))),
                         "target_timestamp": int(row.get("target_timestamp", 0))})
    return rows


def convert(data_dir: Path, output_dir: Path, dataset_name: str = "MovieLens1M") -> dict[str, int]:
    items = json.loads((data_dir / f"{dataset_name}.item.json").read_text(encoding="utf-8"))
    indices = json.loads((data_dir / f"{dataset_name}.index.json").read_text(encoding="utf-8"))
    if set(items) != set(indices):
        raise ValueError("item.json and index.json must contain the same remapped item IDs")
    sid = {key: "".join(tokens) for key, tokens in indices.items()}
    counts = {}
    for split in ("train", "valid", "test"):
        source = data_dir / f"{dataset_name}.{split}.inter"
        rows = []
        for row in load_interactions(source):
            history_keys = [str(value) for value in row["history"]]
            target = str(row["target"])
            if target not in items or any(key not in items for key in history_keys):
                raise ValueError(f"Unknown item ID in {source}")
            rows.append({"user_id": row["user_id"],
                         "history_item_title": [items[key]["title"] for key in history_keys],
                         "item_title": items[target]["title"], "history_item_id": row["history"],
                         "item_id": row["target"], "history_item_sid": [sid[key] for key in history_keys],
                         "item_sid": sid[target], "target_timestamp": row["target_timestamp"]})
        target_dir = output_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target_dir / f"{dataset_name}.csv", index=False)
        counts[split] = len(rows)
    info_dir = output_dir / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    with (info_dir / f"{dataset_name}.txt").open("w", encoding="utf-8") as handle:
        for item_id in map(str, range(len(items))):
            handle.write(f'{sid[item_id]}\t{items[item_id]["title"]}\t{item_id}\n')
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="MovieLens1M")
    args = parser.parse_args()
    print(convert(Path(args.data_dir), Path(args.output_dir), args.dataset_name))


if __name__ == "__main__":
    main()
