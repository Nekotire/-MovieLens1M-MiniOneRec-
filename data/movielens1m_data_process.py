#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minionerec_utils.config import load_config, resolve_data_root, resolve_path


def read_dat(path: Path, fields: int) -> Iterable[tuple[int, list[str]]]:
    with path.open(encoding="latin-1") as handle:
        for row_index, line in enumerate(handle):
            parts = line.rstrip("\n").split("::")
            if len(parts) != fields:
                raise ValueError(f"Malformed {path.name} line {row_index + 1}: expected {fields} fields")
            yield row_index, parts


def iterative_k_core(interactions: list[dict], user_k: int, item_k: int) -> list[dict]:
    current = interactions
    while True:
        user_counts = collections.Counter(row["original_user_id"] for row in current)
        item_counts = collections.Counter(row["original_item_id"] for row in current)
        filtered = [
            row for row in current
            if user_counts[row["original_user_id"]] >= user_k
            and item_counts[row["original_item_id"]] >= item_k
        ]
        if len(filtered) == len(current):
            return filtered
        current = filtered


def stable_split(samples: list[dict], train_ratio: float, valid_ratio: float) -> dict[str, list[dict]]:
    samples = sorted(samples, key=lambda row: (row["target_timestamp"], row["original_row_index"]))
    total = len(samples)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    return {"train": samples[:train_end], "valid": samples[train_end:valid_end], "test": samples[valid_end:]}


def leave_two_out(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[int, list[dict]] = collections.defaultdict(list)
    for sample in samples:
        grouped[sample["user_id"]].append(sample)
    result = {"train": [], "valid": [], "test": []}
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["target_timestamp"], row["original_row_index"]))
        if len(rows) >= 2:
            result["train"].extend(rows[:-2])
            result["valid"].append(rows[-2])
            result["test"].append(rows[-1])
        else:
            result["test"].extend(rows)
    for rows in result.values():
        rows.sort(key=lambda row: (row["target_timestamp"], row["original_row_index"]))
    return result


def process(raw_dir: Path, output_dir: Path, *, user_k: int = 5, item_k: int = 5,
            min_rating: float | None = None, max_history: int = 10,
            split_strategy: str = "global_time", train_ratio: float = .8,
            valid_ratio: float = .1) -> dict[str, int]:
    required = [raw_dir / name for name in ("ratings.dat", "movies.dat", "users.dat")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("MovieLens raw dataset is incomplete: " + ", ".join(missing))

    interactions = []
    for row_index, (user, item, rating, timestamp) in read_dat(raw_dir / "ratings.dat", 4):
        rating_value = float(rating)
        if min_rating is None or rating_value >= min_rating:
            interactions.append({"original_user_id": user, "original_item_id": item,
                                 "rating": rating_value, "timestamp": int(timestamp),
                                 "original_row_index": row_index})
    interactions.sort(key=lambda row: (row["original_user_id"], row["timestamp"], row["original_row_index"]))
    interactions = iterative_k_core(interactions, user_k, item_k)
    if not interactions:
        raise ValueError("No interactions remain after iterative k-core filtering")

    users = sorted({row["original_user_id"] for row in interactions}, key=int)
    items = sorted({row["original_item_id"] for row in interactions}, key=int)
    user2id = {original: internal for internal, original in enumerate(users)}
    item2id = {original: internal for internal, original in enumerate(items)}
    id2user = {str(value): key for key, value in user2id.items()}
    id2item = {str(value): key for key, value in item2id.items()}

    user_profiles = {}
    for _, (user_id, gender, age, occupation, zipcode) in read_dat(raw_dir / "users.dat", 5):
        if user_id in user2id:
            user_profiles[str(user2id[user_id])] = {
                "original_user_id": user_id, "gender": gender, "age": age,
                "occupation": occupation, "zipcode": zipcode,
            }
    if len(user_profiles) != len(users):
        raise ValueError("users.dat does not contain every retained ratings user")

    movies = {}
    for _, (movie_id, title, genres_value) in read_dat(raw_dir / "movies.dat", 3):
        if movie_id in item2id:
            genres = genres_value.split("|") if genres_value else []
            movies[str(item2id[movie_id])] = {
                "title": title,
                "description": "Genres: " + ", ".join(genres),
                "genres": genres,
                "original_item_id": movie_id,
            }

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in interactions:
        grouped[row["original_user_id"]].append(row)
    samples = []
    for original_user, rows in grouped.items():
        for position in range(1, len(rows)):
            history = rows[max(0, position - max_history):position]
            target = rows[position]
            samples.append({
                "user_id": user2id[original_user],
                "history_item_ids": [item2id[row["original_item_id"]] for row in history],
                "target_item_id": item2id[target["original_item_id"]],
                "target_timestamp": target["timestamp"],
                "original_row_index": target["original_row_index"],
            })
    if split_strategy == "global_time":
        splits = stable_split(samples, train_ratio, valid_ratio)
    elif split_strategy == "leave_two_out":
        splits = leave_two_out(samples)
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "MovieLens1M.user2id.json": user2id, "MovieLens1M.id2user.json": id2user,
        "MovieLens1M.item2id.json": item2id, "MovieLens1M.id2item.json": id2item,
        "MovieLens1M.item.json": movies,
        "MovieLens1M.user.json": user_profiles,
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for split, rows in splits.items():
        with (output_dir / f"MovieLens1M.{split}.inter").open("w", encoding="utf-8") as handle:
            handle.write("user_id\thistory_item_ids\ttarget_item_id\ttarget_timestamp\toriginal_row_index\n")
            for row in rows:
                handle.write(f'{row["user_id"]}\t{" ".join(map(str, row["history_item_ids"]))}\t'
                             f'{row["target_item_id"]}\t{row["target_timestamp"]}\t{row["original_row_index"]}\n')
    summary = {"num_users": len(users), "num_items": len(items), "num_interactions": len(interactions),
               **{f"num_{name}_samples": len(rows) for name, rows in splits.items()}}
    (output_dir / "MovieLens1M.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/movielens1m.yaml")
    parser.add_argument("--raw-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_data_root(config)
    options = config["preprocessing"]
    summary = process(
        Path(args.raw_dir) if args.raw_dir else resolve_path(config, "paths.raw_dir", root),
        Path(args.output_dir) if args.output_dir else resolve_path(config, "paths.processed_dir", root),
        user_k=int(options["user_k"]), item_k=int(options["item_k"]), min_rating=options.get("min_rating"),
        max_history=int(options["max_history"]), split_strategy=options["split_strategy"],
        train_ratio=float(options["train_ratio"]), valid_ratio=float(options["valid_ratio"]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
