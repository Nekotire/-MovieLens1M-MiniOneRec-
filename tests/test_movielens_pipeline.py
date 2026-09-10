from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from convert_dataset import convert
from rq.text2emb.item_text2emb import ordered_item_texts, validate_embeddings


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("movielens1m_data_process", REPO / "data/movielens1m_data_process.py")
PROCESS_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(PROCESS_MODULE)
process = PROCESS_MODULE.process


def tiny_raw(root: Path) -> Path:
    raw = root / "ml-1m"
    raw.mkdir(parents=True)
    (raw / "ratings.dat").write_text(
        "1::10::5::100\n2::10::4::101\n1::20::5::102\n2::30::3::103\n1::30::4::104\n2::20::5::105\n",
        encoding="latin-1")
    (raw / "movies.dat").write_text(
        "10::Ten (2000)::Drama\n20::Twenty (2001)::Comedy|Drama\n30::Thirty (2002)::Action\n",
        encoding="latin-1")
    (raw / "users.dat").write_text("1::F::25::1::00000\n2::M::35::2::11111\n", encoding="latin-1")
    (raw / "README").write_text("tiny")
    return raw


def test_preprocessing_mappings_metadata_sequences_and_time_split(tmp_path):
    output = tmp_path / "processed"
    summary = process(tiny_raw(tmp_path / "raw"), output, user_k=1, item_k=1,
                      train_ratio=.5, valid_ratio=.25, max_history=10)
    assert summary["num_users"] == 2 and summary["num_items"] == 3
    user2id = json.loads((output / "MovieLens1M.user2id.json").read_text())
    id2user = json.loads((output / "MovieLens1M.id2user.json").read_text())
    item2id = json.loads((output / "MovieLens1M.item2id.json").read_text())
    id2item = json.loads((output / "MovieLens1M.id2item.json").read_text())
    assert sorted(user2id.values()) == list(range(len(user2id)))
    assert sorted(item2id.values()) == list(range(len(item2id)))
    assert all(id2user[str(value)] == key for key, value in user2id.items())
    assert all(id2item[str(value)] == key for key, value in item2id.items())
    items = json.loads((output / "MovieLens1M.item.json").read_text())
    assert len(items) == summary["num_items"]
    assert items[str(item2id["20"])]["description"] == "Genres: Comedy, Drama"

    all_rows = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(output / f"MovieLens1M.{split}.inter", sep="\t")
        all_rows[split] = frame
        for _, row in frame.iterrows():
            history = [int(value) for value in str(row.history_item_ids).split()]
            assert len(history) <= 10
            assert all(0 <= value < len(items) for value in history)
            assert 0 <= row.target_item_id < len(items)
    assert all_rows["train"].target_timestamp.max() <= all_rows["valid"].target_timestamp.min()
    assert all_rows["valid"].target_timestamp.max() <= all_rows["test"].target_timestamp.min()


def test_embedding_and_conversion_consistency(tmp_path):
    processed = tmp_path / "processed"
    summary = process(tiny_raw(tmp_path / "raw"), processed, user_k=1, item_k=1,
                      train_ratio=.5, valid_ratio=.25)
    texts = ordered_item_texts(processed / "MovieLens1M.item.json")
    embeddings = np.ones((summary["num_items"], 8), dtype=np.float32)
    validate_embeddings(embeddings, summary["num_items"], 8)
    indices = {str(i): [f"<a_{i}>", f"<b_{i}>", f"<c_{i}>"] for i in range(summary["num_items"])}
    (processed / "MovieLens1M.index.json").write_text(json.dumps(indices))
    output = tmp_path / "minionerec"
    counts = convert(processed, output)
    assert counts == {"train": 2, "valid": 1, "test": 1}
    assert len((output / "info/MovieLens1M.txt").read_text().splitlines()) == summary["num_items"]
    for split in counts:
        frame = pd.read_csv(output / split / "MovieLens1M.csv")
        assert all(repr(value).count("<") >= 3 for value in frame.item_sid)


def test_preprocessing_consumes_downloader_canonical_path(tmp_path):
    source = tmp_path / "tiny.zip"
    raw_source = tiny_raw(tmp_path / "source")
    with zipfile.ZipFile(source, "w") as archive:
        for path in raw_source.iterdir():
            archive.write(path, f"ml-1m/{path.name}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = --output ]; then shift; out=\"$1\"; fi; shift; done\ncp \"$MOCK_ZIP\" \"$out\"\n")
    curl.chmod(0o755)
    data_root = tmp_path / "data"
    env = os.environ | {"DATA_ROOT": str(data_root), "MOCK_ZIP": str(source),
                        "PATH": f"{bin_dir}:{os.environ['PATH']}", "MOVIELENS_MIN_ARCHIVE_BYTES": "1",
                        "DELETE_DATASET_ARCHIVE_AFTER_EXTRACT": "true"}
    result = subprocess.run(["bash", "scripts/download_movielens1m.sh"], cwd=REPO, env=env)
    assert result.returncode == 0
    summary = process(data_root / "datasets/MovieLens1M/raw/ml-1m", tmp_path / "processed", user_k=1, item_k=1)
    assert summary["num_items"] == 3


def test_evaluation_sampling_is_deterministic(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "sample_evaluation_csv", REPO / "scripts/sample_evaluation_csv.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    source = tmp_path / "test.csv"
    pd.DataFrame({"x": list(range(100)), "y": [f"v{i}" for i in range(100)]}).to_csv(source, index=False)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    assert module.sample_csv(source, first, 10, 2024) == (100, 10)
    assert module.sample_csv(source, second, 10, 2024) == (100, 10)
    assert first.read_text() == second.read_text()
    assert len(pd.read_csv(first)) == 10
