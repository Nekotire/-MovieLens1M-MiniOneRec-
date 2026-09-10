from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path

import yaml

from minionerec_utils.checkpoints import cleanup_checkpoints, latest_checkpoint
from minionerec_utils.rl_data import deterministic_sample_csv
from scripts.check_sid_tokenizer import validate_sid_tokenizer


REPO = Path(__file__).resolve().parents[1]


def test_latest_checkpoint_and_cleanup(tmp_path):
    output = tmp_path / "sft"
    for name in ("checkpoint-2", "checkpoint-40", "final_checkpoint", "checkpoint-bad"):
        (output / name).mkdir(parents=True)
    assert latest_checkpoint(output).name == "checkpoint-40"
    removed = cleanup_checkpoints(output)
    assert [path.name for path in removed] == ["checkpoint-2"]
    assert (output / "checkpoint-40").is_dir()
    assert (output / "final_checkpoint").is_dir()


def test_rl_rows_are_sampled_deterministically_before_dataset_build(tmp_path):
    source = tmp_path / "train.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user", "history", "target"])
        writer.writerows([[number, f"[{number}]", number + 1] for number in range(100)])
    first, count, total = deterministic_sample_csv(source, tmp_path / "first.csv", 20, 2024)
    second, _, _ = deterministic_sample_csv(source, tmp_path / "second.csv", 20, 2024)
    assert count == 20 and total == 100
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text().splitlines()) == 21


def test_data_root_selection_uses_most_free_space_and_persistence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("select_data_root", REPO / "scripts/select_data_root.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    small, large = tmp_path / "small", tmp_path / "large"
    small.mkdir()
    large.mkdir()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: usage._replace(free=1 if path == small else 2))
    persisted = tmp_path / "persisted"
    assert module.select_data_root(None, persisted, (small, large)) == large
    persisted.write_text(str(small))
    assert module.select_data_root(None, persisted, (large,)) == small


class SidTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.ids = {}

    def add_tokens(self, tokens):
        for token in tokens:
            self.ids.setdefault(token, len(self.ids) + 1)
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def get_vocab(self):
        return dict(self.ids)

    def __call__(self, text, add_special_tokens=False):
        values = []
        remaining = text
        while remaining:
            token = next((value for value in self.ids if remaining.startswith(value)), None)
            if token is None:
                values.append(1000 + ord(remaining[0]))
                remaining = remaining[1:]
            else:
                values.append(self.ids[token])
                remaining = remaining[len(token):]
        return {"input_ids": values}


def test_sid_tokenizer_preflight_checks_atomic_tokens_and_tree():
    indices = {"0": ["<a_1>", "<b_2>", "<c_3>"], "1": ["<a_4>", "<b_5>", "<c_6>"]}
    tokenizer = SidTokenizer()
    validate_sid_tokenizer(tokenizer, indices, seed=2024, sample_size=1)
    validate_sid_tokenizer(tokenizer, indices, seed=2024, sample_size=2, require_existing=True)


def test_formal_yaml_parameters_are_wired_to_runtime():
    config = yaml.safe_load((REPO / "config/movielens1m.yaml").read_text())
    assert config["sft"]["epochs"] == 2
    expected_rl = {"reward_type": "ranking", "beam_search": True,
                   "sync_ref_model": True, "beta": 0.001}
    assert expected_rl.items() <= config["rl"].items()
    pipeline = (REPO / "scripts/pipeline.py").read_text()
    for expression in (
        'config["rqvae"]["kmeans_init"]', 'config["embedding"]["dtype"]',
        'config["evaluation"]["cutoffs"]', 'config["evaluation"]["stop_on_invalid"]',
        'config["rl"]["reward_type"]', 'config["rl"]["beam_search"]',
        'config["rl"]["sync_ref_model"]', 'config["rl"]["beta"]',
    ):
        assert expression in pipeline
    assert 'config.get("rqvae", {}).get("max_collisions", 0)' in pipeline
    assert config["rqvae"]["max_collisions"] == 3
    assert config["evaluation"]["sample_size"] == 10000
    assert config["rl"]["enabled"] is False
    assert "RL disabled; training skipped" in pipeline
    assert "--fresh" in pipeline and "--resume_from_checkpoint" in pipeline
    rl_source = (REPO / "rl.py").read_text()
    assert "AutoModelForCausalLM" not in rl_source
    assert 'report_to="none"' in rl_source
    assert rl_source.index("deterministic_sample_csv(") < rl_source.index("SidDataset(sampled_train_file")
