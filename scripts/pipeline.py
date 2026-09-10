#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from minionerec_utils.config import REPO_ROOT, load_config, resolve_data_root, resolve_path
from minionerec_utils.checkpoints import cleanup_checkpoints, latest_checkpoint


@dataclass
class Stage:
    key: str
    label: str
    log_name: str
    command: Callable[[], list[str]]
    valid: Callable[[], bool]
    expensive: bool = False
    on_success: Callable[[], None] | None = None


def nonempty(*paths: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def remove_generated_path(path: Path, data_root: Path, state_path: Path) -> None:
    resolved = path.resolve()
    root = data_root.resolve()
    if resolved != state_path.resolve() and (resolved == root or not resolved.is_relative_to(root)):
        raise RuntimeError(f"Refusing --fresh deletion outside DATA_ROOT: {resolved}")
    if path.exists():
        shutil.rmtree(path)
        print(f"FRESH removed generated output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/movielens1m.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--from-stage", type=int, choices=range(1, 14))
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_data_root(config, require_exists=True)
    raw = resolve_path(config, "paths.raw_dir", root)
    processed = resolve_path(config, "paths.processed_dir", root)
    mini = resolve_path(config, "paths.minionerec_dir", root)
    output = resolve_path(config, "paths.output_dir", root)
    results = resolve_path(config, "paths.results_dir", root)
    state = REPO_ROOT / ".state"
    if args.fresh:
        for path in (output, results, state):
            remove_generated_path(path, root, state)
    state.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "logs" / "MovieLens1M" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / "pipeline.log"
    py = sys.executable
    dataset = config["dataset"]
    item = processed / f"{dataset}.item.json"
    embedding = processed / f"{dataset}.emb.npy"
    index = processed / f"{dataset}.index.json"
    index_stats = processed / f"{dataset}.index.stats.json"
    info = mini / "info" / f"{dataset}.txt"
    sft_dir = output / "sft"
    rl_dir = output / "rl"
    sft_predictions = results / "sft_predictions.json"
    rl_predictions = results / "rl_predictions.json"
    full_test_csv = mini / "test" / f"{dataset}.csv"
    evaluation_sample_size = int(config.get("evaluation", {}).get("sample_size", 0) or 0)
    evaluation_sample_seed = int(config.get("evaluation", {}).get("sample_seed", config["seed"]))
    evaluation_csv = (results / "evaluation_samples" /
                      f"{dataset}_test_{evaluation_sample_size}_seed{evaluation_sample_seed}.csv"
                      if evaluation_sample_size > 0 else full_test_csv)
    rq_dir = output / "rqvae"
    download_config = config["download"]
    os.environ.update({
        "MOVIELENS_URL": str(download_config["url"]),
        "MOVIELENS_DOWNLOAD_RETRIES": str(download_config["retries"]),
        "MOVIELENS_RETRY_DELAY_SECONDS": str(download_config["retry_delay_seconds"]),
        "MOVIELENS_MIN_ARCHIVE_BYTES": str(download_config["minimum_archive_bytes"]),
        "MOVIELENS_RAW_PARENT": str(raw.parent),
        "DELETE_DATASET_ARCHIVE_AFTER_EXTRACT": str(config["storage"]["delete_dataset_archive_after_extract"]).lower(),
    })
    if download_config.get("sha256"):
        os.environ["MOVIELENS_SHA256"] = str(download_config["sha256"])

    def model_reference(value: str) -> str:
        return str(value).replace("${DATA_ROOT}", str(root))

    embedding_model = model_reference(str(config["paths"]["embedding_model"]))
    base_model = model_reference(str(config["model"]["base_model"]))

    def rq_checkpoint() -> Path:
        matches = sorted(rq_dir.glob("*/best_collision_model.pth"), key=lambda p: p.stat().st_mtime)
        return matches[-1] if matches else rq_dir / "MISSING.pth"

    def summary() -> dict:
        try:
            return json.loads((processed / f"{dataset}.summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def valid_preprocessing() -> bool:
        values = summary()
        try:
            items = json.loads(item.read_text())
            return (len(items) == values["num_items"] and sorted(map(int, items)) == list(range(len(items)))
                    and all((processed / f"{dataset}.{split}.inter").is_file() for split in ("train", "valid", "test")))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def valid_embedding() -> bool:
        try:
            import numpy as np
            array = np.load(embedding, mmap_mode="r")
            return array.ndim == 2 and array.shape[0] == summary()["num_items"] and np.isfinite(array).all()
        except (OSError, ValueError, KeyError):
            return False

    def valid_index() -> bool:
        try:
            values = json.loads(index.read_text())
            collision = json.loads(index_stats.read_text())
            max_collisions = int(config.get("rqvae", {}).get("max_collisions", 0))
            pattern = re.compile(r"^<([abc])_\d+>$")
            return (len(values) == summary()["num_items"] and sorted(map(int, values)) == list(range(len(values)))
                    and collision["collision_count"] <= max_collisions
                    and collision["total_items"] == len(values)
                    and collision["unique_sid"] == len(values) - collision["collision_count"]
                    and all(len(tokens) == 3 and [pattern.fullmatch(token).group(1) for token in tokens] == list("abc")
                            for tokens in values.values()))
        except (OSError, ValueError, KeyError, AttributeError, json.JSONDecodeError):
            return False

    def valid_conversion() -> bool:
        try:
            return (len(info.read_text().splitlines()) == summary()["num_items"]
                    and all((mini / split / f"{dataset}.csv").is_file() for split in ("train", "valid", "test")))
        except (OSError, KeyError):
            return False

    def evaluation_command(model: Path, prediction_path: Path, metric_path: Path) -> list[str]:
        stop_flag = "--stop-on-invalid" if config["evaluation"]["stop_on_invalid"] else "--no-stop-on-invalid"
        command: list[str] = []
        if evaluation_sample_size > 0:
            command.extend([py, "scripts/sample_evaluation_csv.py", "--input", str(full_test_csv),
                            "--output", str(evaluation_csv), "--size", str(evaluation_sample_size),
                            "--seed", str(evaluation_sample_seed), "&&"])
        command.extend([py, "evaluate.py", "--base_model", str(model), "--info_file", str(info),
                        "--category", dataset, "--test_data_path", str(evaluation_csv),
                        "--result_json_data", str(prediction_path), "--batch_size", str(config["evaluation"]["batch_size"]),
                        "--num_beams", str(config["evaluation"]["num_beams"]), "&&", py,
                        "scripts/evaluate_metrics.py", "--predictions", str(prediction_path), "--info", str(info),
                        "--output", str(metric_path), "--cutoffs", *map(str, config["evaluation"]["cutoffs"]), stop_flag])
        return command

    def resume_argument(directory: Path) -> list[str]:
        checkpoint = latest_checkpoint(directory)
        if checkpoint:
            print(f"Resuming trainer from {checkpoint}")
            return ["--resume_from_checkpoint", str(checkpoint)]
        return []

    def sft_command() -> list[str]:
        return [py, "scripts/check_sid_tokenizer.py", "--model", base_model, "--index", str(index),
                "--seed", str(config["seed"]), "&&", py, "sft.py", "--base_model", base_model,
                "--train_file", str(mini / "train" / f"{dataset}.csv"), "--eval_file", str(mini / "valid" / f"{dataset}.csv"),
                "--output_dir", str(sft_dir), "--category", dataset, "--sid_index_path", str(index),
                "--item_meta_path", str(item), "--num_epochs", str(config["sft"]["epochs"]),
                "--batch_size", str(config["sft"]["batch_size"]), "--micro_batch_size", str(config["sft"]["micro_batch_size"]),
                "--learning_rate", str(config["sft"]["learning_rate"]), "--cutoff_len", str(config["sft"]["max_length"]),
                "--save_total_limit", str(config["sft"]["save_total_limit"]), "--save_steps", str(config["sft"]["save_steps"]),
                "--bf16", str(config["sft"]["bf16"]), "--gradient_accumulation_steps", str(config["sft"]["gradient_accumulation_steps"]),
                "--gradient_checkpointing", str(config["sft"]["gradient_checkpointing"]), "--seed", str(config["seed"]),
                *resume_argument(sft_dir), "&&", py, "scripts/check_sid_tokenizer.py", "--model", str(sft_dir / "final_checkpoint"),
                "--index", str(index), "--seed", str(config["seed"]), "--require-existing"]

    def rl_command() -> list[str]:
        return [py, "rl.py", "--model_path", str(sft_dir / "final_checkpoint"), "--train_file", str(mini / "train" / f"{dataset}.csv"),
                "--eval_file", str(mini / "valid" / f"{dataset}.csv"), "--info_file", str(info), "--category", dataset,
                "--output_dir", str(rl_dir), "--sid_index_path", str(index), "--item_meta_path", str(item),
                "--max_samples", str(config["rl"]["max_samples"]), "--num_train_epochs", str(config["rl"]["epochs"]),
                "--learning_rate", str(config["rl"]["learning_rate"]), "--save_total_limit", str(config["rl"]["save_total_limit"]),
                "--train_batch_size", str(config["rl"]["train_batch_size"]), "--eval_batch_size", str(config["rl"]["eval_batch_size"]),
                "--gradient_accumulation_steps", str(config["rl"]["gradient_accumulation_steps"]),
                "--num_generations", str(config["rl"]["num_generations"]), "--seed", str(config["seed"]),
                "--save_steps", str(config["rl"]["save_steps"]), "--bf16", str(config["rl"]["bf16"]),
                "--reward_type", str(config["rl"]["reward_type"]), "--beam_search", str(config["rl"]["beam_search"]),
                "--sync_ref_model", str(config["rl"]["sync_ref_model"]), "--beta", str(config["rl"]["beta"]),
                *resume_argument(rl_dir)]

    stages = [
        Stage("environment", "Environment Check", "environment.log",
              lambda: [py, "scripts/check_environment.py", "&&", "env", "-u", "MOVIELENS_RAW_PARENT", py, "-m", "pytest", "-q", "tests/test_movielens_downloader.py",
                       "tests/test_movielens_pipeline.py", "tests/test_constrained_decoding.py",
                       "tests/test_movielens_entrypoints.py", "tests/test_training_hardening.py"],
              lambda: True),
        Stage("download", "MovieLens Download", "download.log",
              lambda: ["bash", "scripts/download_movielens1m.sh"],
              lambda: nonempty(raw / "ratings.dat", raw / "movies.dat", raw / "users.dat")),
        Stage("data", "Data Preprocessing", "data.log",
              lambda: [py, "data/movielens1m_data_process.py", "--config", args.config],
              valid_preprocessing),
        Stage("embedding", "Item Embedding", "embedding.log",
              lambda: [py, "rq/text2emb/item_text2emb.py", "--item-file", str(item), "--output-file", str(embedding),
                       "--model", embedding_model, "--batch-size", str(config["embedding"]["batch_size"]),
                       "--max-length", str(config["embedding"]["max_length"]), "--dtype", str(config["embedding"]["dtype"])],
              valid_embedding, True),
        Stage("rqvae", "RQ-VAE", "rqvae.log",
              lambda: [py, "rq/rqvae.py", "--data_path", str(embedding), "--ckpt_dir", str(rq_dir),
                       "--lr", str(config["rqvae"]["learning_rate"]), "--epochs", str(config["rqvae"]["epochs"]),
                       "--batch_size", str(config["rqvae"]["batch_size"]), "--e_dim", str(config["rqvae"]["e_dim"]),
                       "--num_emb_list", *map(str, config["rqvae"]["num_emb_list"]), "--save_limit", str(config["rqvae"]["save_total_limit"]),
                       "--layers", *map(str, config["rqvae"]["layers"]), "--loss_type", str(config["rqvae"]["loss_type"]),
                       "--kmeans_init", str(config["rqvae"]["kmeans_init"]),
                       "--kmeans_iters", str(config["rqvae"]["kmeans_iters"]), "--device", str(config["rqvae"]["device"])],
              lambda: rq_checkpoint().is_file(), True),
        Stage("sid", "Semantic ID", "sid.log",
              lambda: [py, "rq/generate_indices.py", "--data-path", str(embedding), "--checkpoint", str(rq_checkpoint()),
                       "--output-file", str(index)], valid_index, True),
        Stage("convert", "Dataset Conversion", "convert.log",
              lambda: [py, "convert_dataset.py", "--data-dir", str(processed), "--output-dir", str(mini),
                       "--dataset-name", dataset],
              valid_conversion),
        Stage("sft", "SFT", "sft.log",
              sft_command, lambda: (sft_dir / "final_checkpoint" / "config.json").is_file(), True,
              lambda: cleanup_checkpoints(sft_dir)),
        Stage("eval_sft", "SFT Evaluation", "eval_sft.log",
              lambda: evaluation_command(sft_dir / "final_checkpoint", sft_predictions, results / "sft_metrics.json"),
              lambda: valid_metrics(results / "sft_metrics.json", config["evaluation"]["cutoffs"],
                                    config["evaluation"]["stop_on_invalid"]), True),
        Stage("rl", "Recommendation RL", "rl.log",
              lambda: ([py, "-c", "print('RL disabled; training skipped')"] if not config["rl"]["enabled"] else rl_command()),
              lambda: not config["rl"]["enabled"] or (rl_dir / "final_checkpoint" / "config.json").is_file(), True,
              lambda: None if not config["rl"]["enabled"] else cleanup_checkpoints(rl_dir)),
        Stage("eval_rl", "RL Evaluation", "eval_rl.log",
              lambda: ([py, "-c", "print('RL disabled; evaluation skipped')"] if not config["rl"]["enabled"] else
                       evaluation_command(rl_dir / "final_checkpoint", rl_predictions, results / "rl_metrics.json")),
              lambda: not config["rl"]["enabled"] or valid_metrics(
                  results / "rl_metrics.json", config["evaluation"]["cutoffs"], config["evaluation"]["stop_on_invalid"]), True),
        Stage("comparison", "Comparison", "comparison.log",
              lambda: [py, "scripts/finalize_results.py", "--results-dir", str(results), "--dataset", dataset],
              lambda: nonempty(results / "comparison.json")),
        Stage("report", "Final Report", "final_report.log",
              lambda: [py, "scripts/finalize_results.py", "--results-dir", str(results), "--dataset", dataset, "--report"],
              lambda: nonempty(results / "final_report.md")),
    ]

    if args.force:
        for marker in state.glob("*.done"):
            marker.unlink()

    print("=" * 50)
    print("MiniOneRec MovieLens1M Reproduction")
    print("=" * 50)
    for number, stage in enumerate(stages, 1):
        if args.from_stage and number < args.from_stage:
            print(f"[{number:02d}/13] {stage.label:<24} SKIPPED (--from-stage)")
            continue
        if stage.key == "sft" and not valid_index():
            collision_count = None
            if index_stats.is_file():
                collision_count = json.loads(index_stats.read_text()).get("collision_count")
            max_collisions = int(config.get("rqvae", {}).get("max_collisions", 0))
            raise SystemExit(f"ERROR: Semantic ID collisions exceed allowed threshold (max={max_collisions}) "
                             f"(collision_count={collision_count})")
        marker = state / f"{stage.key}.done"
        if marker.exists() and stage.valid() and not args.force and stage.key != "environment":
            print(f"[{number:02d}/13] {stage.label:<24} PASS (resume)")
            continue
        if marker.exists() and not stage.valid():
            marker.unlink()
        if stage.expensive:
            run_stream(["bash", "scripts/check_disk.sh", str(root), str(config["storage"]["minimum_free_gb"])], log_dir / stage.log_name, pipeline_log)
        print(f"[{number:02d}/13] {stage.label:<24} RUNNING")
        command = stage.command()
        run_stream(command, log_dir / stage.log_name, pipeline_log)
        if not stage.valid():
            if stage.key == "sid" and index_stats.is_file():
                collision_count = json.loads(index_stats.read_text()).get("collision_count")
                max_collisions = int(config.get("rqvae", {}).get("max_collisions", 0))
                raise SystemExit(f"ERROR: Semantic ID collision_count={collision_count} exceeds max={max_collisions}; "
                                 "SFT is blocked")
            raise SystemExit(f"ERROR: stage {stage.label} finished without valid required outputs")
        if stage.on_success:
            stage.on_success()
        marker.touch()
        print(f"[{number:02d}/13] {stage.label:<24} PASS")


def valid_metrics(path: Path, cutoffs: list[int], stop_on_invalid: bool) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        values = json.loads(path.read_text())
        required = {f"{metric}@{cutoff}" for metric in ("HR", "NDCG") for cutoff in cutoffs}
        return required.issubset(values) and (not stop_on_invalid or values.get("valid_experiment") is True)
    except (OSError, json.JSONDecodeError):
        return False


def run_stream(command: list[str], stage_log: Path, pipeline_log: Path) -> None:
    shell = "&&" in command
    invocation: str | list[str] = " ".join(shlex.quote(value) if value != "&&" else value for value in command) if shell else command
    env = os.environ.copy()
    with stage_log.open("a", encoding="utf-8") as stage_handle, pipeline_log.open("a", encoding="utf-8") as pipeline_handle:
        process = subprocess.Popen(invocation, cwd=REPO_ROOT, env=env, shell=shell, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout
        for line in process.stdout:
            print(line, end="")
            stage_handle.write(line)
            pipeline_handle.write(line)
        if process.wait() != 0:
            raise SystemExit(f"Command failed: {invocation}")


if __name__ == "__main__":
    main()
