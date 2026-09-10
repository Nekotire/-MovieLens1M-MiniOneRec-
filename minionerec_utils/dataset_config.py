from __future__ import annotations

from pathlib import Path

from .config import load_config, resolve_data_root, resolve_path


CATEGORY_NAMES = {
    "MovieLens1M": "movies",
}


def category_name(dataset: str) -> str:
    return CATEGORY_NAMES.get(dataset, dataset.replace("_", " ").lower())


def dataset_paths(config_path: str | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    root = resolve_data_root(config)
    return {
        "data_root": root,
        "raw_dir": resolve_path(config, "paths.raw_dir", root),
        "processed_dir": resolve_path(config, "paths.processed_dir", root),
        "minionerec_dir": resolve_path(config, "paths.minionerec_dir", root),
        "output_dir": resolve_path(config, "paths.output_dir", root),
        "results_dir": resolve_path(config, "paths.results_dir", root),
    }
