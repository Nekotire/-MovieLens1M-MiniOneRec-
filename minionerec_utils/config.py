from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("MINIONEREC_CONFIG", REPO_ROOT / "config/movielens1m.yaml"))
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    return config


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def resolve_data_root(config: dict[str, Any] | None = None, require_exists: bool = False) -> Path:
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        path = Path(os.path.expandvars(os.path.expanduser(explicit))).resolve()
    else:
        persisted = REPO_ROOT / ".data_root"
        if persisted.is_file():
            path = Path(persisted.read_text(encoding="utf-8").strip()).expanduser().resolve()
        else:
            configured = (config or {}).get("paths", {}).get("data_root")
            path = Path(os.path.expandvars(os.path.expanduser(str(configured or "/data")))).resolve()
    if require_exists and (not path.is_dir() or not os.access(path, os.W_OK | os.X_OK)):
        raise RuntimeError(f"DATA_ROOT is not a writable directory: {path}")
    return path


def resolve_path(config: dict[str, Any], key: str, data_root: Path | None = None) -> Path:
    root = data_root or resolve_data_root(config)
    raw = str(get_nested(config, key))
    raw = raw.replace("${DATA_ROOT}", str(root))
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
