"""Portable YAML configuration and E-drive-first experiment storage."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def merge(base: dict, update: dict) -> dict:
    """Recursively merge dictionaries without mutating the originals."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML; `extends` resolves from project root, then config directory."""
    file = Path(path)
    if not file.is_absolute() and not file.exists():
        file = ROOT / file
    file = file.resolve()
    seen = set() if _seen is None else _seen.copy()
    if file in seen:
        raise ValueError(f"Cyclic config inheritance: {file}")
    seen.add(file)
    config = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = ROOT / parent if (ROOT / parent).exists() else file.parent / parent
        return merge(load_config(parent_path, seen), config)
    return config


def storage_root() -> Path:
    """Use FWI_HOME, an existing E-drive project directory, or the repo root."""
    if os.environ.get("FWI_HOME"):
        return Path(os.environ["FWI_HOME"]).expanduser().resolve()
    e_drive = Path("E:/Codex/environment-adaptive-genetic-gradient-fwi")
    return e_drive if os.name == "nt" and e_drive.exists() else ROOT


def storage_path(path: str | Path) -> Path:
    """Resolve generated files without redirecting source/config paths."""
    path = Path(path)
    return path if path.is_absolute() else storage_root() / path


def save_config(config: dict, path: str | Path) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
