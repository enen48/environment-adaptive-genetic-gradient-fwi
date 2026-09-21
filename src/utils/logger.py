"""Structured JSONL logging with strict finite JSON values."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("".join(json.dumps(json_safe(r), allow_nan=False) + "\n" for r in records), encoding="utf-8")
