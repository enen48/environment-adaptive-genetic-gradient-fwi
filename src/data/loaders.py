"""Shared training/validation partition construction."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split
from src.data.dataset import VelocityDataset
from src.data.synthetic_generator import synthetic_velocity_models
from src.utils.config import ROOT, storage_path


def data_path(path: str) -> Path:
    """Prefer explicit/repository inputs; redirect only missing paths to storage."""
    file = Path(path)
    if file.is_absolute() or file.exists():
        return file
    return ROOT / file if (ROOT / file).exists() else storage_path(file)


def make_datasets(config: dict[str, Any]) -> tuple[Dataset | Tensor, Dataset | Tensor]:
    """Build separate synthetic sets or explicitly split real model shards."""
    data = config["data"]
    seed, shape = config.get("seed", 42), tuple(data["shape"])
    kwargs = {"shape": shape, "vmin": data["velocity_min"], "vmax": data["velocity_max"]}
    if data.get("path"):
        dataset = VelocityDataset(data_path(data["path"]), **kwargs)
        if data.get("validation_path"):
            return dataset, VelocityDataset(data_path(data["validation_path"]), **kwargs)
        validation_size = min(int(data.get("validation_samples", max(1, len(dataset) // 5))), len(dataset) - 1)
        if validation_size < 1:
            raise ValueError("External data need at least two models or a validation_path")
        return tuple(random_split(dataset, [len(dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(seed)))
    return (synthetic_velocity_models(int(data.get("samples", 96)), seed=seed, **kwargs),
            synthetic_velocity_models(int(data.get("validation_samples", 16)), seed=seed + 1, **kwargs))


