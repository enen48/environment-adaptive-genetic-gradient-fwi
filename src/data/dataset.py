"""Memory-mapped NumPy velocity models; only requested samples enter RAM."""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .preprocessing import validate_velocity


class VelocityDataset(Dataset[Tensor]):
    """Read .npy arrays shaped N,1,H,W or N,H,W, returning physical m/s.

    Directory input discovers .npy files; pass explicit files to avoid mixing
    train and validation shards. Resizing is opt-in via ``shape`` and preserves
    numerical velocity units. No normalization, clamping, or augmentation occurs.
    """

    def __init__(self, path: str | Path | Sequence[str | Path],
                 shape: tuple[int, int] | None = None,
                 vmin: float | None = None, vmax: float | None = None) -> None:
        if isinstance(path, (str, Path)):
            root = Path(path)
            paths = sorted(root.glob("*.npy")) if root.is_dir() else [root]
        else:
            paths = [Path(item) for item in path]
        if not paths:
            raise ValueError("No .npy velocity shards found")
        self.paths, self.shape = paths, shape
        self.vmin, self.vmax = vmin, vmax
        self.arrays: list[np.ndarray] = []
        self.ends: list[int] = []
        count = 0
        for file in paths:
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            if array.ndim not in (3, 4) or (array.ndim == 4 and array.shape[1] != 1):
                raise ValueError(f"Expected velocity N,1,H,W or N,H,W, got {array.shape} in {file}")
            if not np.issubdtype(array.dtype, np.floating) or len(array) == 0:
                raise ValueError(f"Velocity shard must contain floating-point models: {file}")
            self.arrays.append(array)
            count += len(array)
            self.ends.append(count)

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        offset = 0 if shard == 0 else self.ends[shard - 1]
        # Copy one sample: read-only mmap views must never be mutated by Torch.
        sample = torch.from_numpy(np.array(self.arrays[shard][index - offset], dtype=np.float32, copy=True))
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        validate_velocity(sample)
        if self.vmin is not None and sample.min() < self.vmin:
            raise ValueError("Dataset contains velocity below configured velocity_min")
        if self.vmax is not None and sample.max() > self.vmax:
            raise ValueError("Dataset contains velocity above configured velocity_max")
        if self.shape is not None and tuple(sample.shape[-2:]) != tuple(self.shape):
            sample = F.interpolate(sample.unsqueeze(0), size=self.shape, mode="bilinear", align_corners=False)[0]
        return sample
