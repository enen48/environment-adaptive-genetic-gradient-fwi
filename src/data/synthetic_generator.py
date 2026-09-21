"""Small reproducible layered/anomaly velocity models for CPU smoke tests."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def synthetic_velocity_models(count: int, shape: tuple[int, int] = (32, 32),
                              seed: int = 42, vmin: float = 1500.,
                              vmax: float = 4500.) -> Tensor:
    """Generate bounded m/s models shaped (N,1,H,W), using a private RNG.

    The generator combines increasing sediment layers, gently curved interfaces,
    and optional compact anomalies. It is a toy prior, not field geology.
    """
    height, width = map(int, shape)
    if count <= 0 or min(height, width) < 8 or not 0 < vmin < vmax:
        raise ValueError("Require count>0, H/W>=8 and 0<vmin<vmax")
    generator = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(0., 1., height), torch.linspace(0., 1., width), indexing="ij")
    models = []
    for index in range(count):
        layer_count = int(torch.randint(3, 6, (), generator=generator))
        boundaries = torch.sort(torch.rand(layer_count - 1, generator=generator) * 0.8 + 0.1).values
        levels = torch.sort(torch.rand(layer_count, generator=generator) * 0.8 + 0.1).values
        warped = y
        if index % 3:
            phase = float(torch.rand((), generator=generator)) * 2 * math.pi
            warped = y + 0.06 * torch.sin(2 * math.pi * x + phase)
        layer_indices = torch.bucketize(warped.contiguous(), boundaries)
        normalized = levels[layer_indices]
        if index % 2:
            cx, cy = (torch.rand(2, generator=generator) * 0.5 + 0.25).tolist()
            radius = float(torch.rand((), generator=generator) * 0.06 + 0.07)
            sign = 1. if index % 4 == 1 else -1.
            normalized = normalized + sign * 0.12 * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        models.append(vmin + normalized.clamp(0., 1.) * (vmax - vmin))
    return torch.stack(models).unsqueeze(1).to(torch.float32)
