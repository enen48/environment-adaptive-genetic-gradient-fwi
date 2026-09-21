"""Differentiable bounded physical-velocity decoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.data.preprocessing import denormalize_velocity


class Decoder(nn.Module):
    """Decode z to m/s; output resolution is independent of latent dimension."""

    def __init__(self, latent_dim: int = 32, channels: int = 16,
                 velocity_min: float = 1500., velocity_max: float = 4500.,
                 output_shape: tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        self.latent_dim, self.channels = latent_dim, channels
        self.velocity_min, self.velocity_max = velocity_min, velocity_max
        self.output_shape = tuple(output_shape)
        self.projection = nn.Linear(latent_dim, channels * 4 * 4 * 4)
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels * 2, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, z: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Decoder expects (N,{self.latent_dim})")
        target = tuple(shape or self.output_shape)
        if len(target) != 2 or min(target) < 8:
            raise ValueError("Decoder H,W must be >=8")
        features = self.projection(z).reshape(-1, self.channels * 4, 4, 4)
        logits = F.interpolate(self.features(features), size=target, mode="bilinear", align_corners=False)
        return denormalize_velocity(torch.sigmoid(logits), self.velocity_min, self.velocity_max)
