"""Spatially adaptive velocity-model encoder."""
from __future__ import annotations

from torch import Tensor, nn

from src.data.preprocessing import normalize_velocity


class Encoder(nn.Module):
    """Compress physical velocity models into a configurable latent vector."""

    def __init__(self, latent_dim: int = 32, channels: int = 16,
                 velocity_min: float = 1500., velocity_max: float = 4500.) -> None:
        super().__init__()
        self.velocity_min, self.velocity_max = velocity_min, velocity_max
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.projection = nn.Linear(channels * 4 * 4 * 4, latent_dim)

    def forward(self, model: Tensor) -> Tensor:
        if model.ndim != 4 or model.shape[1] != 1 or min(model.shape[-2:]) < 8:
            raise ValueError("Encoder expects (N,1,H,W) with H,W>=8")
        normalized = normalize_velocity(model, self.velocity_min, self.velocity_max)
        return self.projection(self.features(normalized))
