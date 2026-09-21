"""Velocity autoencoder; frozen weights still propagate gradients to latent z."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .decoder import Decoder
from .encoder import Encoder


class AutoEncoder(nn.Module):
    """A:m→z and D:z→m in physical m/s, supporting variable H×W."""

    def __init__(self, latent_dim: int = 32, channels: int = 16,
                 velocity_min: float = 1500., velocity_max: float = 4500.,
                 output_shape: tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        if latent_dim <= 0 or channels <= 0 or not 0 < velocity_min < velocity_max:
            raise ValueError("Invalid latent/channel count or velocity bounds")
        if len(output_shape) != 2 or min(output_shape) < 8:
            raise ValueError("output_shape must have H,W>=8")
        self.latent_dim, self.channels = latent_dim, channels
        self.velocity_min, self.velocity_max = velocity_min, velocity_max
        self.output_shape = tuple(output_shape)
        self.encoder = Encoder(latent_dim, channels, velocity_min, velocity_max)
        self.decoder = Decoder(latent_dim, channels, velocity_min, velocity_max, self.output_shape)

    def encode(self, model: Tensor) -> Tensor:
        """Encode only velocity models, never observed seismic gathers."""
        return self.encoder(model)

    def decode(self, z: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
        """Return physical velocities while preserving ∂D/∂z."""
        return self.decoder(z, shape)

    def forward(self, model: Tensor) -> Tensor:
        return self.decode(self.encode(model), tuple(model.shape[-2:]))

    def freeze(self) -> AutoEncoder:
        """Freeze parameters and switch to eval, without disabling autograd."""
        self.eval()
        self.requires_grad_(False)
        return self

    def architecture_config(self) -> dict[str, Any]:
        """Serializable constructor arguments for portable checkpoints."""
        return {"latent_dim": self.latent_dim, "channels": self.channels,
                "velocity_min": self.velocity_min, "velocity_max": self.velocity_max,
                "output_shape": list(self.output_shape)}


def save_checkpoint(path: str | Path, model: AutoEncoder,
                    config: dict[str, Any], **state: Any) -> None:
    """Atomically save architecture, weights, configuration and training state.

    Extra state should contain only tensors and safe Python primitives, so the
    checkpoint can be loaded with PyTorch's weights_only=True deserializer.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model_state": model.state_dict(),
               "model_config": model.architecture_config(), "config": config, **state}
    temporary = file.with_name(file.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(file)


def load_autoencoder(path: str | Path, device: str | torch.device = "cpu") -> tuple[AutoEncoder, dict[str, Any]]:
    """Load model + original training configuration from a project checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported autoencoder checkpoint format")
    model = AutoEncoder(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload["config"]
