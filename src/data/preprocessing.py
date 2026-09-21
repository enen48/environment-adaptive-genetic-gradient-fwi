"""Physical velocity units and explicit, reversible normalization."""
from __future__ import annotations

import torch
from torch import Tensor


def normalize_velocity(model: Tensor, vmin: float, vmax: float) -> Tensor:
    """Map m/s to [0, 1] without silently clipping out-of-range values."""
    if vmax <= vmin:
        raise ValueError("velocity_max must exceed velocity_min")
    return (model - vmin) / (vmax - vmin)


def denormalize_velocity(model: Tensor, vmin: float, vmax: float) -> Tensor:
    """Map normalized values back to m/s."""
    if vmax <= vmin:
        raise ValueError("velocity_max must exceed velocity_min")
    return vmin + model * (vmax - vmin)


def validate_velocity(model: Tensor) -> None:
    """Reject malformed physical velocity tensors before training or physics."""
    if model.ndim not in (3, 4) or model.shape[-3] != 1:
        raise ValueError("Expected (1,H,W) or (N,1,H,W) velocity models")
    if not torch.isfinite(model).all() or (model <= 0).any():
        raise ValueError("Velocity models must be finite and positive in m/s")
