"""Independent per-coordinate Gaussian mutation."""
import torch


def mutate(genes: torch.Tensor, probability: float, sigma: float, latent_clip: float | None = None) -> torch.Tensor:
    if not 0 <= probability <= 1 or sigma < 0:
        raise ValueError("Invalid mutation configuration")
    result = genes + torch.randn_like(genes) * sigma * (torch.rand_like(genes) < probability)
    return result.clamp(-latent_clip, latent_clip) if latent_clip is not None else result
