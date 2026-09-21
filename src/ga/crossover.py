"""Arithmetic and BLX-alpha crossover in latent coordinates."""
import torch


def crossover(a: torch.Tensor, b: torch.Tensor, probability: float, method: str = "blend", alpha: float = 0.2) -> torch.Tensor:
    if method not in {"arithmetic", "blend"} or not 0 <= probability <= 1 or alpha < 0:
        raise ValueError("Invalid crossover configuration")
    weights = torch.rand_like(a)
    if method == "blend":
        weights = weights * (1 + 2 * alpha) - alpha
    child = weights * a + (1 - weights) * b
    return torch.where(torch.rand((len(a), 1), device=a.device) < probability, child, a)
