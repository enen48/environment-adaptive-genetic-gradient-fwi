"""Population contracts."""
from dataclasses import dataclass
import torch


@dataclass
class Population:
    """N latent vectors with fitness evaluated in one common environment."""
    genes: torch.Tensor
    fitness: torch.Tensor

    def __post_init__(self) -> None:
        if self.genes.ndim != 2 or self.fitness.shape != (len(self.genes),):
            raise ValueError("Expected genes [N,L] and fitness [N]")
        if len(self.genes) < 2 or not torch.isfinite(self.genes).all():
            raise ValueError("Population needs at least two finite genes")
