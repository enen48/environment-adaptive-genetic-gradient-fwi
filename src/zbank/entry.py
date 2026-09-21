"""Portable, CPU-resident memory entry."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import torch


@dataclass
class BankEntry:
    """Historical quality is meaningful only together with its environment."""
    z: torch.Tensor
    environment: list[float]
    fitness: float
    waveform_loss: float = float("inf")
    cycle_loss: float = 0.0
    geological_score: float = 0.0
    diversity_score: float = 0.0
    generation: int = -1
    frequency_weights: list[float] = field(default_factory=lambda: [0.8, 0.2, 0.0])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reference_fitness: float | None = None

    def __post_init__(self) -> None:
        self.z = self.z.detach().cpu().flatten().clone()
        if not torch.isfinite(self.z).all():
            raise ValueError("Bank cannot contain nonfinite genes")

    def state_dict(self) -> dict:
        return dict(vars(self))
