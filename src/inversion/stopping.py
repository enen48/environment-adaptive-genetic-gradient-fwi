"""Stopping criteria only compare a fixed reference objective across generations."""
from dataclasses import dataclass
import math


@dataclass
class Stopping:
    """Track reference improvement, stagnation and failed memory recoveries."""
    config: dict
    best: float = float("inf")
    stale: int = 0
    failed_injections: int = 0

    def update(self, reference_loss: float, injection_success: bool | None = None,
               validation_loss: float | None = None, progress: float = 1.) -> str | None:
        if math.isfinite(reference_loss) and reference_loss < self.best-self.config.get("epsilon",1e-6):
            self.best,self.stale = reference_loss,0
        else:
            self.stale += 1
        if progress < self.config.get("min_progress",0.):
            self.stale = 0
        if injection_success is not None:
            self.failed_injections = 0 if injection_success else self.failed_injections+1
        if reference_loss <= self.config.get("target",-float("inf")):
            return "target_fitness"
        if self.stale >= self.config.get("patience",1000000):
            return "reference_stagnation"
        if self.failed_injections >= self.config.get("failed_injections",1000000):
            return "collapse_injection_without_improvement"
        threshold = self.config.get("validation_threshold")
        if threshold is not None and validation_loss is not None and validation_loss <= threshold:
            return "validation_threshold"
        return None
