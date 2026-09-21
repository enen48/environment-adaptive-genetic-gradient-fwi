"""Cycle consistency is a heuristic, not proof of geological feasibility."""
from __future__ import annotations
import torch


class LatentQC:
    """Hard, calibrated-percentile or accepted-sample adaptive thresholds."""
    def __init__(self, config: dict) -> None:
        self.mode = config.get("mode","hard")
        if self.mode not in {"hard","percentile","adaptive"}:
            raise ValueError("Invalid latent QC mode")
        self.threshold = float(config.get("threshold",4.))
        self.percentile = float(config.get("percentile",95.))
        self.minimum = float(config.get("min_threshold",.1))
        self.calibrated: float | None = None
        if self.threshold <= 0 or not 0 < self.percentile <= 100 or self.minimum <= 0:
            raise ValueError("Invalid latent QC thresholds")

    def calibrate(self, norms: torch.Tensor) -> None:
        finite = norms.detach()[torch.isfinite(norms)]
        if not len(finite):
            raise ValueError("Cannot calibrate QC from nonfinite cycles")
        self.calibrated = max(self.minimum,float(torch.quantile(finite,self.percentile/100)))

    def update(self, accepted_norms: torch.Tensor) -> None:
        if self.mode == "adaptive" and accepted_norms.numel():
            old = self.calibrated
            self.calibrate(accepted_norms)
            self.calibrated = self.calibrated if old is None else .9*old+.1*self.calibrated

    def limit(self, environment_threshold: float) -> float:
        if self.mode == "hard":
            return min(self.threshold,environment_threshold)
        if self.calibrated is None:
            raise ValueError("Percentile/adaptive QC must be calibrated from prior genes")
        return max(self.minimum,self.calibrated*environment_threshold/self.threshold)

    def state_dict(self) -> dict:
        return dict(vars(self))
