"""Basic differentiable constraints with a future custom-constraint hook."""
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn.functional as F


@dataclass
class GeologicalResult:
    """Scalar violation penalty and feasibility decision."""
    penalty: torch.Tensor
    passed: bool
    measurements: dict[str,float]


class GeologicalQC:
    """Velocity range, vertical roughness, gradient, TV and optional depth trend."""
    def __init__(self, config: dict, extra_constraints: list[Callable] | None = None) -> None:
        self.config,self.extra_constraints = config,extra_constraints or []

    def __call__(self, velocity: torch.Tensor) -> GeologicalResult:
        c = self.config
        vmin,vmax = c.get("velocity_min",1500.),c.get("velocity_max",4500.)
        scale = vmax-vmin
        if scale <= 0:
            raise ValueError("Invalid velocity bounds")
        dz,dx = velocity[...,1:,:]-velocity[...,:-1,:],velocity[...,:,1:]-velocity[...,:,:-1]
        vertical = dz.abs().mean()
        maximum = torch.maximum(dz.abs().max(),dx.abs().max())
        tv = vertical+dx.abs().mean()
        penalty = (F.relu(vmin-velocity).square()+F.relu(velocity-vmax).square()).mean()/scale**2
        finite = bool(torch.isfinite(velocity).all())
        passed = finite and bool((velocity>=vmin).all() and (velocity<=vmax).all())
        for value,key in [(vertical,"vertical_smoothness"),(maximum,"max_gradient"),(tv,"tv_threshold")]:
            bound = c.get(key)
            if bound is not None:
                penalty = penalty + (F.relu(value-bound)/scale).square()
                passed = passed and bool(value <= bound)
        if c.get("monotonic",False):
            # Check mean horizontal depth trend, not every pixel in an anomaly.
            reversals = F.relu(-dz.mean(dim=-1))
            penalty = penalty + (reversals/scale).square().mean()
            passed = passed and bool(reversals.max() <= c.get("monotonic_tolerance",20.))
        for constraint in self.extra_constraints:
            extra_penalty,extra_pass = constraint(velocity)
            penalty,passed = penalty+extra_penalty,passed and bool(extra_pass)
        return GeologicalResult(penalty,passed,{"vertical_smoothness":float(vertical.detach()),
            "maximum_gradient":float(maximum.detach()),"total_variation":float(tv.detach())})
