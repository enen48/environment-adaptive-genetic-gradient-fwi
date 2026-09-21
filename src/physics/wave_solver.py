"""Small differentiable acoustic solver with a padded sponge, not a true PML."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SolverCounters:
    """Distinguish Python calls, candidate-model solves and shot-equivalent work."""
    calls: int = 0
    model_evaluations: int = 0
    shot_evaluations: int = 0


class AcousticSolver(nn.Module):
    """Solve u_tt=v² Δu+f with 2nd-order time/space finite differences.

    velocity: [H,W], [B,H,W] or [B,1,H,W], m/s.
    locations: integer (depth,x); source [S,2] or [S,Nsrc,2], receiver
    [R,2] or [S,R,2]. wavelet [T], [S,T] or [S,Nsrc,T].
    Return [B,S,R,T]. Wavelet values are discrete update increments.
    `pml_width` is an exterior sponge width, retained as a backend-compatible
    configuration name, not a claim of mathematically perfect matching.
    """
    def __init__(self, dx: float = 10., dt: float = 0.001, pml_width: int = 6,
                 damping: float = 0.18, cfl_limit: float = 0.7) -> None:
        super().__init__()
        if dx <= 0 or dt <= 0 or pml_width < 0 or damping < 0 or not 0 < cfl_limit <= 1/math.sqrt(2):
            raise ValueError("Invalid solver parameters")
        self.dx, self.dt, self.pml_width, self.damping, self.cfl_limit = dx,dt,pml_width,damping,cfl_limit
        self.counters = SolverCounters()

    def reset_counters(self) -> None:
        self.counters = SolverCounters()

    def counter_state(self) -> dict:
        return asdict(self.counters)

    @staticmethod
    def _locations(locations: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
        if locations.dtype not in (torch.int32, torch.int64) or locations.shape[-1] != 2:
            raise ValueError("Locations must be integer (...,2) depth,x coordinates")
        if (locations < 0).any() or (locations[...,0] >= h).any() or (locations[...,1] >= w).any():
            raise ValueError("Location outside physical model")
        return locations.to(device=device, dtype=torch.long)

    def forward(self, velocity_model: torch.Tensor, source_locations: torch.Tensor,
                receiver_locations: torch.Tensor, source_wavelet: torch.Tensor) -> torch.Tensor:
        v = velocity_model
        if v.ndim == 2:
            v = v[None,None]
        elif v.ndim == 3:
            v = v[:,None]
        if v.ndim != 4 or v.shape[1] != 1 or min(v.shape[-2:]) < 3 or not v.is_floating_point():
            raise ValueError("Velocity must be floating [B,1,H,W] with H,W>=3")
        if not torch.isfinite(v).all() or v.min().item() <= 0:
            raise ValueError("Velocity must be finite and positive")
        cfl = v.detach().max().item()*self.dt/self.dx
        if cfl > self.cfl_limit:
            raise ValueError(f"CFL {cfl:.4f} exceeds {self.cfl_limit}; reduce dt")
        batch,_,h,w = v.shape
        src = self._locations(source_locations,h,w,v.device)
        rec = self._locations(receiver_locations,h,w,v.device)
        src = src[:,None] if src.ndim == 2 else src
        if src.ndim != 3 or src.shape[0] < 1 or src.shape[1] < 1:
            raise ValueError("Source shape must be [S,Nsrc,2]")
        shots, ns = src.shape[:2]
        rec = rec[None].expand(shots,-1,-1) if rec.ndim == 2 else rec
        if rec.ndim != 3 or rec.shape[0] != shots or rec.shape[1] < 1:
            raise ValueError("Receiver shot dimension mismatch")
        wavelet = source_wavelet.to(v)
        if wavelet.ndim == 1:
            wavelet = wavelet[None,None].expand(shots,ns,-1)
        elif wavelet.ndim == 2:
            wavelet = wavelet[:,None]
        if wavelet.ndim != 3 or wavelet.shape[:2] != (shots,ns) or not torch.isfinite(wavelet).all():
            raise ValueError("Wavelet shape must agree with source shots/count")
        p = self.pml_width
        hh,ww = h+2*p,w+2*p
        padded = F.pad(v, (p,p,p,p), mode="replicate")
        coefficient = (padded[:,0,None]*self.dt/self.dx).square()
        # Interior coordinates have zero damping; physical sources need no offset config.
        yy = torch.arange(hh, device=v.device, dtype=v.dtype)
        xx = torch.arange(ww, device=v.device, dtype=v.dtype)
        dy = torch.maximum((p-yy).clamp_min(0), (yy-(p+h-1)).clamp_min(0))/max(1,p)
        dx = torch.maximum((p-xx).clamp_min(0), (xx-(p+w-1)).clamp_min(0))/max(1,p)
        mask = torch.exp(-self.damping*(dy[:,None].square()+dx[None,:].square()))
        source_index = (src[...,0]+p)*ww+src[...,1]+p
        receiver_index = (rec[...,0]+p)*ww+rec[...,1]+p
        source_map = F.one_hot(source_index, num_classes=hh*ww).to(v)
        u = v.new_zeros((batch,shots,hh,ww))
        previous = torch.zeros_like(u)
        receiver_index = receiver_index[None].expand(batch,-1,-1)
        self.counters.calls += 1
        self.counters.model_evaluations += batch
        self.counters.shot_evaluations += batch*shots
        traces = []
        for t in range(wavelet.shape[-1]):
            padded_u = F.pad(u, (1,1,1,1))
            laplacian = (padded_u[...,2:,1:-1] + padded_u[...,:-2,1:-1]
                         + padded_u[...,1:-1,2:] + padded_u[...,1:-1,:-2] - 4*u)
            source = (source_map*wavelet[...,t,None]).sum(1).reshape(1,shots,hh,ww)
            following = (2*u-previous+coefficient*laplacian+source)*mask
            previous,u = u*mask,following
            traces.append(u.flatten(-2).gather(-1,receiver_index))
        result = torch.stack(traces, -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite acoustic wavefield")
        return result
