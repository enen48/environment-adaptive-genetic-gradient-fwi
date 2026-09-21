"""Frequency components from one shared acoustic trace per candidate."""
import torch
from .frequency import spectral_masks


class MultiFrequencyMisfit:
    """J=sum_b w_b mean((F_b(d)-F_b(dobs))²)/mean(F_b(dobs)²).

    An energy floor avoids division by an empty observed band. No solver is
    called here; observed FFTs are cached. Inputs use [B,S,R,T].
    """
    def __init__(self, observed: torch.Tensor, dt: float, bands: list,
                 filter: str = "smooth", transition: float = 2., energy_floor: float = 1e-6) -> None:
        if observed.ndim != 4 or observed.shape[0] != 1 or not torch.isfinite(observed).all():
            raise ValueError("Observed must be finite [1,S,R,T]")
        if energy_floor <= 0 or dt <= 0:
            raise ValueError("Invalid misfit parameters")
        self.observed = observed.detach()
        self.dt,self.bands,self.filter,self.transition,self.energy_floor = dt,bands,filter,transition,energy_floor
        _,nfft = spectral_masks(observed.shape[-1],dt,bands,observed.device,observed.dtype,filter,transition)
        self.observed_spectrum = torch.fft.rfft(self.observed,n=nfft)

    def components(self, predicted: torch.Tensor, center_frequency: float | None = None) -> torch.Tensor:
        if predicted.shape[1:] != self.observed.shape[1:]:
            raise ValueError("Predicted and observed sampling/acquisition must match")
        nt = predicted.shape[-1]
        masks,nfft = spectral_masks(nt,self.dt,self.bands,predicted.device,predicted.dtype,self.filter,self.transition,center_frequency)
        spectrum = torch.fft.rfft(predicted,n=nfft)
        # Broadcast as [B,band,S,R,F], then integrate over shot/receiver/time.
        filters = masks[None,:,None,None,:]
        synthetic = torch.fft.irfft(spectrum[:,None]*filters,n=nfft)[...,:nt]
        observed = torch.fft.irfft(self.observed_spectrum[:,None]*filters,n=nfft)[...,:nt]
        floor = self.observed.square().mean().clamp_min(1e-12)*self.energy_floor
        energy = observed.square().mean(dim=(-3,-2,-1)).clamp_min(floor)
        return (synthetic-observed).square().mean(dim=(-3,-2,-1))/energy

    def __call__(self, predicted: torch.Tensor, weights: list[float], center_frequency: float | None = None) -> torch.Tensor:
        components = self.components(predicted,center_frequency)
        weight = predicted.new_tensor(weights)
        if len(weight) != components.shape[1] or (weight<0).any() or weight.sum() <= 0:
            raise ValueError("Invalid frequency weights")
        return (components*weight/weight.sum()).sum(-1)
