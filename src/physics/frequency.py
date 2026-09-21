"""Differentiable spectral bandpasses: hard FFT, smooth cosine and Butterworth."""
import math
import torch


def bandpass_mask(frequencies: torch.Tensor, low: float, high: float, method: str = "smooth",
                  transition: float = 2., order: int = 4) -> torch.Tensor:
    if not 0 <= low < high or high > frequencies[-1].item() or transition <= 0 or order < 1:
        raise ValueError("Invalid band limits/transition/order")
    if method == "fft":
        return ((frequencies >= low) & (frequencies <= high)).to(frequencies)
    if method in {"smooth", "tukey"}:
        rise = 0.5-0.5*torch.cos(math.pi*((frequencies-(low-transition))/transition).clamp(0,1)) if low > 0 else torch.ones_like(frequencies)
        fall = 0.5+0.5*torch.cos(math.pi*((frequencies-high)/transition).clamp(0,1))
        return rise*fall
    if method == "butterworth":
        lowpass = (1+(frequencies/high).pow(2*order)).rsqrt()
        highpass = (1+(low/frequencies.clamp_min(1e-12)).pow(2*order)).rsqrt() if low > 0 else torch.ones_like(frequencies)
        return lowpass*highpass
    raise ValueError(f"Unknown filter: {method}")


def spectral_masks(nt: int, dt: float, bands: list, device, dtype, method: str = "smooth",
                   transition: float = 2., center_frequency: float | None = None) -> tuple[torch.Tensor,int]:
    """Pad to at least twice the trace length; padding is not added information."""
    nfft = 1 << (2*nt-1).bit_length()
    frequencies = torch.fft.rfftfreq(nfft,dt,device=device,dtype=dtype)
    masks = torch.stack([bandpass_mask(frequencies,*band,method,transition) for band in bands])
    if center_frequency is not None:
        # Same smooth low-pass for observed/predicted data; the wavelet is unchanged.
        cutoff = min(2*center_frequency, frequencies[-1].item())
        masks = masks * bandpass_mask(frequencies,0.,cutoff,"smooth",transition)
    return masks,nfft
