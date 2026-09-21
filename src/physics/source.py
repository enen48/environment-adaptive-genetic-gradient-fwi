"""Source time functions."""
import math
import torch


def ricker(frequency: float, nt: int, dt: float, delay: float | None = None,
           amplitude: float = 1., device=None, dtype=torch.float32) -> torch.Tensor:
    if frequency <= 0 or nt < 2 or dt <= 0 or frequency >= 0.5/dt:
        raise ValueError("Invalid Ricker wavelet sampling")
    t = torch.arange(nt, dtype=dtype, device=device)*dt - (1.5/frequency if delay is None else delay)
    a = (math.pi*frequency*t).square()
    return amplitude*(1-2*a)*torch.exp(-a)
