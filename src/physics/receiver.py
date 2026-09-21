"""Surface-style acquisition geometry in physical-grid coordinates."""
import torch


def surface_geometry(shape: tuple[int, int], n_sources: int = 2, n_receivers: int = 12,
                     depth: int = 2, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    h,w = shape
    if h < 8 or w < 8 or n_sources < 1 or not 1 <= n_receivers <= w-2 or not 0 <= depth < h:
        raise ValueError("Invalid acquisition geometry")
    sx = torch.linspace(2, w-3, n_sources, device=device).round().long() if n_sources > 1 else torch.tensor([w//2], device=device)
    rx = torch.linspace(1, w-2, n_receivers, device=device).round().long()
    return torch.stack([torch.full_like(sx, depth), sx], -1), torch.stack([torch.full_like(rx, depth), rx], -1)
