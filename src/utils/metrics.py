"""Velocity error metrics; all arrays use the same physical units and grid."""
import torch
import torch.nn.functional as F


def velocity_metrics(predicted: torch.Tensor, truth: torch.Tensor, data_range: float = 3000.) -> dict[str, float]:
    if predicted.shape != truth.shape:
        raise ValueError("Metric grids must match; do not silently resample truth")
    error = predicted - truth
    # Local SSIM with fixed physical dynamic range; no per-image renormalization.
    x = predicted.reshape(-1, 1, *predicted.shape[-2:])
    y = truth.reshape_as(x)
    ux, uy = F.avg_pool2d(x, 7, 1), F.avg_pool2d(y, 7, 1)
    vx = (F.avg_pool2d(x*x, 7, 1) - ux*ux).clamp_min(0)
    vy = (F.avg_pool2d(y*y, 7, 1) - uy*uy).clamp_min(0)
    cov = F.avg_pool2d(x*y, 7, 1) - ux*uy
    if data_range <= 0:
        raise ValueError("SSIM data_range must be positive")
    c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    ssim = ((2*ux*uy+c1)*(2*cov+c2)/((ux*ux+uy*uy+c1)*(vx+vy+c2))).mean()
    return {"velocity_mae": error.abs().mean().item(), "velocity_rmse": error.square().mean().sqrt().item(),
            "relative_l2": (error.norm()/truth.norm().clamp_min(1e-12)).item(), "ssim": ssim.item()}
