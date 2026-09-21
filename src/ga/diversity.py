"""Scale-aware latent population diversity."""
import math
import torch


def diversity(genes: torch.Tensor, metric: str = "pairwise", normalize: bool = True) -> float:
    if genes.ndim != 2 or not torch.isfinite(genes).all():
        raise ValueError("Diversity expects finite [N,L] genes")
    if len(genes) < 2:
        return 0.0
    centered = genes - genes.mean(0)
    if metric == "pairwise":
        value = torch.pdist(genes).mean()
    elif metric == "centroid":
        value = centered.norm(dim=1).mean()
    elif metric == "covariance":
        value = centered.square().sum() / (len(genes)-1)
    else:
        raise ValueError(f"Unknown diversity metric: {metric}")
    if normalize:
        value = value / (genes.shape[1] if metric == "covariance" else math.sqrt(genes.shape[1]))
    return float(value)
