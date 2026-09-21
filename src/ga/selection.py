"""Tournament and rank selection robust to rejected candidates."""
import torch


def select(fitness: torch.Tensor, count: int, method: str = "tournament", tournament_size: int = 3) -> torch.Tensor:
    if method not in {"tournament", "rank"}:
        raise ValueError(f"Unknown selection: {method}")
    valid = torch.where(torch.isfinite(fitness))[0]
    if not len(valid):
        raise ValueError("No feasible individuals available for selection")
    if method == "tournament":
        entrants = valid[torch.randint(len(valid), (count, max(1, tournament_size)), device=fitness.device)]
        return entrants.gather(1, fitness[entrants].argmin(dim=1, keepdim=True)).squeeze(1)
    ranked = valid[fitness[valid].argsort()]
    weights = torch.arange(len(ranked), 0, -1, dtype=fitness.dtype, device=fitness.device)
    return ranked[torch.multinomial(weights, count, replacement=True)]
