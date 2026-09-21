"""One bounded latent generation with exact elitism."""
import math
import torch
from .selection import select
from .crossover import crossover
from .mutation import mutate


class GeneticOptimizer:
    """Breed children without any PDE work or velocity-space mutation."""
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def step(self, genes: torch.Tensor, fitness: torch.Tensor, environment) -> torch.Tensor:
        target = int(environment.population_target_size)
        if target < 2:
            raise ValueError("Population target must be >=2")
        finite = torch.where(torch.isfinite(fitness))[0]
        if not len(finite):
            raise ValueError("All candidates failed; recalibrate QC or prior")
        elite_count = min(len(finite), target, max(1, math.ceil(target * environment.elite_ratio)))
        elites = genes[finite[fitness[finite].argsort()[:elite_count]]]
        count = target - elite_count
        if not count:
            return elites.detach().clone()
        parents = select(fitness, 2*count, self.config.get("selection", "tournament"), self.config.get("tournament_size", 3))
        children = crossover(genes[parents[:count]], genes[parents[count:]], environment.crossover_probability,
                             self.config.get("crossover", "blend"), self.config.get("blend_alpha", 0.2))
        children = mutate(children, environment.mutation_probability, environment.mutation_sigma, self.config.get("latent_clip", 6.0))
        return torch.cat([elites, children]).detach()
