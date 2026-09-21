from types import SimpleNamespace
import pytest
import torch
from src.ga.genetic_optimizer import GeneticOptimizer
from src.ga.selection import select


@pytest.mark.parametrize("selection", ["rank", "tournament"])
@pytest.mark.parametrize("crossover", ["arithmetic", "blend"])
def test_toy_sphere_converges_with_elitism(selection, crossover):
    torch.manual_seed(4)
    population = torch.randn(32, 6) * 2
    initial = population.square().sum(1).min().item()
    optimizer = GeneticOptimizer({"selection": selection, "crossover": crossover})
    previous = initial
    for generation in range(60):
        env = SimpleNamespace(population_target_size=32, elite_ratio=0.15,
                              crossover_probability=0.8, mutation_probability=0.3,
                              mutation_sigma=0.2*(1-generation/60)+0.01)
        population = optimizer.step(population, population.square().sum(1), env)
        current = population.square().sum(1).min().item()
        assert current <= previous + 1e-6
        previous = current
    assert previous < initial * 0.1


def test_rejected_genes_never_selected():
    fitness = torch.tensor([float("inf"), 1., float("nan")])
    for method in ["rank", "tournament"]:
        assert (select(fitness, 100, method) == 1).all()
    with pytest.raises(ValueError, match="No feasible"):
        select(torch.full((4,), float("inf")), 2)
