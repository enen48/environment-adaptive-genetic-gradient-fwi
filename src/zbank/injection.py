"""Replace a bounded number of non-elite genes with selected memory genes."""
import torch
from .bank import ZBank


def inject(genes: torch.Tensor, bank: ZBank, environment: list[float], count: int,
           max_count: int = 3, protected: int = 1, **weights) -> tuple[torch.Tensor, list[int]]:
    budget = min(max(0, count), max(0, max_count), max(0, len(genes)-protected))
    selected = bank.query(environment, genes, budget, **weights)
    result = genes.clone()
    indices = list(range(len(genes)-len(selected), len(genes)))
    for index, entry in zip(indices, selected):
        result[index] = entry.z.to(result)
    return result, indices
