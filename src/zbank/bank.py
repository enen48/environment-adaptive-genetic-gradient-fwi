"""Memory query, environment matching, diversity-aware selection and pruning."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from .entry import BankEntry


def normalized(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    output = torch.ones_like(values)
    if finite.any():
        minimum, maximum = values[finite].min(), values[finite].max()
        output[finite] = (values[finite]-minimum)/(maximum-minimum).clamp_min(1e-8)
    return output


class ZBank:
    """Bounded memory: normalized quality plus novelty minus environment distance."""
    def __init__(self, capacity: int = 128) -> None:
        if capacity < 2:
            raise ValueError("Bank capacity must be >=2")
        self.capacity = capacity
        self.model_signature: str | None = None
        self.entries: list[BankEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: BankEntry) -> None:
        if self.entries:
            if entry.z.shape != self.entries[0].z.shape or len(entry.environment) != len(self.entries[0].environment):
                raise ValueError("Bank latent/environment dimensions differ")
        # Keep distinct environments, replace a duplicate gene only within the same environment.
        for i, previous in enumerate(self.entries):
            if torch.allclose(previous.z, entry.z, atol=1e-6) and previous.environment == entry.environment:
                if entry.fitness < previous.fitness:
                    self.entries[i] = entry
                return
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.prune()

    def quality(self) -> torch.Tensor:
        """Only fixed-reference scores are comparable; offline quality is unknown."""
        values = torch.tensor([e.reference_fitness if e.reference_fitness is not None else float("inf") for e in self.entries],dtype=torch.float64)
        return normalized(values)

    def top_k(self, k: int) -> list[BankEntry]:
        if not self.entries:
            return []
        return [self.entries[i] for i in self.quality().argsort()[:max(0, k)].tolist()]

    def nearest_environment(self, environment: list[float], k: int) -> list[BankEntry]:
        if not self.entries:
            return []
        vectors = torch.tensor([e.environment for e in self.entries])
        distance = (vectors-torch.tensor(environment)).norm(dim=1)
        return [self.entries[i] for i in distance.argsort()[:max(0, k)].tolist()]

    def query(self, environment: list[float], population: torch.Tensor, k: int,
              alpha: float = 1., beta: float = 1., gamma: float = 0.5,
              allow_unknown: bool = False) -> list[BankEntry]:
        """score = -alpha*quality + beta*novelty - gamma*environment_distance.

        Greedy novelty is recomputed after every pick to avoid injecting clones.
        """
        if not self.entries or k <= 0:
            return []
        genes = torch.stack([e.z for e in self.entries])
        current = population.detach().cpu().to(genes)
        envs = torch.tensor([e.environment for e in self.entries])
        environment_distance = (envs-torch.tensor(environment)).norm(dim=1) / math.sqrt(envs.shape[1])
        quality = self.quality().to(genes)
        known = torch.tensor([e.reference_fitness is not None and math.isfinite(e.reference_fitness) for e in self.entries])
        selected: list[int] = []
        for _ in range(min(k, len(genes))):
            distance = torch.cdist(genes, current).min(1).values / math.sqrt(genes.shape[1])
            score = -alpha*quality + beta*normalized(distance) - gamma*environment_distance
            if not allow_unknown:
                score[~known] = -torch.inf
            # Existing clones carry no useful diversity injection.
            score[distance < 1e-6] = -torch.inf
            if selected:
                score[selected] = -torch.inf
            index = int(score.argmax())
            if not torch.isfinite(score[index]):
                break
            selected.append(index)
            current = torch.cat([current, genes[index:index+1]])
        return [self.entries[i] for i in selected]

    def prune(self) -> None:
        """Retain quality elites plus farthest-point latent/environment coverage."""
        if len(self.entries) <= self.capacity:
            return
        genes = torch.stack([e.z for e in self.entries])
        envs = torch.tensor([e.environment for e in self.entries])
        features = torch.cat([genes/math.sqrt(genes.shape[1]), envs/math.sqrt(envs.shape[1])], dim=1)
        chosen = self.quality().argsort()[:self.capacity//2].tolist()
        while len(chosen) < self.capacity:
            distances = torch.cdist(features, features[chosen]).min(1).values
            distances[chosen] = -1
            chosen.append(int(distances.argmax()))
        self.entries = [self.entries[i] for i in chosen]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "entries": [e.state_dict() for e in self.entries],"model_signature":self.model_signature}

    @classmethod
    def from_state_dict(cls, state: dict) -> ZBank:
        bank = cls(state["capacity"])
        bank.model_signature = state.get("model_signature")
        bank.entries = [BankEntry(**e) for e in state["entries"]]
        bank.prune()
        return bank

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> ZBank:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))
