"""Smooth configurable environment interpolation; only integer budgets round."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math


def smooth_progress(progress: float, method: str = "smoothstep") -> float:
    s = min(1., max(0., progress))
    if method == "linear":
        return s
    if method == "cosine":
        return 0.5*(1-math.cos(math.pi*s))
    if method == "smoothstep":
        return 3*s*s-2*s*s*s
    if method == "sigmoid":
        logistic = lambda x: 1/(1+math.exp(-12*(x-0.5)))
        return (logistic(s)-logistic(0))/(logistic(1)-logistic(0))
    raise ValueError(f"Unknown schedule: {method}")


@dataclass(frozen=True)
class Environment:
    """E(t): objective, search, feasibility and computational budgets."""
    progress: float
    frequency_weights: list[float]
    center_frequency: float
    model_smoothing_scale: float
    mutation_probability: float
    mutation_sigma: float
    crossover_probability: float
    elite_ratio: float
    qc_threshold: float
    injection_count: int
    local_fwi_steps: int
    population_target_size: int
    diversity_threshold: float
    lambda_wave: float
    lambda_cycle: float
    lambda_geo: float

    def vector(self) -> list[float]:
        """Fixed dimensionless embedding for comparable memory distances."""
        return self.frequency_weights + [self.progress, self.center_frequency/20,
            self.model_smoothing_scale/4, self.mutation_probability, self.mutation_sigma,
            self.crossover_probability, self.elite_ratio, self.qc_threshold/10,
            self.injection_count/10, self.local_fwi_steps/10, self.population_target_size/100,
            self.diversity_threshold, self.lambda_wave, self.lambda_cycle, self.lambda_geo]

    def state_dict(self) -> dict:
        return asdict(self)


class EnvironmentScheduler:
    """param(s)=start+(end-start)*g(s), with continuous frequency interpolation."""
    def __init__(self, config: dict, generations: int, dynamic: bool = True) -> None:
        self.config, self.generations, self.dynamic = config, generations, dynamic
        if generations < 1:
            raise ValueError("generations must be >=1")
        smooth_progress(0.5, config.get("schedule", "smoothstep"))
        for point in ("start", "middle", "end"):
            weights = config["frequency_weights"][point]
            if len(weights) != 3 or min(weights) < 0 or sum(weights) <= 0:
                raise ValueError("Frequency weights require three nonnegative components")

    def at_progress(self, progress: float) -> Environment:
        s = min(1., max(0., progress))
        g = smooth_progress(s, self.config.get("schedule", "smoothstep"))
        numeric = {}
        integers = {"injection_count", "local_fwi_steps", "population_target_size"}
        for key in Environment.__dataclass_fields__:
            if key in {"progress", "frequency_weights"}:
                continue
            start, end = self.config[key]
            value = float(start) + (float(end)-float(start))*g
            numeric[key] = int(round(value)) if key in integers else value
        points = self.config["frequency_weights"]
        left, right = (points["start"], points["middle"]) if s <= 0.5 else (points["middle"], points["end"])
        local = smooth_progress(2*s if s <= 0.5 else 2*s-1, self.config.get("schedule", "smoothstep"))
        weights = [a+(b-a)*local for a,b in zip(left,right)]
        weights = [value/sum(weights) for value in weights]
        environment = Environment(s, weights, **numeric)
        if environment.population_target_size < 2 or environment.local_fwi_steps < 0 or environment.injection_count < 0:
            raise ValueError("Invalid environment budgets")
        for name in ["mutation_probability", "crossover_probability", "elite_ratio"]:
            if not 0 <= getattr(environment, name) <= 1:
                raise ValueError(f"Invalid {name}")
        for name in ["mutation_sigma", "model_smoothing_scale", "qc_threshold", "diversity_threshold", "lambda_wave", "lambda_cycle", "lambda_geo"]:
            if getattr(environment, name) < 0:
                raise ValueError(f"Negative {name}")
        return environment

    def __call__(self, generation: int) -> Environment:
        # N generations have indices 0..N-1, hence the final iteration reaches s=1.
        return self.at_progress(generation/max(1, self.generations-1) if self.dynamic else 1.)
