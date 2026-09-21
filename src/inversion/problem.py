"""Shared acquisition/observation construction; truth never enters the optimizer."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from src.data.synthetic_generator import synthetic_velocity_models
from src.physics.source import ricker
from src.physics.receiver import surface_geometry
from src.physics.wave_solver import AcousticSolver
from src.utils.config import ROOT,storage_path


def input_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return ROOT/path if (ROOT/path).exists() else storage_path(path)


def make_solver(config: dict) -> AcousticSolver:
    return AcousticSolver(**{key:config[key] for key in ["dx","dt","pml_width","damping"] if key in config})


@dataclass
class Problem:
    """Observations plus explicit geometry, wavelet, initial model and optional truth."""
    observed: torch.Tensor
    sources: torch.Tensor
    receivers: torch.Tensor
    wavelet: torch.Tensor
    initial_velocity: torch.Tensor
    truth: torch.Tensor | None
    observation_pde_evaluations: int = 0

    def state_dict(self) -> dict:
        return {key: value.detach().cpu() if isinstance(value,torch.Tensor) else value for key,value in vars(self).items()}

    @classmethod
    def from_state_dict(cls,state: dict,device) -> Problem:
        return cls(**{key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in state.items()})


def make_problem(config: dict,initial_velocity: torch.Tensor,device) -> Problem:
    shape = tuple(config["shape"])
    pc = config["physics"]
    sources,receivers = surface_geometry(shape,pc["n_sources"],pc["n_receivers"],pc.get("acquisition_depth",2),device)
    wavelet = ricker(pc["source_frequency"],pc["nt"],pc["dt"],pc.get("source_delay"),pc.get("source_amplitude",1.),device)
    if config.get("truth_path"):
        truth = torch.from_numpy(np.load(input_path(config["truth_path"]),allow_pickle=False)).to(device,dtype=torch.float32)
        if "truth_index" in config:
            if truth.ndim not in (3,4):
                raise ValueError("truth_index requires a model stack")
            truth = truth[int(config["truth_index"])]
        if truth.numel() != shape[0]*shape[1]:
            raise ValueError("truth_path must hold exactly one model matching shape")
        truth = truth.reshape(1,1,*shape)
    else:
        truth = None if config.get("observed_path") else synthetic_velocity_models(1,shape,seed=config["seed"]+10000).to(device)
    observation_count = 0
    if config.get("observed_path"):
        observed = torch.from_numpy(np.load(input_path(config["observed_path"]),allow_pickle=False)).to(device,dtype=torch.float32)
        expected = (1,len(sources),len(receivers),len(wavelet))
        if observed.shape != expected:
            raise ValueError(f"Observed shape must be {expected}, got {tuple(observed.shape)}")
    else:
        with torch.no_grad():
            observed = make_solver(pc)(truth,sources,receivers,wavelet)
        observation_count = 1
    return Problem(observed,sources,receivers,wavelet,initial_velocity.detach(),truth,observation_count)
