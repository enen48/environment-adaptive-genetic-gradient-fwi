"""Velocity-model datasets and reproducible synthetic geology."""
from .dataset import VelocityDataset
from .synthetic_generator import synthetic_velocity_models

__all__ = ["VelocityDataset", "synthetic_velocity_models"]
