"""Noninteractive figures for reproducible experiment reports."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_results(directory: Path, initial: torch.Tensor, recovered: torch.Tensor,
                 truth: torch.Tensor | None, history: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    panels = [("Initial (m/s)", initial), ("Recovered (m/s)", recovered)]
    if truth is not None:
        panels = [("True (m/s)", truth)] + panels + [("Absolute error (m/s)", (truth-recovered).abs())]
    fig, axes = plt.subplots(1, len(panels), figsize=(4*len(panels), 4), constrained_layout=True)
    for ax, (title, model) in zip(np.atleast_1d(axes), panels):
        kwargs = {} if "error" in title else {"vmin": 1500, "vmax": 4500}
        im = ax.imshow(model.detach().cpu().squeeze(), aspect="auto", cmap="viridis", **kwargs)
        ax.set(title=title, xlabel="Horizontal grid cell", ylabel="Depth grid cell")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(directory / "velocity_models.png", dpi=140)
    plt.close(fig)
    metrics = [("best_fitness", "Environment-dependent fitness"), ("waveform_loss", "Waveform misfit"),
               ("reference_loss", "Fixed final-environment loss"), ("diversity", "Latent diversity"),
               ("mutation_sigma", "Mutation sigma"), ("population_size", "Population size"),
               ("bank_size", "Z-bank size"), ("pde_evaluations", "Cumulative model PDE evaluations"),
               ("local_fwi_steps", "Local steps")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot([r["generation"] for r in history], [r.get(key, float("nan")) for r in history], marker=".")
        ax.set(title=title, xlabel="Generation")
        ax.grid(alpha=0.2)
    fig.savefig(directory / "curves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    for i, name in enumerate(["low", "mid", "high"]):
        ax.plot([r["generation"] for r in history], [r["frequency_weights"][i] for r in history], label=name)
    ax.set(xlabel="Generation", ylabel="Frequency weight", ylim=(0, 1))
    ax.legend()
    fig.savefig(directory / "frequency_weights.png", dpi=140)
    plt.close(fig)
