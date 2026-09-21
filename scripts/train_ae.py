"""Train/resume the velocity AutoEncoder; outputs prefer the E-drive cache."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.loaders import data_path, make_datasets
from src.models.autoencoder import AutoEncoder, load_autoencoder, save_checkpoint
from src.utils.config import ROOT, load_config, save_config, storage_path
from src.utils.seed import setup


def reconstruction_terms(model: AutoEncoder, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Normalized reconstruction MSE, spatial roughness and latent L2."""
    z = model.encode(batch)
    reconstruction = model.decode(z, tuple(batch.shape[-2:]))
    scale = model.velocity_max - model.velocity_min
    normalized = (reconstruction - model.velocity_min) / scale
    mse = ((reconstruction - batch) / scale).square().mean()
    smoothness = (normalized[..., 1:, :] - normalized[..., :-1, :]).square().mean()
    smoothness = smoothness + (normalized[..., :, 1:] - normalized[..., :, :-1]).square().mean()
    return mse, smoothness, z.square().mean()


def train(config: dict[str, Any], resume: str | Path | None = None) -> dict[str, Any]:
    """Train to the configured total epoch count, resuming optimizer and RNGs."""
    device = setup(config.get("seed", 42), config.get("device", "auto"), config.get("threads", 2))
    training, data = config["training"], config["data"]
    epochs = int(training["epochs"])
    if epochs < 1 or int(training["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_set, validation_set = make_datasets(config)
    architecture = {**config["model"], "velocity_min": data["velocity_min"],
                    "velocity_max": data["velocity_max"], "output_shape": tuple(data["shape"])}
    model = AutoEncoder(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume:
        resume_path = data_path(str(resume))
        restored, _ = load_autoencoder(resume_path, device)
        if restored.architecture_config() != model.architecture_config():
            raise ValueError("Resume checkpoint architecture/velocity bounds differ from requested config")
        model.load_state_dict(restored.state_dict())
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        start_epoch, history = int(payload["epoch"]), payload.get("history", [])
        if epochs <= start_epoch:
            raise ValueError(f"Configured epochs={epochs} must exceed completed epoch={start_epoch} when resuming")
        generator.set_state(payload["loader_generator_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    train_loader = DataLoader(train_set, batch_size=int(training["batch_size"]), shuffle=True,
                              generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=int(training["batch_size"]), shuffle=False, num_workers=0)
    destination = storage_path(config.get("output", "checkpoints"))
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "training_config.yaml")
    for epoch in range(start_epoch, epochs):
        model.train()
        totals = {"train_loss": 0., "train_mse": 0.}
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            mse, smoothness, latent = reconstruction_terms(model, batch)
            loss = mse + float(training.get("smoothness_weight", 0.)) * smoothness + float(training.get("latent_weight", 0.)) * latent
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite AE loss")
            loss.backward()
            optimizer.step()
            totals["train_loss"] += float(loss.detach()) * len(batch)
            totals["train_mse"] += float(mse.detach()) * len(batch)
        model.eval()
        validation_mse = 0.
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                validation_mse += float(reconstruction_terms(model, batch)[0]) * len(batch)
        row = {"epoch": epoch + 1, **{key: value / len(train_set) for key, value in totals.items()},
               "val_mse": validation_mse / len(validation_set)}
        history.append(row)
        print(f"epoch={epoch + 1}/{epochs} train_mse={row['train_mse']:.6f} val_mse={row['val_mse']:.6f}", flush=True)
        save_checkpoint(destination / "autoencoder.pt", model, config, epoch=epoch + 1,
                        optimizer_state=optimizer.state_dict(), history=history,
                        loader_generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        (destination / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    # Separate component files intentionally include their physical-unit config.
    for component in ("encoder", "decoder"):
        torch.save({"format_version": 1, "component": component,
                    "model_config": model.architecture_config(),
                    "state_dict": getattr(model, component).state_dict(), "config": config},
                   destination / f"{component}.pt")
    if history:
        with (destination / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    result = {"checkpoint": str(destination / "autoencoder.pt"), "epochs": max(epochs, start_epoch),
              "train_samples": len(train_set), "validation_samples": len(validation_set),
              "device": str(device), "final_metrics": history[-1] if history else None}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ae.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to continue to total configured epochs")
    args = parser.parse_args()
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
