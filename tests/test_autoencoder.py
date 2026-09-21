"""Verify variable geometry, latent derivatives, real learning and resuming."""
import copy

import pytest
import torch

from scripts.train_ae import train
from src.data.synthetic_generator import synthetic_velocity_models
from src.models.autoencoder import AutoEncoder, load_autoencoder, save_checkpoint


@pytest.mark.parametrize("latent_dim", [16, 32, 64])
def test_variable_geometry_and_frozen_latent_derivative(latent_dim):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = AutoEncoder(latent_dim=latent_dim, channels=4, output_shape=(19, 27))
    velocities = synthetic_velocity_models(2, (19, 27))
    z = model.encode(velocities)
    assert z.shape == (2, latent_dim)
    reconstructed = model(velocities)
    assert reconstructed.shape == velocities.shape
    assert 1500 <= reconstructed.min() <= reconstructed.max() <= 4500
    assert model.decode(z, (35, 41)).shape == (2, 1, 35, 41)
    model.freeze()
    latent = z.detach().requires_grad_(True)
    model.decode(latent).mean().backward()
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0


def test_checkpoint_round_trip(tmp_path):
    model = AutoEncoder(latent_dim=16, channels=4, output_shape=(16, 20))
    config = {"description": "round-trip"}
    z = torch.randn(2, 16)
    file = tmp_path / "ae.pt"
    save_checkpoint(file, model, config)
    loaded, restored_config = load_autoencoder(file)
    assert restored_config == config
    assert torch.equal(model.decode(z), loaded.decode(z))


def test_training_learns_and_resumes_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("FWI_HOME", str(tmp_path))
    config = {"seed": 13, "device": "cpu", "threads": 2,
              "data": {"path": None, "samples": 12, "validation_samples": 4,
                       "shape": [16, 16], "velocity_min": 1500., "velocity_max": 4500.},
              "model": {"latent_dim": 16, "channels": 4},
              "training": {"epochs": 4, "batch_size": 4, "learning_rate": 0.005},
              "output": "continuous"}
    uninterrupted = train(config)
    first_stage = copy.deepcopy(config)
    first_stage["output"], first_stage["training"]["epochs"] = "resumed", 2
    partial = train(first_stage)
    second_stage = copy.deepcopy(config)
    second_stage["output"] = "resumed"
    resumed = train(second_stage, partial["checkpoint"])
    first = torch.load(uninterrupted["checkpoint"], weights_only=True)
    second = torch.load(resumed["checkpoint"], weights_only=True)
    assert second["epoch"] == 4
    assert second["history"][-1]["train_mse"] < second["history"][0]["train_mse"]
    assert all(torch.equal(first["model_state"][key], second["model_state"][key]) for key in first["model_state"])
    assert (tmp_path / "resumed" / "encoder.pt").is_file()
    assert (tmp_path / "resumed" / "decoder.pt").is_file()
