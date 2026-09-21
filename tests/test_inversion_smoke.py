import copy
import torch
import pytest
from src.models.autoencoder import AutoEncoder,save_checkpoint
from src.inversion.engine import InversionEngine
from src.utils.config import load_config
from src.zbank.initializer import make_bank
from src.environment.scheduler import EnvironmentScheduler


def tiny_config(tmp_path,monkeypatch):
    monkeypatch.setenv("FWI_HOME",str(tmp_path))
    torch.manual_seed(1)
    ae = AutoEncoder(16,4,output_shape=(12,12))
    training = load_config("configs/ae.yaml")
    training["data"].update(samples=12,validation_samples=4,shape=[12,12])
    training["model"].update(latent_dim=16,channels=4)
    save_checkpoint(tmp_path/"checkpoints/autoencoder.pt",ae,training)
    config = load_config("configs/smoke.yaml")
    config.update(shape=[12,12],generations=3,output="full")
    config["physics"].update(nt=35,n_sources=1,n_receivers=3,source_delay=.012)
    config["environment_overrides"].update(population_target_size=[4,3],qc_threshold=[100.,100.],diversity_threshold=[100.,100.])
    config["qc"]["threshold"] = 100.
    config["stopping"].update(target=-1,patience=100,failed_injections=100)
    return config


def test_end_to_end_and_exact_boundary_resume(tmp_path,monkeypatch):
    config = tiny_config(tmp_path,monkeypatch)
    continuous = InversionEngine(config)
    summary = continuous.run(plots=False)
    partial_config = copy.deepcopy(config)
    partial_config["output"] = "resumed"
    partial = InversionEngine(partial_config)
    partial.run(max_additional_generations=1,plots=False)
    restored = InversionEngine(partial_config,tmp_path/"resumed/inversion.pt")
    restored_summary = restored.run(plots=False)
    assert summary["stop_reason"] == "max_generations"
    assert summary["pde_evaluations"] == restored_summary["pde_evaluations"]
    assert summary["pde_evaluations"] > 3*3
    assert torch.equal(continuous.genes,restored.genes)
    assert torch.equal(continuous.best_z,restored.best_z)
    assert summary["best_reference_fitness"] <= summary["initial_reference_fitness"]
    assert summary["injection_count"] > 0
    assert (tmp_path/"resumed/recovered_velocity.npy").exists()
    assert all(row["refined_count"] < row["population_size"] for row in restored.history)


def test_resume_rejects_schedule_change(tmp_path,monkeypatch):
    config = tiny_config(tmp_path,monkeypatch)
    engine = InversionEngine(config)
    engine.run(max_additional_generations=1,plots=False)
    changed = copy.deepcopy(config)
    changed["generations"] = 4
    with pytest.raises(ValueError,match="Resume config differs"):
        InversionEngine(changed,tmp_path/"full/inversion.pt")


def test_stale_bank_is_rejected(tmp_path,monkeypatch):
    config = tiny_config(tmp_path,monkeypatch)
    bank = make_bank(torch.zeros(3,16),EnvironmentScheduler(load_config("configs/environment.yaml"),3)(0))
    bank.save(tmp_path/"checkpoints/zbank.pt")
    with pytest.raises(ValueError,match="different/unknown AE weights"):
        InversionEngine(config)
