"""Verify data reproducibility, split separation and physical-unit preservation."""
import numpy as np
import pytest
import torch

from src.data.dataset import VelocityDataset
from src.data.synthetic_generator import synthetic_velocity_models
from src.utils.config import ROOT


def test_synthetic_seed_and_bounds():
    first = synthetic_velocity_models(5, (17, 23), seed=3)
    assert first.shape == (5, 1, 17, 23)
    assert torch.equal(first, synthetic_velocity_models(5, (17, 23), seed=3))
    assert not torch.equal(first, synthetic_velocity_models(5, (17, 23), seed=4))
    assert torch.isfinite(first).all()
    assert first.min() >= 1500 and first.max() <= 4500


def test_mmap_preserves_values_and_does_not_mutate_source(tmp_path):
    array = synthetic_velocity_models(3, (16, 20)).numpy()
    np.save(tmp_path / "models.npy", array)
    dataset = VelocityDataset(tmp_path / "models.npy")
    assert isinstance(dataset.arrays[0], np.memmap)
    assert np.array_equal(dataset[0].numpy(), array[0])
    sample = dataset[0]
    sample.zero_()
    assert np.array_equal(dataset[0].numpy(), array[0])
    assert VelocityDataset(tmp_path / "models.npy", shape=(11, 13))[0].shape == (1, 11, 13)
    with pytest.raises(ValueError, match="above"):
        VelocityDataset(tmp_path / "models.npy", vmax=1600)[0]


def test_shard_indexing_and_gather_rejection(tmp_path):
    for index, size in enumerate([2, 3]):
        np.save(tmp_path / f"model{index}.npy", np.full((size, 12, 16), 2000. + 100 * index, np.float32))
    dataset = VelocityDataset(tmp_path)
    assert len(dataset) == 5 and dataset[1].mean() == 2000 and dataset[2].mean() == 2100
    assert torch.equal(dataset[-1], dataset[4])
    with pytest.raises(IndexError):
        dataset[5]
    np.save(tmp_path / "gathers.npy", np.zeros((2, 5, 100, 16), np.float32))
    with pytest.raises(ValueError, match="velocity"):
        VelocityDataset(tmp_path / "gathers.npy")


def test_bundled_openfwi_train_and_validation_are_distinct():
    train_set = VelocityDataset(ROOT / "data/sample/openfwi_flatvel_a_train.npy")
    validation = VelocityDataset(ROOT / "data/sample/openfwi_flatvel_a_val.npy")
    assert len(train_set) == 64 and len(validation) == 8
    assert train_set[0].shape == (1, 70, 70)
    train_hashes = {sample.numpy().tobytes() for sample in train_set}
    assert not any(sample.numpy().tobytes() in train_hashes for sample in validation)
