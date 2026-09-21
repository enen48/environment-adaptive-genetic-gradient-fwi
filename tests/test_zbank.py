import torch
from src.ga.diversity import diversity
from src.zbank.entry import BankEntry
from src.zbank.bank import ZBank
from src.zbank.injection import inject


def test_memory_bounded_save_query_and_injection(tmp_path):
    bank = ZBank(8)
    for i in range(20):
        bank.add(BankEntry(torch.ones(4)*i/10, [0.8,0.2,0.0], float(i),reference_fitness=float(i)))
    assert len(bank) == 8
    bank.save(tmp_path / "bank.pt")
    restored = ZBank.load(tmp_path / "bank.pt")
    assert len(restored) == 8
    assert len(restored.nearest_environment([0.8,0.2,0.0], 2)) == 2
    assert restored.top_k(1)[0].fitness == 0
    population = torch.zeros(6, 4)
    result, indices = inject(population, restored, [0.8,0.2,0.0], 100, max_count=2)
    assert result.shape == population.shape and len(indices) == 2
    assert torch.equal(result[0], population[0])
    assert diversity(result) > diversity(population)
    assert not torch.equal(result[-1], result[-2])


def test_diversity_definitions_and_dimension_normalization():
    population = torch.tensor([[0.,0.],[1.,1.]])
    assert abs(diversity(population) - 1) < 1e-6
    assert abs(diversity(population, "covariance") - 0.5) < 1e-6
    assert abs(diversity(population, "centroid") - 0.5) < 1e-6


def test_query_penalizes_environment_distance():
    bank = ZBank(4)
    bank.add(BankEntry(torch.ones(4), [1.,0.,0.], 1.,reference_fitness=1.))
    bank.add(BankEntry(-torch.ones(4), [0.,0.,1.], 1., frequency_weights=[0.,0.,1.],reference_fitness=1.))
    assert torch.equal(bank.query([1.,0.,0.], torch.zeros(4,4), 1)[0].z, torch.ones(4))


def test_reference_quality_and_unknown_offline_seeds():
    bank = ZBank(4)
    bank.add(BankEntry(torch.ones(4),[1.,0.,0.],.001,reference_fitness=10.))
    bank.add(BankEntry(-torch.ones(4),[0.,0.,1.],10.,frequency_weights=[0.,0.,1.],reference_fitness=.001))
    assert bank.top_k(1)[0].reference_fitness == .001
    seeds = ZBank(4)
    seeds.add(BankEntry(torch.ones(4),[1.,0.,0.],float("inf")))
    assert not seeds.query([1.,0.,0.],torch.zeros(4,4),1)
    assert len(seeds.query([1.,0.,0.],torch.zeros(4,4),1,allow_unknown=True)) == 1
