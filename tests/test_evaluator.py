import torch
from src.models.autoencoder import AutoEncoder
from src.physics.wave_solver import AcousticSolver
from src.physics.source import ricker
from src.physics.misfit import MultiFrequencyMisfit
from src.environment.scheduler import EnvironmentScheduler
from src.utils.config import load_config
from src.inversion.evaluator import Evaluator


def make_evaluator(threshold=100., dtype=torch.float32):
    torch.manual_seed(10)
    ae = AutoEncoder(16,4,output_shape=(12,12)).to(dtype=dtype).freeze()
    solver = AcousticSolver(pml_width=3)
    src,rec = torch.tensor([[2,6]]),torch.tensor([[2,4],[2,8]])
    wavelet = ricker(15,45,.001,delay=.015,dtype=dtype)
    obs = solver(torch.full((12,12),2200.,dtype=dtype),src,rec,wavelet).detach()
    solver.reset_counters()
    loss = MultiFrequencyMisfit(obs,.001,[[0.5,6.],[5.,11.],[10.,20.]])
    evaluator = Evaluator(ae,solver,loss,src,rec,wavelet,{"threshold":threshold},(12,12))
    env = EnvironmentScheduler(load_config("configs/environment.yaml"),3)(0)
    return evaluator,env


def test_cheap_qc_rejects_without_pde_and_bands_share_one_solve():
    evaluator,env = make_evaluator()
    accepted = evaluator(torch.zeros(16),env)
    assert accepted.accepted and evaluator.solver.counters.model_evaluations == 1
    rejected = evaluator(torch.full((16,),100.),env)
    assert not rejected.accepted and evaluator.solver.counters.model_evaluations == 1
    assert evaluator.rejected == 1


def test_full_decoder_physics_chain_has_latent_gradient():
    evaluator,env = make_evaluator()
    z = torch.zeros(16,requires_grad=True)
    evaluation = evaluator(z,env)
    evaluation.total.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
