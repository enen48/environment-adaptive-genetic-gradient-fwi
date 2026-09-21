import pytest
import torch
from src.physics.wave_solver import AcousticSolver
from src.physics.source import ricker


def test_batch_shots_dtype_and_counters():
    solver = AcousticSolver(pml_width=3)
    v = torch.full((2,1,12,14), 2200., dtype=torch.float64)
    src = torch.tensor([[2,3],[2,10]])
    rec = torch.tensor([[2,4],[3,7],[2,9]])
    traces = solver(v, src, rec, ricker(15,40,.001,delay=.015,dtype=v.dtype))
    assert traces.shape == (2,2,3,40) and traces.dtype == v.dtype
    assert traces.abs().sum() > 0 and torch.equal(traces[0], traces[1])
    assert solver.counter_state() == {"calls":1,"model_evaluations":2,"shot_evaluations":4}
    with pytest.raises(ValueError, match="CFL"):
        solver(v*4,src,rec,torch.ones(20))


def test_acoustic_directional_derivative_matches_finite_difference():
    torch.set_num_threads(2)
    solver = AcousticSolver(pml_width=3)
    v = torch.full((1,1,10,12), 2300.,dtype=torch.float64,requires_grad=True)
    src,rec = torch.tensor([[2,5]]),torch.tensor([[2,7],[4,5]])
    wavelet = ricker(20,45,.001,delay=.012,dtype=v.dtype)
    loss = solver(v,src,rec,wavelet).square().mean()
    gradient, = torch.autograd.grad(loss,v)
    torch.manual_seed(2)
    direction = torch.randn_like(v)
    step = .1
    with torch.no_grad():
        plus = solver(v+step*direction,src,rec,wavelet).square().mean()
        minus = solver(v-step*direction,src,rec,wavelet).square().mean()
    finite_difference = (plus-minus)/(2*step)
    assert torch.allclose((gradient*direction).sum(), finite_difference, rtol=1e-4,atol=1e-8)
    assert gradient.abs().sum() > 0


def test_wave_propagation_speed_and_zero_source():
    solver = AcousticSolver(pml_width=8)
    src,rec = torch.tensor([[12,8]]),torch.tensor([[12,14]])
    wavelet = ricker(30,100,.001,delay=.025)
    slow = solver(torch.full((25,25),1800.),src,rec,wavelet).abs().flatten()
    fast = solver(torch.full((25,25),3000.),src,rec,wavelet).abs().flatten()
    slow_first = torch.where(slow > slow.max()*.2)[0][0]
    fast_first = torch.where(fast > fast.max()*.2)[0][0]
    assert fast_first < slow_first
    zero = solver(torch.full((25,25),2200.),src,rec,torch.zeros(30))
    assert torch.count_nonzero(zero) == 0
