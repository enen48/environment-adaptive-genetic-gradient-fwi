import pytest
import torch
from src.physics.misfit import MultiFrequencyMisfit


@pytest.mark.parametrize("method", ["fft","smooth","tukey","butterworth"])
def test_multiband_identity_filtering_and_gradient(method):
    t = torch.arange(1000,dtype=torch.float64)*.001
    low = torch.sin(2*torch.pi*5*t)
    high = torch.sin(2*torch.pi*50*t)
    observed = (low+high).reshape(1,1,1,-1)
    misfit = MultiFrequencyMisfit(observed,.001,[[1.,10.],[20.,35.],[40.,70.]],method)
    assert misfit(observed,[.8,.2,0.]).item() == 0
    predicted = low.reshape_as(observed).clone().requires_grad_()
    parts = misfit.components(predicted)
    assert parts[0,2] > parts[0,0]*10
    loss = misfit(predicted,[0.,0.,1.]).sum()
    loss.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0


def test_empty_band_is_finite_and_center_filter_shared():
    observed = torch.zeros(1,1,2,31)
    misfit = MultiFrequencyMisfit(observed,.001,[[1.,8.],[8.,16.],[16.,30.]])
    assert torch.isfinite(misfit(torch.zeros_like(observed),[.8,.2,0.],4.)).all()
