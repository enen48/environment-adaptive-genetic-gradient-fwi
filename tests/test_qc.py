from types import SimpleNamespace
import pytest
import torch
from src.models.autoencoder import AutoEncoder
from src.qc.latent_qc import LatentQC
from src.qc.geological_qc import GeologicalQC
from src.qc.quality_control import QualityControl


def test_hard_percentile_and_adaptive_thresholds():
    for mode in ["percentile","adaptive"]:
        qc = LatentQC({"mode":mode,"percentile":50.,"threshold":4.})
        qc.calibrate(torch.tensor([1.,2.,3.]))
        assert qc.limit(4.) == 2.
        qc.update(torch.tensor([1.,1.,1.]))
        assert qc.limit(4.) == pytest.approx(1.9 if mode == "adaptive" else 2.)


def test_geology_constraints_and_cycle_rejection():
    model = torch.full((1,1,12,12),2200.)
    qc = GeologicalQC({"velocity_min":1500.,"velocity_max":4500.,"max_gradient":500.})
    assert qc(model).passed
    model[...,6:,:] = 4400.
    assert not qc(model).passed and qc(model).penalty > 0
    ae = AutoEncoder(16,4,output_shape=(12,12)).freeze()
    cycle = QualityControl(ae,{"threshold":.001},(12,12))
    assert not cycle(torch.ones(16)*100,SimpleNamespace(qc_threshold=.001)).passed
