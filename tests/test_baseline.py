import copy
import numpy as np
from src.fwi.baseline import run_baseline
from src.inversion.ablation import variant_config
from test_inversion_smoke import tiny_config


def test_baseline_runs_and_resumes(tmp_path,monkeypatch):
    config = tiny_config(tmp_path,monkeypatch)
    full = run_baseline(config,"baseline_full",plots=False)
    run_baseline(config,"baseline_partial",plots=False,max_additional_generations=1)
    resumed = run_baseline(config,"baseline_partial",plots=False,resume=tmp_path/"baseline_partial/baseline.pt")
    assert full["pde_evaluations"] == resumed["pde_evaluations"]
    assert np.array_equal(np.load(tmp_path/"baseline_full/recovered_velocity.npy"),np.load(tmp_path/"baseline_partial/recovered_velocity.npy"))
    assert full["waveform_misfit"] <= full["initial_reference_fitness"]


def test_ablations_are_independent_and_do_not_mutate_config():
    config = {"local_fwi":{"enabled":True},"zbank":{"enabled":True},"qc":{"enabled":True},"dynamic_environment":True}
    original = copy.deepcopy(config)
    for name,key in [("no_local","local_fwi"),("no_bank","zbank"),("no_qc","qc")]:
        variant = variant_config(config,name)
        assert variant[key]["enabled"] is False
    assert variant_config(config,"no_dynamic")["dynamic_environment"] is False
    assert config == original
