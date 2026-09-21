import pytest
from src.environment.scheduler import EnvironmentScheduler
from src.utils.config import load_config


@pytest.mark.parametrize("method", ["linear", "cosine", "smoothstep", "sigmoid"])
def test_continuous_schedules_and_endpoints(method):
    config = load_config("configs/environment.yaml")
    config["schedule"] = method
    schedule = EnvironmentScheduler(config, 101)
    first, final = schedule(0), schedule(100)
    assert first.frequency_weights == pytest.approx([0.8,0.2,0.])
    assert final.frequency_weights == pytest.approx([0.,0.2,0.8])
    assert schedule(50).frequency_weights == pytest.approx([0.2,0.6,0.2])
    assert first.population_target_size == 24 and final.population_target_size == 8
    assert first.mutation_sigma == pytest.approx(0.25)
    assert final.mutation_sigma == pytest.approx(0.03)
    a,b = schedule.at_progress(0.5-1e-7),schedule.at_progress(0.5+1e-7)
    assert max(abs(x-y) for x,y in zip(a.frequency_weights,b.frequency_weights)) < 1e-5
    states = [schedule(g) for g in range(101)]
    assert all(abs(sum(s.frequency_weights)-1)<1e-7 for s in states)
    assert all(a.mutation_sigma >= b.mutation_sigma for a,b in zip(states,states[1:]))
    assert all(a.local_fwi_steps <= b.local_fwi_steps for a,b in zip(states,states[1:]))


def test_static_environment_is_really_static():
    schedule = EnvironmentScheduler(load_config("configs/environment.yaml"), 30, False)
    assert schedule(0) == schedule(29)
