from src.inversion.stopping import Stopping


def test_stagnation_warmup_and_injection_stop():
    stopping = Stopping({"patience":2,"min_progress":.6,"epsilon":1e-4})
    assert stopping.update(1.,progress=0.) is None
    assert stopping.update(1.,progress=.3) is None
    assert stopping.update(1.,progress=.6) is None
    assert stopping.update(1.,progress=.8) == "reference_stagnation"
    stopping = Stopping({"failed_injections":2})
    assert stopping.update(1.,False) is None
    assert stopping.update(.9,False) == "collapse_injection_without_improvement"


def test_target_and_optional_validation():
    assert Stopping({"target":.1}).update(.01) == "target_fitness"
    assert Stopping({"validation_threshold":.1}).update(1.,validation_loss=.05) == "validation_threshold"
