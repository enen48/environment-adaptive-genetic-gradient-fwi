from src.utils.config import load_config, merge, storage_path


def test_configs_and_overrides():
    cfg = load_config("configs/smoke.yaml")
    assert cfg["physics"]["nt"] == 100
    assert cfg["physics"]["dt"] == 0.001
    assert cfg["generations"] == 3
    assert merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_storage_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FWI_HOME", str(tmp_path))
    assert storage_path("checkpoints/a.pt") == tmp_path / "checkpoints/a.pt"
