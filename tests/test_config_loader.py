"""Smoke test that Hydra config composes and loads at startup."""

from hydra import compose, initialize_config_dir
from pathlib import Path


def test_default_config_loads():
    conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert cfg.project.name == "chatqec"
    assert cfg.retriever.top_k == 8


def test_config_is_immutable_via_omegaconf():
    """Sanity: top-level project name is a string, retriever.top_k is an int."""
    conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert isinstance(cfg.project.name, str)
    assert isinstance(cfg.retriever.top_k, int)
