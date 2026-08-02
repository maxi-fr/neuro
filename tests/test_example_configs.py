from pathlib import Path

import pytest
from simulate.config import load_config as load_sim_config

from neuro.config import load_config as load_nn_config

CONFIG_DIR = Path("configs")

nn_configs = list((CONFIG_DIR / "nn_predictor").rglob("*.yaml"))
sim_configs = list((CONFIG_DIR / "simulation").rglob("*.yaml"))


@pytest.mark.parametrize("config_path", nn_configs, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_nn_predictor_config_valid(config_path: Path) -> None:
    """Ensure that all NN predictor example configs load successfully."""

    config = load_nn_config(config_path)
    assert config is not None


@pytest.mark.parametrize("config_path", sim_configs, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_simulation_config_valid(config_path: Path) -> None:
    """Ensure that all simulation example configs load successfully."""

    config = load_sim_config(config_path)
    assert config is not None
