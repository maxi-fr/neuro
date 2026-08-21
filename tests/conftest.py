from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.observable import ObservableArtifact
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.config import ObservableGeometry
    from neuro.predictor.artifact import Activation


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --runslow command line option."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with @pytest.mark.slow unless --runslow is specified."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def make_observable_artifact() -> Callable[..., ObservableArtifact]:
    """Factory for a tiny synthetic observable artifact with random weights and real standardizers."""

    def build(
        geometry: ObservableGeometry,
        *,
        n_y: int = 4,
        n_u: int = 3,
        horizon: int = 12,
        n_channels: int = 2,
        activation: Activation = "softplus",
        dt: float = 0.02,
    ) -> ObservableArtifact:
        rng = np.random.default_rng(11)
        n_controls, z_dim, hidden, depth = 2, 5, 6, 2

        def layers(sizes: list[int]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
            return tuple(
                (rng.standard_normal((out, inp)) / np.sqrt(inp), rng.standard_normal(out) * 0.1)
                for inp, out in itertools.pairwise(sizes)
            )

        n_values = geometry.n_values(1.0 / dt)
        n_out = n_channels * n_values
        return ObservableArtifact(
            lift=layers([n_y * n_channels + n_u * n_controls, *[hidden] * depth, z_dim]),
            transition=layers([z_dim + n_controls, *[hidden] * depth, z_dim]),
            readout=(rng.standard_normal((n_out, z_dim)) / np.sqrt(z_dim), rng.standard_normal(n_out) * 0.1),
            activation=activation,
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            dt=dt,
            downsample=1,
            geometry=geometry,
            y_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_channels), scale=rng.uniform(0.5, 2.0, n_channels)),
            u_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_controls), scale=rng.uniform(0.5, 2.0, n_controls)),
            l_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_out), scale=rng.uniform(0.5, 2.0, n_out)),
        )

    return build
