"""Tests for the simulate-framework Jansen-Rit plant and EEG output components.

Covers :class:`~neuro.jansen_rit.JansenRitDynamics` and
:class:`~neuro.output.EEGOutput`. The equivalence test pins the stepped component
to the standalone :func:`~neuro.jansen_rit.simulate_network` loop on a small
synthetic connectome (no TVB load); one smoke test exercises ``from_config`` against
the real TVB connectome.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, simulate_network
from neuro.output import EEGOutput

_DT = 1e-3
_N_REGIONS_TVB = 76
_N_CHANNELS_TVB = 62
_STATE_DIM = 6


def _toy_connectome(n_nodes: int = 3) -> Connectome:
    """Build a small deterministic connectome for fast, TVB-free tests."""
    rng = np.random.default_rng(0)
    weights = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0.0)
    tract_lengths = rng.random((n_nodes, n_nodes)) * 10.0
    speed = 50.0
    labels = np.array([f"r{i}" for i in range(n_nodes)], dtype=np.str_)
    channels = np.array([f"c{i}" for i in range(4)], dtype=np.str_)
    return Connectome(
        weights=weights,
        tract_lengths=tract_lengths,
        centres=rng.random((n_nodes, 3)),
        region_labels=labels,
        hemispheres=np.zeros(n_nodes, dtype=np.bool_),
        speed=speed,
        delays=tract_lengths / speed,
        gain=rng.random((4, n_nodes)),
        channel_labels=channels,
        region_index={label: idx for idx, label in enumerate(labels)},
        channel_index={label: idx for idx, label in enumerate(channels)},
    )


def test_dynamics_matches_simulate_network() -> None:
    """Stepping the component reproduces simulate_network exactly (same seed/params)."""
    conn = _toy_connectome(3)
    params = JansenRitParams(A=3.4)
    _t, x_ref = simulate_network(
        params=params,
        connectome=conn,
        K=0.5,
        duration=0.01,
        dt=_DT,
        seed=7,
    )
    n_steps = x_ref.shape[2] - 1

    dyn = JansenRitDynamics(dt=_DT, connectome=conn, K=0.5, params=params, seed=7)
    for k in range(n_steps):
        out, _ = dyn.evaluate(k * _DT, 0.0)
        np.testing.assert_allclose(np.asarray(out).reshape(6, 3), x_ref[:, :, k + 1], atol=1e-12)


def test_dynamics_from_config_builds_network() -> None:
    """from_config loads the TVB connectome and one step yields a finite flat state."""
    dyn = JansenRitDynamics.from_config({"dt": _DT, "K": 0.75, "params": {"A": 3.25}, "seed": 42, "speed": 50.0})
    assert dyn.x.shape == (_STATE_DIM, _N_REGIONS_TVB)

    out, _ = dyn.evaluate(0.0, 0.0)
    out = np.asarray(out)
    assert out.shape == (_STATE_DIM * _N_REGIONS_TVB,)
    assert np.isfinite(out).all()


def test_eeg_output_applies_gain() -> None:
    """EEGOutput maps the flat state to eeg = gain @ (x2 - x3)."""
    n_nodes = 3
    gain = np.arange(12, dtype=np.float64).reshape(4, n_nodes)
    out_comp = EEGOutput(dt=_DT, gain=gain)

    x_flat = np.arange(6 * n_nodes, dtype=np.float64)
    x_grid = x_flat.reshape(6, n_nodes)
    expected = gain @ (x_grid[1] - x_grid[2])

    y, _ = out_comp.update(0.0, x_flat, 0.0)
    np.testing.assert_allclose(np.asarray(y), expected)
    assert np.asarray(y).shape == (4,)


def test_eeg_output_from_config_channel_count() -> None:
    """from_config wires the (62, 76) TVB forward operator."""
    out_comp = EEGOutput.from_config({"dt": _DT, "speed": 50.0})
    assert out_comp.gain.shape == (_N_CHANNELS_TVB, _N_REGIONS_TVB)
