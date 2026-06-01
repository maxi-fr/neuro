"""Tests for the whole-brain plant components in the ``simulate`` framework.

Each plant is split into a :class:`~simulate.dynamics.Dynamics` component (state
transition, driven one ``dt`` per :meth:`evaluate`) and an
:class:`~simulate.output.Output` component (EEG lead-field projection).  The
``Dynamics`` run with ``integrator=None`` as discrete-time step transitions.

Coverage:

* Output shapes and lead-field projection (``Output.update``).
* Reproducibility: same seed -> identical trajectory.
* ``reset()`` reproduces the first run (including the injected control parameter).
* Finite, non-degenerate (oscillating) dynamics.
* Control input ``B @ u`` actually moves the trajectory.
* ``from_config`` round-trips.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from neuro.jansen_rit_plant import TVBJansenRitDynamics, TVBJansenRitOutput
from neuro.plant import (
    FHNOutput,
    NativeFHNDynamics,
    TVBFHNDynamics,
    TVBFHNOutput,
)

_SEED = 42
_DT = 0.1
_N_SENSORS = 64
_N_SENSORS_TVB = 65  # TVB ships a 65-sensor BEM surface EEG projection
_N_NODES = 80  # neurolib HCP connectome
_N_NODES_TVB = 76  # TVB default connectivity_76.zip (AAL atlas)

FloatArray = np.ndarray


def _run(
    dynamics: NativeFHNDynamics | TVBFHNDynamics | TVBJansenRitDynamics,
    output: FHNOutput | TVBFHNOutput | TVBJansenRitOutput,
    duration_ms: float,
    u: np.ndarray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Drive ``dynamics`` + ``output`` for ``duration_ms`` via ``evaluate``.

    Returns ``(states, eeg)`` with shapes ``(state_dim, n_samples)`` and
    ``(n_sensors, n_samples)``.
    """
    n_steps = round(duration_ms / dynamics.dt)
    u_vec = np.zeros((dynamics.b.shape[1], 1)) if u is None else u
    state_cols: list[FloatArray] = []
    eeg_cols: list[FloatArray] = []
    t = 0.0
    for _ in range(n_steps):
        x_raw, _ = dynamics.evaluate(t, u_vec)
        y_raw, _ = output.evaluate(t, x_raw, u_vec)
        state_cols.append(np.atleast_1d(x_raw).reshape(-1, 1))
        eeg_cols.append(np.atleast_1d(y_raw).reshape(-1, 1))
        t += dynamics.dt
    return np.hstack(state_cols), np.hstack(eeg_cols)


# ---------------------------------------------------------------------------
# NativeFHNDynamics + FHNOutput
# ---------------------------------------------------------------------------


def _make_native(sigma_ou: float = 0.0, seed: int = _SEED) -> tuple[NativeFHNDynamics, FHNOutput]:
    dynamics = NativeFHNDynamics(dt=_DT, sigma_ou=sigma_ou, seed=seed)
    output = FHNOutput(dt=_DT, n_sensors=_N_SENSORS, leadfield_seed=0, n_nodes=dynamics.n_nodes)
    return dynamics, output


def test_native_output_shapes() -> None:
    """State is (2*n_nodes, n_samples) and EEG is (n_sensors, n_samples)."""
    dynamics, output = _make_native()
    states, eeg = _run(dynamics, output, 100.0)
    n_samples = round(100.0 / _DT)
    assert states.shape == (2 * _N_NODES, n_samples)
    assert eeg.shape == (_N_SENSORS, n_samples)


def test_native_reproducible() -> None:
    """Two instances with the same seed produce identical trajectories."""
    d1, o1 = _make_native(sigma_ou=0.05)
    d2, o2 = _make_native(sigma_ou=0.05)
    s1, e1 = _run(d1, o1, 200.0)
    s2, e2 = _run(d2, o2, 200.0)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.parametrize("sigma_ou", [0.0, 0.05])
def test_native_reset_reproduces(sigma_ou: float) -> None:
    """After reset() the dynamics reproduce the same trajectory."""
    dynamics, output = _make_native(sigma_ou=sigma_ou)
    s_first, _ = _run(dynamics, output, 100.0)
    dynamics.reset()
    s_second, _ = _run(dynamics, output, 100.0)
    np.testing.assert_array_equal(s_first, s_second)


def test_native_finite_with_noise() -> None:
    """Output stays finite with non-zero OU noise."""
    dynamics, output = _make_native(sigma_ou=0.05, seed=1)
    states, eeg = _run(dynamics, output, 200.0)
    assert np.isfinite(states).all()
    assert np.isfinite(eeg).all()


def test_native_oscillates() -> None:
    """Noiseless FHN settles onto a limit cycle, not a fixed point."""
    dynamics, output = _make_native(sigma_ou=0.0, seed=0)
    states, _ = _run(dynamics, output, 500.0)
    activity = states[:_N_NODES]  # V
    assert np.all(activity.std(axis=1) > 1e-6)


def test_native_control_changes_trajectory() -> None:
    """A non-zero control input B @ u changes the trajectory."""
    d0, o0 = _make_native(sigma_ou=0.0)
    du, ou = _make_native(sigma_ou=0.0)
    s_zero, _ = _run(d0, o0, 50.0)
    s_ctrl, _ = _run(du, ou, 50.0, u=np.array([[5.0]]))
    assert not np.allclose(s_zero, s_ctrl)


def test_native_from_config_roundtrip() -> None:
    """from_config reproduces a directly-constructed instance's trajectory."""
    direct, output = _make_native(sigma_ou=0.05, seed=3)
    cfg = NativeFHNDynamics.from_config({"dt": _DT, "sigma_ou": 0.05, "seed": 3})
    s_direct, _ = _run(direct, output, 100.0)
    cfg_out = FHNOutput(dt=_DT, n_sensors=_N_SENSORS, leadfield_seed=0, n_nodes=cfg.n_nodes)
    s_cfg, _ = _run(cfg, cfg_out, 100.0)
    np.testing.assert_array_equal(s_direct, s_cfg)


# ---------------------------------------------------------------------------
# FHNOutput lead-field
# ---------------------------------------------------------------------------


def test_fhn_output_leadfield_row_normalized() -> None:
    """The default random lead-field has unit-norm rows."""
    output = FHNOutput(dt=_DT, n_sensors=_N_SENSORS, leadfield_seed=0, n_nodes=_N_NODES)
    assert output.leadfield.shape == (_N_SENSORS, _N_NODES)
    np.testing.assert_allclose(np.linalg.norm(output.leadfield, axis=1), 1.0)


def test_fhn_output_update_projects_v() -> None:
    """Output.update applies leadfield @ V (the first n_nodes of [V; W])."""
    output = FHNOutput(dt=_DT, n_sensors=_N_SENSORS, leadfield_seed=0, n_nodes=_N_NODES)
    v = np.arange(_N_NODES, dtype=np.float64).reshape(-1, 1)
    w = np.zeros((_N_NODES, 1))
    x = np.vstack([v, w])
    y, log = output.update(0.0, x, 0.0)
    expected = output.leadfield @ v
    np.testing.assert_allclose(cast("FloatArray", y), expected.flatten())
    np.testing.assert_allclose(log.y, expected)


def test_fhn_output_leadfield_path_loads(tmp_path: Path) -> None:
    """A precomputed (n_sensors, n_nodes) .npy is loaded verbatim."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((_N_SENSORS, _N_NODES))
    path = tmp_path / "leadfield.npy"
    np.save(path, matrix)
    output = FHNOutput(dt=_DT, leadfield_path=path, n_nodes=_N_NODES)
    np.testing.assert_array_equal(output.leadfield, matrix)


def test_fhn_output_leadfield_path_shape_mismatch_raises(tmp_path: Path) -> None:
    """A lead-field whose column count != n_nodes is rejected."""
    path = tmp_path / "bad_leadfield.npy"
    np.save(path, np.zeros((_N_SENSORS, _N_NODES - 1)))
    with pytest.raises(ValueError, match="columns"):
        FHNOutput(dt=_DT, leadfield_path=path, n_nodes=_N_NODES)


def test_fhn_output_from_config() -> None:
    """FHNOutput.from_config honours the supplied lead-field settings."""
    output = FHNOutput.from_config({"dt": _DT, "n_sensors": _N_SENSORS, "leadfield_seed": 0, "n_nodes": _N_NODES})
    reference = FHNOutput(dt=_DT, n_sensors=_N_SENSORS, leadfield_seed=0, n_nodes=_N_NODES)
    np.testing.assert_array_equal(output.leadfield, reference.leadfield)


# ---------------------------------------------------------------------------
# evaluate() zero-order-hold behaviour
# ---------------------------------------------------------------------------


def test_evaluate_zoh_holds_within_dt() -> None:
    """Re-evaluating at the same time returns the cached state (no extra step)."""
    dynamics, _ = _make_native(sigma_ou=0.05)
    x_first, _ = dynamics.evaluate(0.0, np.zeros((1, 1)))
    x_again, _ = dynamics.evaluate(0.0, np.zeros((1, 1)))
    np.testing.assert_array_equal(np.atleast_1d(x_first), np.atleast_1d(x_again))


# ---------------------------------------------------------------------------
# TVBFHNDynamics + TVBFHNOutput
# ---------------------------------------------------------------------------


def _make_tvb(nsig: float = 0.0, seed: int = 0) -> tuple[TVBFHNDynamics, TVBFHNOutput]:
    dynamics = TVBFHNDynamics(dt=_DT, nsig=nsig, seed=seed)
    output = TVBFHNOutput(dt=_DT, n_nodes=dynamics.n_nodes)
    return dynamics, output


def test_tvb_output_shapes() -> None:
    """TVBFHN state is (2*n_nodes, n_samples) and EEG (n_sensors, n_samples)."""
    dynamics, output = _make_tvb(seed=0)
    states, eeg = _run(dynamics, output, 100.0)
    n_samples = round(100.0 / _DT)
    assert states.shape == (2 * _N_NODES_TVB, n_samples)
    assert eeg.shape == (_N_SENSORS_TVB, n_samples)


def test_tvb_leadfield_is_real_projection() -> None:
    """TVBFHNOutput uses TVB's deterministic BEM-derived region-averaged leadfield."""
    lf1 = TVBFHNOutput(dt=_DT, n_nodes=_N_NODES_TVB).leadfield
    lf2 = TVBFHNOutput(dt=_DT, n_nodes=_N_NODES_TVB).leadfield
    assert lf1.shape == (_N_SENSORS_TVB, _N_NODES_TVB)
    np.testing.assert_array_equal(lf1, lf2)
    # Real region-averaged BEM gain — rows are not unit-norm.
    assert not np.allclose(np.linalg.norm(lf1, axis=1), 1.0)


def test_tvb_reproducible() -> None:
    """Two TVBFHN instances with the same seed and nsig=0 produce identical output."""
    d1, o1 = _make_tvb(nsig=0.0, seed=42)
    d2, o2 = _make_tvb(nsig=0.0, seed=42)
    s1, e1 = _run(d1, o1, 100.0)
    s2, e2 = _run(d2, o2, 100.0)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.parametrize("nsig", [0.0, 0.05])
def test_tvb_reset_reproduces(nsig: float) -> None:
    """After reset(), TVBFHN reproduces the same trajectory."""
    dynamics, output = _make_tvb(nsig=nsig, seed=7)
    s_first, _ = _run(dynamics, output, 100.0)
    dynamics.reset()
    s_second, _ = _run(dynamics, output, 100.0)
    np.testing.assert_array_equal(s_first, s_second)


def test_tvb_finite_with_noise() -> None:
    """TVBFHN output is finite with non-zero noise."""
    dynamics, output = _make_tvb(nsig=0.01, seed=1)
    states, eeg = _run(dynamics, output, 200.0)
    assert np.isfinite(states).all()
    assert np.isfinite(eeg).all()


def test_tvb_default_connectome_is_76_nodes() -> None:
    """TVBFHNDynamics defaults to TVB's 76-node AAL atlas (connectivity_76.zip)."""
    assert TVBFHNDynamics().n_nodes == _N_NODES_TVB


def test_tvb_control_changes_trajectory() -> None:
    """A non-zero control input (model.I) changes the TVBFHN trajectory."""
    d0, o0 = _make_tvb(nsig=0.0, seed=0)
    du, ou = _make_tvb(nsig=0.0, seed=0)
    s_zero, _ = _run(d0, o0, 50.0)
    s_ctrl, _ = _run(du, ou, 50.0, u=np.array([[2.0]]))
    assert not np.allclose(s_zero, s_ctrl)


def test_tvb_from_config_roundtrip() -> None:
    """from_config reproduces a directly-constructed TVBFHN trajectory."""
    direct, output = _make_tvb(nsig=0.0, seed=5)
    cfg = TVBFHNDynamics.from_config({"dt": _DT, "nsig": 0.0, "seed": 5})
    cfg_out = TVBFHNOutput(dt=_DT, n_nodes=cfg.n_nodes)
    s_direct, _ = _run(direct, output, 100.0)
    s_cfg, _ = _run(cfg, cfg_out, 100.0)
    np.testing.assert_array_equal(s_direct, s_cfg)


# ---------------------------------------------------------------------------
# TVBJansenRitDynamics + TVBJansenRitOutput
# ---------------------------------------------------------------------------


def _make_jr(nsig: float = 0.0, seed: int = 0) -> tuple[TVBJansenRitDynamics, TVBJansenRitOutput]:
    dynamics = TVBJansenRitDynamics(dt=_DT, nsig=nsig, seed=seed)
    output = TVBJansenRitOutput(dt=_DT, n_nodes=dynamics.n_nodes)
    return dynamics, output


def test_jr_output_shapes() -> None:
    """JR state is (6*n_nodes, n_samples) and EEG (n_sensors, n_samples)."""
    dynamics, output = _make_jr(seed=0)
    states, eeg = _run(dynamics, output, 100.0)
    n_samples = round(100.0 / _DT)
    assert states.shape == (6 * _N_NODES_TVB, n_samples)
    assert eeg.shape == (_N_SENSORS_TVB, n_samples)


def test_jr_default_connectome_is_76_nodes() -> None:
    """TVBJansenRitDynamics defaults to TVB's 76-node AAL atlas."""
    assert TVBJansenRitDynamics().n_nodes == _N_NODES_TVB


def test_jr_leadfield_is_real_projection() -> None:
    """TVBJansenRitOutput uses TVB's deterministic region-averaged leadfield."""
    lf1 = TVBJansenRitOutput(dt=_DT, n_nodes=_N_NODES_TVB).leadfield
    lf2 = TVBJansenRitOutput(dt=_DT, n_nodes=_N_NODES_TVB).leadfield
    assert lf1.shape == (_N_SENSORS_TVB, _N_NODES_TVB)
    np.testing.assert_array_equal(lf1, lf2)


def test_jr_reproducible() -> None:
    """Two JR instances with the same seed and nsig=0 produce identical output."""
    d1, o1 = _make_jr(nsig=0.0, seed=42)
    d2, o2 = _make_jr(nsig=0.0, seed=42)
    s1, e1 = _run(d1, o1, 100.0)
    s2, e2 = _run(d2, o2, 100.0)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.parametrize("nsig", [0.0, 0.05])
def test_jr_reset_reproduces(nsig: float) -> None:
    """After reset(), JR reproduces the same trajectory."""
    dynamics, output = _make_jr(nsig=nsig, seed=7)
    s_first, _ = _run(dynamics, output, 100.0)
    dynamics.reset()
    s_second, _ = _run(dynamics, output, 100.0)
    np.testing.assert_array_equal(s_first, s_second)


def test_jr_finite_with_noise() -> None:
    """JR output is finite with non-zero noise."""
    dynamics, output = _make_jr(nsig=0.01, seed=1)
    states, eeg = _run(dynamics, output, 200.0)
    assert np.isfinite(states).all()
    assert np.isfinite(eeg).all()


def test_jr_oscillates() -> None:
    """Over ~1 s the JR network oscillates rather than resting at a fixed point."""
    dynamics, output = _make_jr(nsig=0.0, seed=0)
    states, _ = _run(dynamics, output, 1000.0)
    # Pyramidal potential y1 - y2 per node (the EEG source signal).
    activity = states[_N_NODES_TVB : 2 * _N_NODES_TVB] - states[2 * _N_NODES_TVB : 3 * _N_NODES_TVB]
    assert np.isfinite(activity).all()
    assert np.all(activity.std(axis=1) > 1e-6)


def test_jr_control_changes_trajectory() -> None:
    """A non-zero control input (model.mu) changes the JR trajectory."""
    d0, o0 = _make_jr(nsig=0.0, seed=0)
    du, ou = _make_jr(nsig=0.0, seed=0)
    s_zero, _ = _run(d0, o0, 50.0)
    s_ctrl, _ = _run(du, ou, 50.0, u=np.array([[0.1]]))
    assert not np.allclose(s_zero, s_ctrl)


def test_jr_from_config_roundtrip() -> None:
    """from_config reproduces a directly-constructed JR trajectory."""
    direct, output = _make_jr(nsig=0.0, seed=5)
    cfg = TVBJansenRitDynamics.from_config({"dt": _DT, "nsig": 0.0, "seed": 5})
    cfg_out = TVBJansenRitOutput(dt=_DT, n_nodes=cfg.n_nodes)
    s_direct, _ = _run(direct, output, 100.0)
    s_cfg, _ = _run(cfg, cfg_out, 100.0)
    np.testing.assert_array_equal(s_direct, s_cfg)
