"""Test SysID."""

import numpy as np

from neuro.connectome import Connectome
from neuro.control import ZeroController
from neuro.jansen_rit import JansenRitParams, simulate_network
from neuro.symbolic_model import JRSymbolicModel
from neuro.sysid import SysIDSolver


def _toy_connectome(n_nodes: int = 2) -> Connectome:
    """Small TVB-free connectome with zero delays (history_depth 0) for fast config tests."""
    rng = np.random.default_rng(0)
    weights = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0.0)
    labels = np.array([f"r{i}" for i in range(n_nodes)], dtype=np.str_)
    channels = np.array([f"c{i}" for i in range(n_nodes)], dtype=np.str_)
    return Connectome(
        weights=weights,
        tract_lengths=np.zeros((n_nodes, n_nodes)),
        centres=rng.random((n_nodes, 3)),
        region_labels=labels,
        hemispheres=np.zeros(n_nodes, dtype=np.bool_),
        speed=50.0,
        delays=np.zeros((n_nodes, n_nodes)),
        gain=np.eye(n_nodes),
        channel_labels=channels,
        region_index={label: idx for idx, label in enumerate(labels)},
        channel_index={label: idx for idx, label in enumerate(channels)},
        gamma=np.zeros((1, n_nodes)),
    )


def test_sysid_smoke() -> None:
    dt = 1e-3
    n_steps = 500

    base = JansenRitParams(A=3.25, sigma=0.0)

    # Generate synthetic data with A=4.0
    true_params = JansenRitParams(A=4.0, sigma=0.0)
    _, x_open = simulate_network(params=true_params, duration=0.5, dt=dt, seed=42)

    y_data = (x_open[1, 0, 1:] - x_open[2, 0, 1:]).reshape(-1, 1)
    u_data = np.zeros((n_steps, 1))

    solver = SysIDSolver(
        dt=dt,
        base_params=base,
        free_params=["A"],
        n_steps=n_steps,
        D=n_steps,
    )
    res = solver.solve(u_data, y_data)

    # Should recover A=4.0 approximately
    A_val = np.atleast_1d(res.free_params["A"])[0]  # noqa: N806
    assert A_val > 3.5


def test_sysid_weights_symmetric_zero_diagonal() -> None:
    """Identifying ``w_weights`` yields a symmetric, zero-diagonal, nonnegative matrix."""
    dt = 1e-3
    n_steps = 150
    n_nodes = 2

    w_true = np.array([[0.0, 0.8], [0.8, 0.0]])
    base = JansenRitParams(
        A=3.25,
        sigma=0.0,
        w_weights=w_true,
        delay_steps=np.zeros((n_nodes, n_nodes), dtype=np.int64),
        eeg_gain=np.eye(n_nodes),
        gamma=np.zeros((1, n_nodes)),
    )

    _, x_open = simulate_network(params=base, duration=n_steps * dt, dt=dt, seed=0)
    y_data = (x_open[1, :, 1:] - x_open[2, :, 1:]).T  # (n_steps, n_nodes)
    u_data = np.zeros((n_steps, 1))
    solver = SysIDSolver(dt=dt, base_params=base, free_params=["w_weights"], n_steps=n_steps)
    res = solver.solve(u_data, y_data)

    w = np.asarray(res.free_params["w_weights"])
    assert w.shape == (n_nodes, n_nodes)
    assert np.allclose(w, w.T, atol=1e-8)  # symmetric by construction
    assert np.allclose(np.diag(w), 0.0, atol=1e-8)  # zero diagonal
    assert (w >= -1e-8).all()  # nonnegative weights


def test_sysid_gamma_free_param_shape() -> None:
    """Freeing ``gamma`` builds an (n_elec, n_nodes) input projection and solves."""
    dt = 1e-3
    n_steps = 150
    n_nodes = 2
    gamma_true = np.array([[1.0, -0.5]])

    base = JansenRitParams(
        A=3.25,
        sigma=0.0,
        w_weights=np.zeros((n_nodes, n_nodes)),
        delay_steps=np.zeros((n_nodes, n_nodes), dtype=np.int64),
        eeg_gain=np.eye(n_nodes),
        gamma=gamma_true,
    )

    # Stimulate the plant so gamma is excited, then mirror the constant-current
    # schedule simulate_network applies internally over the same window.
    amp = 10.0
    stim_window = (0.02, 0.10)
    _, x_open = simulate_network(
        params=base, duration=n_steps * dt, dt=dt, seed=0, u_hat_tES=amp, stim_window=stim_window
    )
    y_data = (x_open[1, :, 1:] - x_open[2, :, 1:]).T  # (n_steps, n_nodes)

    t_grid = np.arange(n_steps) * dt
    u_data = np.where((t_grid >= stim_window[0]) & (t_grid < stim_window[1]), amp, 0.0).reshape(-1, 1)

    # Single shooting (D=n_steps): no continuity constraints to seed badly.
    solver = SysIDSolver(dt=dt, base_params=base, free_params=["gamma"], n_steps=n_steps, D=n_steps)
    res = solver.solve(u_data, y_data)

    g = np.asarray(res.free_params["gamma"])
    assert g.size == n_nodes  # (n_elec, n_nodes) input projection, not the old (1,1) scalar
    assert np.isfinite(res.cost)


def test_sysid_psd_loss_recovers_a() -> None:
    """A combined time + log-Welch-PSD loss still recovers the excitatory gain A."""
    dt = 1e-3
    n_steps = 500

    base = JansenRitParams(A=3.25, sigma=0.0)
    true_params = JansenRitParams(A=4.0, sigma=0.0)
    _, x_open = simulate_network(params=true_params, duration=0.5, dt=dt, seed=42)

    y_data = (x_open[1, 0, 1:] - x_open[2, 0, 1:]).reshape(-1, 1)
    u_data = np.zeros((n_steps, 1))

    solver = SysIDSolver(
        dt=dt,
        base_params=base,
        free_params=["A"],
        n_steps=n_steps,
        D=n_steps,
        psd_loss={"weight": 1.0, "nperseg": 128, "band": [1.0, 100.0]},
    )
    res = solver.solve(u_data, y_data)

    assert solver.psd_freqs is not None
    assert solver.psd_freqs.max() <= 100.0  # band mask applied
    a_val = np.atleast_1d(res.free_params["A"])[0]
    assert a_val > 3.5


def test_with_free_params_exposes_free_function_inputs() -> None:
    """``with_free_params`` makes each free parameter a trailing input of ``f_step``/``f_out``."""
    n_nodes = 2
    base = JansenRitParams(
        A=3.25,
        sigma=0.0,
        w_weights=np.zeros((n_nodes, n_nodes)),
        delay_steps=np.zeros((n_nodes, n_nodes), dtype=np.int64),
        eeg_gain=np.eye(n_nodes),
        gamma=np.zeros((1, n_nodes)),
    )

    model = JRSymbolicModel.with_free_params(base, dt=1e-3, free_params=["A", "w_weights"])

    assert list(model.free_syms) == ["A", "w_weights"]
    assert model.f_step.n_in() == 1 + 1 + 2  # one history state + control + two free params
    assert model.f_out.n_in() == 1 + 2  # state + two free params


def test_sysid_from_config_builds_from_base_model() -> None:
    """``from_config`` reads the ``base_model`` block (incl. ``w_weights_scale``) and builds a solver."""
    config = {
        "dt": 1e-3,
        "n_steps": 30,
        "free_params": ["A", "w_weights"],
        "base_model": {
            "connectome": _toy_connectome(2),
            "dt": 1e-3,
            "params": {"K": 0.5, "A": 3.25, "sigma": 0.0},
            "w_weights_scale": 2.0,
        },
    }

    solver = SysIDSolver.from_config(config)

    assert solver.n_steps == 30
    assert solver.free_params == ["A", "w_weights"]
    assert set(solver.model.free_syms) == {"A", "w_weights"}
