"""Test SysID."""

import numpy as np

from neuro.control import ZeroController
from neuro.jansen_rit import JansenRitParams, simulate_network
from neuro.sysid import SysIDSolver


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

    res = solver.solve(u_data, y_data, x0_guess=x_open[:, :, 0])

    # Should recover A=4.0 approximately
    A_val = np.atleast_1d(res.params.A)[0]  # noqa: N806
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
    res = solver.solve(u_data, y_data, x0_guess=x_open[:, :, 0])

    w = np.asarray(res.params.w_weights)
    assert w.shape == (n_nodes, n_nodes)
    assert np.allclose(w, w.T, atol=1e-8)  # symmetric by construction
    assert np.allclose(np.diag(w), 0.0, atol=1e-8)  # zero diagonal
    assert (w >= -1e-8).all()  # nonnegative weights
