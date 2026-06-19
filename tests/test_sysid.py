"""Test SysID."""

import numpy as np

from neuro.control import ZeroController
from neuro.jansen_rit import JansenRitParams, simulate_network
from neuro.sysid import SysIDSolver


def test_sysid_smoke() -> None:
    dt = 1e-3
    n_steps = 20

    base = JansenRitParams(A=3.25, sigma=0.0)
    gamma = np.array([[1.0]])
    gain = np.array([[1.0]])
    w_weights = np.zeros((1, 1))
    delay_steps = np.zeros((1, 1), dtype=np.int64)

    # Generate synthetic data with A=4.0
    true_params = JansenRitParams(A=4.0, sigma=0.0)
    _, x_open = simulate_network(params=true_params, connectome=None, duration=0.02, dt=dt, seed=42)

    y_data = (x_open[1, 0, 1:] - x_open[2, 0, 1:]).reshape(-1, 1)
    u_data = np.zeros((n_steps, 1))

    # We will pass the first state as history
    x0_hist = [x_open[:, :, 0]]

    solver = SysIDSolver(
        dt=dt,
        base_params=base,
        free_params=["A"],
        gamma=gamma,
        gain=gain,
        w_weights=w_weights,
        delay_steps=delay_steps,
        n_steps=n_steps,
    )

    res = solver.solve(u_data, y_data, x0_hist)

    # Should recover A=4.0 approximately
    A_val = np.atleast_1d(res.params.A)[0]  # noqa: N806
    assert A_val > 3.5
