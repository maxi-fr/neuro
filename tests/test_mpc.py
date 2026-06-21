"""Verify the MPC Controller functionality."""

from __future__ import annotations

import numpy as np
import pytest

from neuro.jansen_rit import JansenRitParams, heun_step, simulate_network
from neuro.jansen_rit_casadi import JRSymbolicParams
from neuro.mpc import MPCController
from neuro.symbolic_model import JRSymbolicModel


def test_mpc_smoke_limit_cycle() -> None:
    """Smoke test for MPC controller reducing a limit cycle."""
    dt = 1e-3
    n_nodes = 1
    horizon_steps = 10

    # We want a limit cycle regime

    # We construct MPCController directly to avoid loading TVB connectome which takes time
    # and has 76 nodes.

    base = JansenRitParams(A=3.6, sigma=0.0)  # A=3.6 is limit cycle regime
    gamma = np.array([[1.0]])

    params = JRSymbolicParams(
        A=np.array([[base.A]]),
        B=base.B,
        a=base.a,
        b=base.b,
        C1=base.C1,
        C2=base.C2,
        C3=base.C3,
        C4=base.C4,
        e0=base.e0,
        v0=base.v0,
        r=base.r,
        mean_input=np.array([[base.mean_input]]),
        sigma=base.sigma,
        eeg_gain=np.ones((1, n_nodes)),
        gamma=gamma,
        K=1.0,
        w_weights=np.zeros((1, 1)),
        delay_steps=np.zeros((1, 1), dtype=np.int64),
    )

    mpc = MPCController(
        dt=dt,
        model=JRSymbolicModel(params, dt),
        horizon_steps=horizon_steps,
        R=0.01,
        R_du=0.0,
        bounds=(-10.0, 10.0),
    )

    # Run a short open loop to get into the limit cycle
    # simulate_network takes duration in seconds
    _, x_open = simulate_network(params=base, duration=1.0, dt=dt, seed=42)

    # Start MPC at the end of open loop
    x_hat = x_open[:, :, -1]

    u_vals = []
    y_vals = []

    # The reference is 0.0 (we want to suppress the limit cycle)
    ref = 0.0

    # We step the controller and the plant
    x_current = x_hat

    for _k in range(50):
        # Update controller
        u, _ = mpc.update(t=_k * dt, ref=ref, x_hat=x_current)
        u_arr = np.atleast_1d(u)
        u_vals.append(u_arr[0])

        # Step plant
        x_current = heun_step(
            x_current,
            u_arr[0],
            params=base,
            dt=dt,
            xi=0.0,
            coupling=0.0,  # single node
        )
        y_vals.append(x_current[1, 0] - x_current[2, 0])

    u_vals = np.array(u_vals)
    y_vals = np.array(y_vals)

    # Assert u respects bounds
    assert np.all(u_vals >= -10.0 - 1e-5)
    assert np.all(u_vals <= 10.0 + 1e-5)

    # We can't guarantee it fully suppresses it in 50 steps, but it should do something
    # compared to doing nothing.

    zero_ctrl_y = []
    x_zero = x_hat
    for _k in range(50):
        x_zero = heun_step(x_zero, 0.0, params=base, dt=dt, xi=0.0, coupling=0.0)
        zero_ctrl_y.append(x_zero[1, 0] - x_zero[2, 0])

    zero_ctrl_y = np.array(zero_ctrl_y)

    # Error should be lower with MPC
    assert np.sum(y_vals**2) < np.sum(zero_ctrl_y**2)


def test_mpc_multinode_delayed_build_and_update() -> None:
    """Build + solve a multi-node MPC with coupling and heterogeneous delays.

    Exercises the delayed, coupled path through the wrapped single-step ca.Function (the
    1-node smoke test above does not touch coupling or the delay history).
    """
    dt = 1e-3
    n_nodes = 3
    base = JansenRitParams(sigma=0.0)

    rng = np.random.default_rng(0)
    w = rng.uniform(0.0, 0.4, (n_nodes, n_nodes))
    w = np.triu(w, 1)
    w = w + w.T
    delays = np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=np.int64)

    params = JRSymbolicParams(
        A=np.full((1, n_nodes), base.A),
        B=base.B,
        a=base.a,
        b=base.b,
        C1=base.C1,
        C2=base.C2,
        C3=base.C3,
        C4=base.C4,
        e0=base.e0,
        v0=base.v0,
        r=base.r,
        mean_input=np.full((1, n_nodes), base.mean_input),
        sigma=0.0,
        eeg_gain=np.ones((1, n_nodes)),
        gamma=np.ones((1, n_nodes)),
        K=1.0,
        w_weights=w,
        delay_steps=delays,
    )

    mpc = MPCController(
        dt=dt,
        model=JRSymbolicModel(params, dt),
        horizon_steps=8,
        R=0.01,
        R_du=0.01,
        bounds=(-5.0, 5.0),
    )

    x_hat = np.zeros((6, n_nodes))
    x_hat[1, :] = 1.0
    for _k in range(3):
        u, _ = mpc.update(t=_k * dt, ref=0.0, x_hat=x_hat)
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
        assert np.all(np.isfinite(u_arr))
        assert np.all(u_arr >= -5.0 - 1e-5)
        assert np.all(u_arr <= 5.0 + 1e-5)
