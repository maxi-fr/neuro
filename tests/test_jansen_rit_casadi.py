"""Verify the CasADi Jansen-Rit implementation against the numba/NumPy reference.

All tests call CasADi functions numerically and compare against:
  * ``_jr_rhs_jit`` / ``heun_step`` from ``jansen_rit.py``
  * ``_fx_step_jit`` from ``estimator.py``

The CasADi functions are deterministic (no noise), so comparisons are made with
``sigma = 0`` and ``xi = 0`` on the numba side.

State layout for CasADi (flat, length ``6 * n_nodes``, C-order from ``(6, N)``):
    x[k * N : (k+1) * N]  =  x_{k+1}  for all N nodes.
For N=1 this is ``[x1, x2, x3, x4, x5, x6]``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro.estimator import _fx_step_jit
from neuro.jansen_rit import JansenRitParams, _jr_rhs_jit, heun_step, sigmoid_jit, simulate_network
from neuro.jansen_rit_casadi import build_heun_step, build_jr_rhs

_DT = 1e-4
_SEED = 42
_RNG = np.random.default_rng(_SEED)


def _ca(result: object) -> np.ndarray:
    """Convert a CasADi DM to a flat numpy array."""
    import casadi as ca  # noqa: PLC0415

    return np.array(ca.DM(result)).flatten()


def test_rhs_single_node_matches_numba() -> None:
    """CasADi RHS equals numba RHS at five random states (zero coupling, zero u_tes)."""
    params = JansenRitParams()
    rhs = build_jr_rhs(params, n_nodes=1)
    params_tuple = params.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED)

    for _ in range(5):
        x = rng.standard_normal(6)
        dx_numba = _jr_rhs_jit(x, params_tuple, 0.0, 0.0)
        dx_casadi = _ca(rhs(x, [0.0], [0.0]))
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_single_node_with_coupling_and_utes() -> None:
    """CasADi RHS equals numba RHS with nonzero coupling and u_tes."""
    params = JansenRitParams()
    rhs = build_jr_rhs(params, n_nodes=1)
    params_tuple = params.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED + 1)

    for _ in range(5):
        x = rng.standard_normal(6)
        coupling = float(rng.uniform(-1.0, 1.0))
        u_tes = float(rng.uniform(-0.5, 0.5))
        dx_numba = _jr_rhs_jit(x, params_tuple, coupling, u_tes)
        dx_casadi = _ca(rhs(x, [coupling], [u_tes]))
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_multi_node_matches_numba() -> None:
    """CasADi RHS (N=2) equals numba RHS after flattening the (6, 2) output."""
    n_nodes = 2
    params = JansenRitParams(A=np.array([3.25, 3.6]))
    rhs = build_jr_rhs(params, n_nodes=n_nodes)
    params_tuple = params.to_numba_tuple(n_nodes)
    rng = np.random.default_rng(_SEED + 2)

    for _ in range(5):
        x_2d = rng.standard_normal((6, n_nodes))
        coupling = rng.uniform(-1.0, 1.0, size=n_nodes)
        u_tes = rng.uniform(-0.5, 0.5, size=n_nodes)

        dx_numba = _jr_rhs_jit(x_2d, params_tuple, coupling, u_tes)  # shape (6, 2)
        dx_casadi = _ca(rhs(x_2d.flatten(), coupling, u_tes))  # shape (12,)
        np.testing.assert_allclose(dx_casadi, dx_numba.flatten(), rtol=1e-10, atol=1e-12)


def test_heun_step_matches_noiseless_heun() -> None:
    """CasADi Heun step equals numba heun_step(..., sigma=0, xi=0) for single node."""
    params = JansenRitParams(sigma=0.0)
    step = build_heun_step(params, _DT, n_nodes=1)
    rng = np.random.default_rng(_SEED + 3)

    for _ in range(5):
        x = rng.standard_normal(6)
        coupling = float(rng.uniform(-1.0, 1.0))
        u_tes = float(rng.uniform(-0.5, 0.5))

        x_numba = heun_step(x, u_tes, params, _DT, xi=0.0, coupling=coupling)
        x_casadi = _ca(step(x, [coupling], [u_tes]))
        np.testing.assert_allclose(x_casadi, x_numba, rtol=1e-10, atol=1e-12)


def test_heun_step_matches_estimator_fx() -> None:
    """CasADi Heun step matches the dynamic-state slice of ``_fx_step_jit`` (N=2, zero delays)."""
    n_nodes = 2
    params = JansenRitParams()
    step = build_heun_step(params, _DT, n_nodes=n_nodes)

    # Build params_tuple as the estimator does (A broadcast to array)
    a_vec = np.full(n_nodes, params.A, dtype=np.float64)
    net_params = JansenRitParams(
        A=a_vec,
        B=params.B,
        a=params.a,
        b=params.b,
        C1=params.C1,
        C2=params.C2,
        C3=params.C3,
        C4=params.C4,
        e0=params.e0,
        v0=params.v0,
        r=params.r,
        mean_input=params.mean_input,
        sigma=params.sigma,
    )
    params_tuple = net_params.to_numba_tuple(n_nodes)

    rng = np.random.default_rng(_SEED + 4)
    x_2d = rng.standard_normal((6, n_nodes))
    x_flat = x_2d.flatten()  # C-order: [x1_0, x1_1, x2_0, x2_1, ..., x6_0, x6_1]

    K = 0.5  # noqa: N806
    W = np.array([[0.0, 0.8], [0.3, 0.0]])  # (2, 2), zero diagonal  # noqa: N806

    # Augmented state vector: [x_dyn(12), K(1), W.flat(4)]
    x_aug = np.zeros(6 * n_nodes + 1 + n_nodes**2)
    x_aug[: 6 * n_nodes] = x_flat
    x_aug[6 * n_nodes] = K
    x_aug[6 * n_nodes + 1 :] = W.flatten()

    u_node = rng.uniform(-0.1, 0.1, size=n_nodes)

    # Zero delays: _fx_step_jit uses current sigmoid values, not history
    delay_steps = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    history = np.zeros((1, n_nodes), dtype=np.float64)  # max_history_len=1

    x_aug_next = _fx_step_jit(x_aug, _DT, u_node, 0, 1, delay_steps, history, params_tuple, -1, -1)
    x_dyn_next_estimator = x_aug_next[: 6 * n_nodes]

    # Manually compute coupling as _fx_step_jit does with delay=0
    s_y = sigmoid_jit(x_2d[1] - x_2d[2], params.e0, params.v0, params.r)
    coupling = K * (W @ s_y)

    x_next_casadi = _ca(step(x_flat, coupling, u_node))

    np.testing.assert_allclose(x_next_casadi, x_dyn_next_estimator, rtol=1e-10, atol=1e-12)


def test_casadi_trajectory_matches_simulate_network() -> None:
    """1000-step CasADi trajectory equals ``simulate_network`` with sigma=0 (single node)."""
    n_steps = 1000
    params = JansenRitParams(sigma=0.0)
    step = build_heun_step(params, _DT, n_nodes=1)

    _, x_ref = simulate_network(params=params, connectome=None, duration=n_steps * _DT, dt=_DT, seed=_SEED)
    # x_ref shape: (6, 1, n_steps+1)

    x = x_ref[:, 0, 0].copy()  # initial state
    for _ in range(n_steps):
        x = _ca(step(x, [0.0], [0.0]))

    np.testing.assert_allclose(x, x_ref[:, 0, n_steps], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("a_gain", "expect_oscillation"), [(3.25, False), (3.6, True)])
def test_casadi_fixed_point_and_limit_cycle(a_gain: float, *, expect_oscillation: bool) -> None:
    """A=3.25 is a fixed point; A=3.6 produces a limit cycle (ptp > 5 mV)."""
    params = JansenRitParams(A=a_gain, sigma=0.0)
    step = build_heun_step(params, _DT, n_nodes=1)

    duration = 8.0
    transient_steps = round(3.0 / _DT)
    n_steps = round(duration / _DT)

    x = np.zeros(6)
    traj = np.empty((n_steps + 1, 6))
    traj[0] = x
    for k in range(n_steps):
        x = _ca(step(x, [0.0], [0.0]))
        traj[k + 1] = x

    y = traj[transient_steps:, 1] - traj[transient_steps:, 2]  # x2 - x3
    ptp = np.ptp(y)

    if expect_oscillation:
        assert ptp > 5.0, f"Expected limit cycle at A={a_gain}, ptp={ptp:.3f}"
    else:
        assert ptp < 0.05, f"Expected fixed point at A={a_gain}, ptp={ptp:.3f}"
