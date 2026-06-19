# ruff: noqa: N806
"""Verify the CasADi Jansen-Rit implementation against the numba/NumPy reference.

The CasADi functions build raw symbolic ``ca.SX`` expressions, so each test wraps
the expression in a ``ca.Function`` and evaluates it numerically, comparing against:
  * ``sigmoid_jit`` / ``_jr_rhs_jit`` / ``heun_step`` from ``jansen_rit.py``
  * ``_fx_step_jit`` from ``estimator.py``

The CasADi functions are deterministic (no noise), so comparisons use ``sigma = 0``
and ``xi = 0`` on the numba side.

State layout for CasADi: a ``6 x N`` matrix, row ``k`` = state variable ``x_{k+1}``
for all ``N`` nodes (this is ``np.reshape(x_aug[:6N], (6, N))`` from the estimator;
flattening in C order reproduces the estimator's dynamic-state slice).
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from neuro.estimator import _fx_step_jit
from neuro.jansen_rit import (
    JansenRitParams,
    _jr_rhs_jit,
    heun_step,
    sigmoid_jit,
    simulate_network,
)
from neuro.jansen_rit_casadi import f_rhs, get_network_coupling, sigmoid
from neuro.jansen_rit_casadi import heun_step as ca_heun_step

_DT = 1e-4
_SEED = 42


def _np2(result: object) -> np.ndarray:
    """Convert a CasADi DM/evaluation result to a 2-D numpy array."""
    return np.array(ca.DM(result))


def _a_row(params: JansenRitParams, n_nodes: int) -> np.ndarray:
    """Broadcast ``params.A`` to a ``(1, n_nodes)`` row for the CasADi A input."""
    a = np.asarray(params.A, dtype=np.float64).reshape(-1)
    if a.size == 1:
        a = np.full(n_nodes, a.item())
    return a.reshape(1, n_nodes)


def _rhs_fn(params: JansenRitParams, n_nodes: int) -> ca.Function:
    """Wrap ``f_rhs`` in an evaluable ``ca.Function`` of (X, coupling, u, A)."""
    X = ca.SX.sym("X", 6, n_nodes)
    coupling = ca.SX.sym("coupling", 1, n_nodes)
    u = ca.SX.sym("u", 1, n_nodes)
    A = ca.SX.sym("A", 1, n_nodes)
    return ca.Function("rhs", [X, coupling, u, A], [f_rhs(X, coupling, u, A, params)])


def _heun_fn(params: JansenRitParams, dt: float, n_nodes: int, delay_steps: np.ndarray) -> ca.Function:
    """Wrap ``ca_heun_step`` in an evaluable ``ca.Function`` of (history..., u, K, W, A).

    The history list length is fixed by the largest delay; ``delay_steps`` is baked
    into the symbolic expression (it indexes the history list at build time).
    """
    max_hist = int(delay_steps.max()) + 1
    hist = [ca.SX.sym(f"X{t}", 6, n_nodes) for t in range(max_hist)]
    u = ca.SX.sym("u", 1, n_nodes)
    K = ca.SX.sym("K")
    W = ca.SX.sym("W", n_nodes, n_nodes)
    A = ca.SX.sym("A", 1, n_nodes)
    x_next = ca_heun_step(hist, u, K, W, delay_steps, A, params, dt)
    return ca.Function("heun", [*hist, u, K, W, A], [x_next])


def test_sigmoid_matches_numba() -> None:
    """CasADi ``sigmoid`` equals ``sigmoid_jit`` across a range of potentials."""
    params = JansenRitParams()
    v = ca.SX.sym("v")
    fn = ca.Function("s", [v], [sigmoid(v, params)])
    for val in np.linspace(-20.0, 20.0, 11):
        got = _np2(fn(val)).item()
        want = sigmoid_jit(val, params.e0, params.v0, params.r)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_rhs_single_node_matches_numba() -> None:
    """CasADi RHS equals numba RHS at five random states (zero coupling, zero u)."""
    params = JansenRitParams()
    rhs = _rhs_fn(params, 1)
    params_tuple = params.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED)

    for _ in range(5):
        x = rng.standard_normal(6)
        dx_numba = _jr_rhs_jit(x, params_tuple, 0.0, 0.0)
        dx_casadi = _np2(rhs(x.reshape(6, 1), 0.0, 0.0, params.A)).flatten()
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_single_node_with_coupling_and_u() -> None:
    """CasADi RHS equals numba RHS with nonzero coupling and u."""
    params = JansenRitParams()
    rhs = _rhs_fn(params, 1)
    params_tuple = params.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED + 1)

    for _ in range(5):
        x = rng.standard_normal(6)
        coupling = float(rng.uniform(-1.0, 1.0))
        u = float(rng.uniform(-0.5, 0.5))
        dx_numba = _jr_rhs_jit(x, params_tuple, coupling, u)
        dx_casadi = _np2(rhs(x.reshape(6, 1), coupling, u, params.A)).flatten()
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_multi_node_matches_numba() -> None:
    """CasADi RHS (N=2) equals numba RHS, comparing the (6, 2) matrices directly."""
    n_nodes = 2
    params = JansenRitParams(A=np.array([3.25, 3.6]))
    rhs = _rhs_fn(params, n_nodes)
    params_tuple = params.to_numba_tuple(n_nodes)
    rng = np.random.default_rng(_SEED + 2)

    for _ in range(5):
        x_2d = rng.standard_normal((6, n_nodes))
        coupling = rng.uniform(-1.0, 1.0, size=n_nodes)
        u = rng.uniform(-0.5, 0.5, size=n_nodes)

        dx_numba = _jr_rhs_jit(x_2d, params_tuple, coupling, u)  # (6, 2)
        dx_casadi = _np2(rhs(x_2d, coupling.reshape(1, n_nodes), u.reshape(1, n_nodes), _a_row(params, n_nodes)))
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_coupling_zero_delay_matches_reference() -> None:
    """Zero-delay coupling equals ``K * (W @ S(x2 - x3))`` (the paper/numba form)."""
    n_nodes = 3
    params = JansenRitParams()
    rng = np.random.default_rng(_SEED + 5)

    X = ca.SX.sym("X", 6, n_nodes)
    K = ca.SX.sym("K")
    W = ca.SX.sym("W", n_nodes, n_nodes)
    D = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    fn = ca.Function("c", [X, K, W], [get_network_coupling([X], K, W, D, params)])

    x_2d = rng.standard_normal((6, n_nodes))
    k_val = 0.4
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)

    got = _np2(fn(x_2d, k_val, w)).flatten()
    s_y = sigmoid_jit(x_2d[1] - x_2d[2], params.e0, params.v0, params.r)
    want = k_val * (w @ s_y)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_coupling_with_delays_matches_reference() -> None:
    """Delayed coupling reads ``S(x2 - x3)`` from the per-edge delayed history slot."""
    n_nodes = 2
    params = JansenRitParams()
    rng = np.random.default_rng(_SEED + 6)

    D = np.array([[0, 2], [1, 0]], dtype=np.int64)
    max_hist = int(D.max()) + 1
    hist_syms = [ca.SX.sym(f"X{t}", 6, n_nodes) for t in range(max_hist)]
    K = ca.SX.sym("K")
    W = ca.SX.sym("W", n_nodes, n_nodes)
    fn = ca.Function("c", [*hist_syms, K, W], [get_network_coupling(hist_syms, K, W, D, params)])

    hist = [rng.standard_normal((6, n_nodes)) for _ in range(max_hist)]
    k_val = 0.7
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))

    got = _np2(fn(*hist, k_val, w)).flatten()

    want = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            past = hist[D[i, j]]
            s = sigmoid_jit(past[1, j] - past[2, j], params.e0, params.v0, params.r)
            want[i] += k_val * w[i, j] * s
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_heun_step_single_node_matches_jansen_rit() -> None:
    """CasADi Heun step (self-coupled node) equals numba ``heun_step`` with matching coupling."""
    params = JansenRitParams(sigma=0.0)
    D = np.zeros((1, 1), dtype=np.int64)
    fn = _heun_fn(params, _DT, 1, D)
    rng = np.random.default_rng(_SEED + 3)

    for _ in range(5):
        x = rng.standard_normal(6)
        u = float(rng.uniform(-0.5, 0.5))
        k_val = float(rng.uniform(0.0, 1.0))
        w = float(rng.uniform(0.0, 1.0))

        s_y = sigmoid_jit(x[1] - x[2], params.e0, params.v0, params.r)
        coupling = k_val * w * s_y
        x_numba = heun_step(x, u, params, _DT, xi=0.0, coupling=coupling)
        x_casadi = _np2(fn(x.reshape(6, 1), u, k_val, np.array([[w]]), _a_row(params, 1))).flatten()
        np.testing.assert_allclose(x_casadi, x_numba, rtol=1e-10, atol=1e-12)


def test_heun_step_matches_estimator_fx() -> None:
    """CasADi Heun step matches the dynamic slice of ``_fx_step_jit`` (N=2, zero delays).

    Coupling is computed internally by ``ca_heun_step`` from (K, W, D), so this is an
    end-to-end check against the estimator's bundled coupling + integration.
    """
    n_nodes = 2
    params = JansenRitParams()
    D = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    fn = _heun_fn(params, _DT, n_nodes, D)

    rng = np.random.default_rng(_SEED + 4)
    x_2d = rng.standard_normal((6, n_nodes))
    x_flat = x_2d.flatten()  # C-order

    K = 0.5
    W = np.array([[0.0, 0.8], [0.3, 0.0]])  # zero diagonal

    # Augmented state vector: [x_dyn(12), K(1), W.flat(4)]
    x_aug = np.zeros(6 * n_nodes + 1 + n_nodes**2)
    x_aug[: 6 * n_nodes] = x_flat
    x_aug[6 * n_nodes] = K
    x_aug[6 * n_nodes + 1 :] = W.flatten()

    u_node = rng.uniform(-0.1, 0.1, size=n_nodes)
    params_tuple = params.to_numba_tuple(n_nodes)
    history = np.zeros((1, n_nodes), dtype=np.float64)  # max_history_len=1, unused at zero delay

    x_aug_next = _fx_step_jit(x_aug, _DT, u_node, 0, 1, D, history, params_tuple, -1, -1)
    want = x_aug_next[: 6 * n_nodes]

    got = _np2(fn(x_2d, u_node.reshape(1, n_nodes), K, W, _a_row(params, n_nodes))).flatten()
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_casadi_trajectory_matches_simulate_network() -> None:
    """1000-step CasADi trajectory equals ``simulate_network`` with sigma=0 (single node)."""
    n_steps = 1000
    params = JansenRitParams(sigma=0.0)
    D = np.zeros((1, 1), dtype=np.int64)
    fn = _heun_fn(params, _DT, 1, D)

    _, x_ref = simulate_network(params=params, connectome=None, duration=n_steps * _DT, dt=_DT, seed=_SEED)

    x = x_ref[:, 0, 0].copy()  # initial state
    a_row = _a_row(params, 1)
    no_coupling = np.array([[0.0]])
    for _ in range(n_steps):
        x = _np2(fn(x.reshape(6, 1), 0.0, 0.0, no_coupling, a_row)).flatten()

    np.testing.assert_allclose(x, x_ref[:, 0, n_steps], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("a_gain", "expect_oscillation"), [(3.25, False), (3.6, True)])
def test_casadi_fixed_point_and_limit_cycle(a_gain: float, *, expect_oscillation: bool) -> None:
    """A=3.25 is a fixed point; A=3.6 produces a limit cycle (ptp > 5 mV)."""
    params = JansenRitParams(A=a_gain, sigma=0.0)
    D = np.zeros((1, 1), dtype=np.int64)
    fn = _heun_fn(params, _DT, 1, D)

    duration = 8.0
    transient_steps = round(3.0 / _DT)
    n_steps = round(duration / _DT)

    x = np.zeros(6)
    traj = np.empty((n_steps + 1, 6))
    traj[0] = x
    a_row = np.array([[a_gain]])
    no_coupling = np.array([[0.0]])
    for k in range(n_steps):
        x = _np2(fn(x.reshape(6, 1), 0.0, 0.0, no_coupling, a_row)).flatten()
        traj[k + 1] = x

    y = traj[transient_steps:, 1] - traj[transient_steps:, 2]  # x2 - x3
    ptp = np.ptp(y)

    if expect_oscillation:
        assert ptp > 5.0, f"Expected limit cycle at A={a_gain}, ptp={ptp:.3f}"
    else:
        assert ptp < 0.05, f"Expected fixed point at A={a_gain}, ptp={ptp:.3f}"
