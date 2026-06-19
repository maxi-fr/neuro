# ruff: noqa: N806
"""Verify the CasADi Jansen-Rit implementation against the numba/NumPy reference."""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from neuro.estimator import _fx_step_jit, _hx_step_jit
from neuro.jansen_rit import (
    JansenRitParams,
    _jr_rhs_jit,
    heun_step,
    sigmoid_jit,
    simulate_network,
)
from neuro.jansen_rit import (
    project_control as np_project_control,
)
from neuro.jansen_rit_casadi import (
    JRSymbolicParams,
    f_rhs,
    get_network_coupling,
    measure,
    project_control,
    rollout,
    sigmoid,
)
from neuro.jansen_rit_casadi import (
    heun_step as ca_heun_step,
)

_DT = 1e-4
_SEED = 42


def _np2(result: object) -> np.ndarray:
    return np.array(ca.DM(result))


def _build_test_params(
    n_nodes: int,
    base: JansenRitParams | None = None,
    k_val: float = 0.0,
    w_weights: np.ndarray | None = None,
    delay_steps: np.ndarray | None = None,
) -> JRSymbolicParams:
    if base is None:
        base = JansenRitParams()
    if w_weights is None:
        w_weights = np.zeros((n_nodes, n_nodes))
    if delay_steps is None:
        delay_steps = np.zeros((n_nodes, n_nodes))

    return JRSymbolicParams(
        A=np.broadcast_to(np.asarray(base.A), (1, n_nodes)),
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
        mean_input=np.broadcast_to(np.asarray(base.mean_input), (1, n_nodes)),
        sigma=base.sigma,
        eeg_gain=np.ones((1, n_nodes)),
        K=k_val,
        w_weights=w_weights,
        delay_steps=delay_steps,
    )


def test_sigmoid_matches_numba() -> None:
    params_base = JansenRitParams()
    params = _build_test_params(1, params_base)
    v = ca.SX.sym("v")
    fn = ca.Function("s", [v], [sigmoid(v, params)])
    for val in np.linspace(-20.0, 20.0, 11):
        got = _np2(fn(val)).item()
        want = sigmoid_jit(val, params_base.e0, params_base.v0, params_base.r)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_rhs_single_node_matches_numba() -> None:
    params = _build_test_params(1)
    X = ca.SX.sym("X", 6, 1)
    fn = ca.Function("rhs", [X], [f_rhs(X, ca.SX(0.0), 0.0, params)])

    params_tuple = JansenRitParams().to_numba_tuple(1)
    rng = np.random.default_rng(_SEED)

    for _ in range(5):
        x = rng.standard_normal(6)
        dx_numba = _jr_rhs_jit(x, params_tuple, 0.0, 0.0)
        dx_casadi = _np2(fn(x.reshape(6, 1))).flatten()
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_multi_node_matches_numba() -> None:
    n_nodes = 2
    base = JansenRitParams(A=np.array([3.25, 3.6]))
    params = _build_test_params(n_nodes, base=base)

    X = ca.SX.sym("X", 6, n_nodes)
    coupling = ca.SX.sym("coupling", 1, n_nodes)
    u = ca.SX.sym("u", 1, n_nodes)
    fn = ca.Function("rhs", [X, coupling, u], [f_rhs(X, coupling, u, params)])

    params_tuple = base.to_numba_tuple(n_nodes)
    rng = np.random.default_rng(_SEED + 2)

    for _ in range(5):
        x_2d = rng.standard_normal((6, n_nodes))
        c_val = rng.uniform(-1.0, 1.0, size=n_nodes)
        u_val = rng.uniform(-0.5, 0.5, size=n_nodes)

        dx_numba = _jr_rhs_jit(x_2d, params_tuple, c_val, u_val)
        dx_casadi = _np2(fn(x_2d, c_val.reshape(1, n_nodes), u_val.reshape(1, n_nodes)))
        np.testing.assert_allclose(dx_casadi, dx_numba, rtol=1e-10, atol=1e-12)


def test_coupling_with_delays_matches_reference() -> None:
    n_nodes = 2
    base = JansenRitParams()
    rng = np.random.default_rng(_SEED + 6)
    D = np.array([[0, 2], [1, 0]], dtype=np.int64)
    W = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    k_val = 0.7

    params = _build_test_params(n_nodes, base=base, k_val=k_val, w_weights=W, delay_steps=D)

    max_hist = int(D.max()) + 1
    hist_syms = [ca.SX.sym(f"X{t}", 6, n_nodes) for t in range(max_hist)]
    fn = ca.Function("c", hist_syms, [get_network_coupling(hist_syms, params)])

    hist = [rng.standard_normal((6, n_nodes)) for _ in range(max_hist)]
    got = _np2(fn(*hist)).flatten()

    want = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            past = hist[D[i, j]]
            s = sigmoid_jit(past[1, j] - past[2, j], base.e0, base.v0, base.r)
            want[i] += k_val * W[i, j] * s
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_heun_step_single_node_matches_jansen_rit() -> None:
    base = JansenRitParams(sigma=0.0)
    rng = np.random.default_rng(_SEED + 3)
    k_val = float(rng.uniform(0.0, 1.0))
    w_val = float(rng.uniform(0.0, 1.0))
    D = np.zeros((1, 1), dtype=np.int64)
    W = np.array([[w_val]])

    params = _build_test_params(1, base=base, k_val=k_val, w_weights=W, delay_steps=D)

    X = ca.SX.sym("X", 6, 1)
    u = ca.SX.sym("u", 1, 1)
    fn = ca.Function("heun", [X, u], [ca_heun_step([X], u, params, _DT)])

    for _ in range(5):
        x = rng.standard_normal(6)
        u_val = float(rng.uniform(-0.5, 0.5))

        s_y = sigmoid_jit(x[1] - x[2], base.e0, base.v0, base.r)
        coupling = k_val * w_val * s_y
        x_numba = heun_step(x, u_val, base, _DT, xi=0.0, coupling=coupling)
        x_casadi = _np2(fn(x.reshape(6, 1), u_val)).flatten()
        np.testing.assert_allclose(x_casadi, x_numba, rtol=1e-10, atol=1e-12)


def test_measure_matches_hx_step() -> None:
    n_nodes = 3
    n_channels = 2
    rng = np.random.default_rng(_SEED + 4)
    x = rng.standard_normal((6, n_nodes))

    X = ca.SX.sym("X", 6, n_nodes)
    gain_sym = ca.SX.sym("gain", n_channels, n_nodes)
    fn = ca.Function("meas", [X, gain_sym], [measure(X, gain_sym)])

    gain_val = rng.uniform(-1, 1, (n_channels, n_nodes))
    y_casadi = _np2(fn(x, gain_val)).flatten()

    x_aug = np.zeros(6 * n_nodes + 1 + n_nodes**2)
    x_aug[: 6 * n_nodes] = x.flatten()
    y_numba = _hx_step_jit(x_aug, gain_val, np.arange(n_channels), n_nodes, estimate_eeg_gains=False)
    np.testing.assert_allclose(y_casadi, y_numba, rtol=1e-10, atol=1e-12)


def test_project_control_matches_numpy() -> None:
    n_nodes = 3
    n_elec = 2
    rng = np.random.default_rng(_SEED + 7)
    gamma = rng.uniform(0, 1, (n_elec, n_nodes))
    u_val = rng.uniform(-1, 1, (n_elec,))

    u_sym = ca.SX.sym("u", n_elec, 1)
    fn = ca.Function("proj", [u_sym], [project_control(u_sym, gamma)])

    u_casadi = _np2(fn(u_val)).flatten()
    u_numba = np_project_control(u_val, gamma, n_elec)
    np.testing.assert_allclose(u_casadi, u_numba, rtol=1e-10, atol=1e-12)


def test_symbolic_theta_step_matches_fx_step() -> None:
    n_nodes = 2
    base = JansenRitParams()
    rng = np.random.default_rng(_SEED + 9)
    D = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    K_val = 0.5
    W_val = np.array([[0.0, 0.8], [0.3, 0.0]])

    params = _build_test_params(
        n_nodes, base, k_val=ca.SX.sym("K"), w_weights=ca.SX.sym("W", n_nodes, n_nodes), delay_steps=D
    )

    X = ca.SX.sym("X", 6, n_nodes)
    u = ca.SX.sym("u", 1, n_nodes)
    fn = ca.Function("heun_sym", [X, u, params.K, params.w_weights], [ca_heun_step([X], u, params, _DT)])

    x_2d = rng.standard_normal((6, n_nodes))
    u_val = rng.uniform(-0.1, 0.1, (1, n_nodes))

    got = _np2(fn(x_2d, u_val, K_val, W_val)).flatten()

    x_aug = np.zeros(6 * n_nodes + 1 + n_nodes**2)
    x_aug[: 6 * n_nodes] = x_2d.flatten()
    x_aug[6 * n_nodes] = K_val
    x_aug[6 * n_nodes + 1 :] = W_val.flatten()

    history = np.zeros((1, n_nodes), dtype=np.float64)
    params_tuple = base.to_numba_tuple(n_nodes)

    x_aug_next = _fx_step_jit(x_aug, _DT, u_val.flatten(), 0, 1, D, history, params_tuple, -1, -1)
    want = x_aug_next[: 6 * n_nodes]

    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_rollout_matches_simulate_network() -> None:
    """Check rollout helper works over multiple steps."""
    n_steps = 10
    base = JansenRitParams(sigma=0.0)
    D = np.zeros((1, 1), dtype=np.int64)
    params = _build_test_params(1, base=base, delay_steps=D)

    x_init = np.zeros((6, 1))
    x0_history = [ca.SX(x_init)]
    controls = [ca.SX(0.0) for _ in range(n_steps)]

    X_seq, _ = rollout(x0_history, controls, params, _DT)

    fn = ca.Function("roll", [], [ca.horzcat(*X_seq)])
    got_traj = _np2(fn()["o0"])

    _, x_ref = simulate_network(params=base, connectome=None, duration=n_steps * _DT, dt=_DT, seed=_SEED)

    want_traj = x_ref[:, 0, 1:]

    np.testing.assert_allclose(got_traj, want_traj, rtol=1e-10, atol=1e-12)
