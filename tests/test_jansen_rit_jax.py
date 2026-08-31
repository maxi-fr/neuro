from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.connectome import Connectome
from neuro.control.mpc import TrajOptMPCController
from neuro.jansen_rit import (
    JansenRitDynamics,
    JansenRitParams,
    _dynamics_history_coupling_jit,
    _heun_step_jit,
    _jr_rhs_jit,
    sigmoid_jit,
    simulate_network,
)
from neuro.predictor.jansen_rit import (
    JansenRitModel,
    JansenRitTrackingCost,
    build_jansen_rit_problem,
    eeg_jax,
    enable_x64,
    lfp_jax,
    project_control_jax,
    sigmoid_jax,
)
from neuro.stimulation.analytical import AnalyticalStim
from neuro.stimulation.base import _AnalyticalConfig

enable_x64()

_DT = 1e-4
_SEED = 42

_D3 = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int64)


def _toy_plant(n_nodes: int = 3, *, sigma: float = 0.0) -> tuple[JansenRitParams, Connectome]:
    """A small deterministic network with mixed delays for fast, TVB-free parity tests."""
    rng = np.random.default_rng(7)
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)
    d = _D3[:n_nodes, :n_nodes].copy()
    a_gain = np.array([3.25, 3.4, 3.6])[:n_nodes]
    params = JansenRitParams(A=a_gain, sigma=sigma)

    # Synthetic connectome
    conn = Connectome(
        K=0.6,
        weights=w,
        tract_lengths=(d * 5.0).astype(np.float64),
        centres=rng.uniform(-50.0, 50.0, (n_nodes, 3)),
        region_labels=np.array([f"reg_{i}" for i in range(n_nodes)], dtype=np.str_),
        hemispheres=np.zeros(n_nodes, dtype=bool),
        speed=50.0,
        delays=(d * (_DT * 1000.0)).astype(np.float64),
        region_index={f"reg_{i}": i for i in range(n_nodes)},
    )
    return params, conn


def test_x64_enabled() -> None:
    assert jnp.zeros(1).dtype == jnp.float64


def test_sigmoid_matches_numba() -> None:
    base = JansenRitParams()
    for val in np.linspace(-20.0, 20.0, 11):
        got = float(sigmoid_jax(val, base.e0, base.v0, base.r))
        want = float(sigmoid_jit(np.array([val]), base.e0, base.v0, base.r)[0])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_rhs_single_node_matches_numba() -> None:
    base = JansenRitParams()
    model = JansenRitModel.from_plant_components(base, dt=_DT, n_nodes=1)
    params_tuple = base.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED)

    for _ in range(5):
        x = rng.standard_normal((6, 1))
        dx_numba = _jr_rhs_jit(x, params_tuple, np.array([0.0]), np.array([0.0]))
        dx_jax = np.asarray(model.rhs(jnp.asarray(x), 0.0, 0.0))
        np.testing.assert_allclose(dx_jax, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_multi_node_matches_numba() -> None:
    n_nodes = 3
    base, conn = _toy_plant(n_nodes)
    model = JansenRitModel.from_plant_components(base, conn=conn, dt=_DT)
    params_tuple = base.to_numba_tuple(n_nodes)
    rng = np.random.default_rng(_SEED + 2)

    for _ in range(5):
        x_2d = rng.standard_normal((6, n_nodes))
        c_val = rng.uniform(-1.0, 1.0, size=n_nodes)
        u_val = rng.uniform(-0.5, 0.5, size=n_nodes)

        dx_numba = _jr_rhs_jit(x_2d, params_tuple, c_val, u_val)
        dx_jax = np.asarray(model.rhs(jnp.asarray(x_2d), jnp.asarray(c_val), jnp.asarray(u_val)))
        np.testing.assert_allclose(dx_jax, dx_numba, rtol=1e-10, atol=1e-12)


def test_coupling_with_delays_matches_numba() -> None:
    n_nodes = 3
    base, conn = _toy_plant(n_nodes)
    model = JansenRitModel.from_plant_components(base, conn=conn, dt=_DT)
    w = conn.weights
    k_val = conn.K
    length = int(_D3.max()) + 1

    rng = np.random.default_rng(_SEED + 6)
    history = rng.standard_normal((length, n_nodes))
    k = 5

    got = np.asarray(model.coupling_from_history(jnp.asarray(history), jnp.asarray(k)))

    want = _dynamics_history_coupling_jit(
        history.copy(),
        k,
        length,
        conn.delay_steps(_DT),
        w,
        k_val,
        history[k % length, :],
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_heun_det_step_matches_numba() -> None:
    base = JansenRitParams(sigma=0.0)
    model = JansenRitModel.from_plant_components(base, dt=_DT, n_nodes=1)
    rng = np.random.default_rng(_SEED + 3)

    for _ in range(5):
        x = rng.standard_normal((6, 1))
        u_val = float(rng.uniform(-0.5, 0.5))
        c_val = float(rng.uniform(-1.0, 1.0))

        x_numba = _heun_step_jit(
            x,
            np.array([u_val]),
            base.to_numba_tuple(1),
            _DT,
            np.zeros(1),
            np.array([c_val]),
        )
        x_jax = np.asarray(model.heun_step(jnp.asarray(x), u_val, c_val, _DT))
        np.testing.assert_allclose(x_jax, x_numba, rtol=1e-10, atol=1e-12)


def test_rollout_matches_simulate_network() -> None:
    n_steps = 20
    params, conn = _toy_plant(3, sigma=0.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)

    x0 = jnp.zeros((6, 3), dtype=jnp.float64)
    controls = jnp.zeros((n_steps, 3), dtype=jnp.float64)
    x_traj, _ = model.forward_rollout(x0, controls, _DT)

    _, x_ref = simulate_network(dyn=dyn, duration=n_steps * _DT)
    np.testing.assert_allclose(np.asarray(x_traj), x_ref[:, :, 1:], rtol=1e-10, atol=1e-12)


def test_lfp_eeg_project_control_match() -> None:
    n_nodes = 3
    rng = np.random.default_rng(_SEED + 4)
    x = rng.standard_normal((6, n_nodes))
    gain = rng.uniform(-1.0, 1.0, (2, n_nodes))

    np.testing.assert_allclose(np.asarray(lfp_jax(jnp.asarray(x))), x[1] - x[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(eeg_jax(jnp.asarray(x), jnp.asarray(gain))), gain @ (x[1] - x[2]), rtol=1e-10, atol=1e-12
    )

    n_elec = 2
    gamma = rng.uniform(0.0, 1.0, (n_elec, n_nodes))
    u = rng.uniform(-1.0, 1.0, n_elec)
    np.testing.assert_allclose(
        np.asarray(project_control_jax(jnp.asarray(u), jnp.asarray(gamma))),
        u @ gamma,
        rtol=1e-10,
        atol=1e-12,
    )


def test_grad_smoke() -> None:
    n_steps = 20
    params, conn = _toy_plant(3, sigma=0.0)
    model = JansenRitModel.from_plant_components(params, conn=conn, dt=_DT)
    x0 = jnp.zeros((6, 3))
    controls = jnp.zeros((n_steps, 3))

    def loss(m: JansenRitModel) -> jax.Array:
        _, y = m.forward_rollout(x0, controls, _DT)
        return jnp.sum(y**2)

    g = eqx.filter_grad(loss)(model)
    assert g.A.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g.A)))
    assert g.w_weights.shape == (3, 3)
    assert np.all(np.isfinite(np.asarray(g.w_weights)))
    assert np.isfinite(float(g.K))
    assert g.mean_input.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g.mean_input)))

    # Central finite-difference cross-check on A[0]
    eps = 1e-4
    m_plus = eqx.tree_at(lambda m: m.A, model, model.A.at[0].add(eps))
    m_minus = eqx.tree_at(lambda m: m.A, model, model.A.at[0].add(-eps))
    fd = (float(loss(m_plus)) - float(loss(m_minus))) / (2.0 * eps)
    np.testing.assert_allclose(float(g.A[0]), fd, rtol=1e-4, atol=1e-6)


def test_jansen_rit_model_adapter_discrete_step() -> None:
    params, conn = _toy_plant(3, sigma=0.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)
    model_dyn = JansenRitModel.from_dynamics(dyn)
    model_comp = JansenRitModel.from_plant_components(params, conn=conn, dt=_DT, x0=dyn.x)

    assert model.n == model_dyn.n == model_comp.n
    assert model.m == model_dyn.m == model_comp.m

    z0 = jnp.asarray(model.initial_state())
    u0 = jnp.zeros(model.m)

    z1 = model.discrete_dynamics(z0, u0, 0.0, _DT)
    assert z1.shape == (model.n,)
    assert np.all(np.isfinite(np.asarray(z1)))

    # Step live plant and compare next state
    x_plant_next = dyn.dynamics(0.0, dyn.x, np.zeros(dyn.n_controls))
    x_model_next, _, _ = model.unpack_state(z1)
    np.testing.assert_allclose(np.asarray(x_model_next), x_plant_next, rtol=1e-10, atol=1e-12)


def test_jansen_rit_mpc_problem_solve() -> None:
    params, conn = _toy_plant(2, sigma=0.0)
    cfg = _AnalyticalConfig(model="analytical", electrodes=["F3", "F4"])
    stim = AnalyticalStim(cfg, conn.centres)

    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, stim=stim, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)

    problem = build_jansen_rit_problem(
        model,
        horizon=5,
        u_max=2.0,
        w_y=1.0,
        w_u=0.1,
        kirchhoff=True,
    )

    solver = SingleShooting(solver=Ipopt(options={"print_level": 0, "max_iter": 50}))
    controller = TrajOptMPCController(dt=_DT, problem=problem, solver=solver)

    u_cmd, log = controller.update(0.0, np.zeros(dyn.x.shape), dyn.x)
    assert u_cmd.shape == (2,)
    assert np.all(np.isfinite(u_cmd))
    assert log.success
    # Kirchhoff check on u_cmd
    np.testing.assert_allclose(float(np.sum(u_cmd)), 0.0, atol=1e-6)


def test_jansen_rit_mpc_problem_from_config_dict() -> None:
    problem = build_jansen_rit_problem(
        horizon=5,
        u_max=2.0,
        dt=_DT,
        connectome={"speed": 50.0, "K": 0.6},
        params={"A": 3.25, "sigma": 0.0},
        stimulation={"model": "none"},
        w_y=1.0,
        w_u=0.1,
    )
    assert isinstance(problem.model, JansenRitModel)
    assert problem.model.n_nodes == 76
    assert problem.model.m == 1
    assert problem.N == 6


def test_jansen_rit_tracking_cost_eval() -> None:
    """Evaluate JansenRitTrackingCost matches manual quadratic tracking formulation."""
    n_nodes = 3
    leadfield = np.array([[1.0, -0.5, 0.2], [0.0, 1.0, -0.5]])
    params, conn = _toy_plant(n_nodes, sigma=0.0)
    model = JansenRitModel.from_plant_components(params, conn=conn, leadfield=leadfield, dt=_DT)

    w_y = 2.0
    horizon = 5
    cost = JansenRitTrackingCost(
        n=model.n,
        m=model.m,
        n_nodes=n_nodes,
        eeg_gain=model.eeg_gain,
        w_y=w_y,
        horizon=horizon,
    )

    rng = np.random.default_rng(_SEED + 10)
    x_ode = rng.standard_normal((6, n_nodes))
    hist = model.seed_history(jnp.asarray(x_ode))
    z = model.pack_state(jnp.asarray(x_ode), hist, 0.0)

    got = float(cost.evaluate(z))
    lfp = x_ode[1] - x_ode[2]
    y = leadfield @ lfp
    want = float((w_y / horizon) * np.sum(y**2))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_absorb_step_index_and_delay_history_advancement() -> None:
    """Step index k increments on State Absorption and circular history updates."""
    params, conn = _toy_plant(3, sigma=0.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)

    state = model.initial_state()
    _, _, k0 = model.unpack_state(jnp.asarray(state))
    np.testing.assert_allclose(float(k0), 0.0)

    for step in range(6):
        dyn.evaluate(step * _DT, np.zeros(dyn.n_controls))
        state = model.absorb(state, dyn.x, np.zeros(dyn.n_controls))
        x_ode, hist, k = model.unpack_state(jnp.asarray(state))
        np.testing.assert_allclose(float(k), float(step + 1))
        # Check ODE state is updated
        np.testing.assert_allclose(np.asarray(x_ode), dyn.x, rtol=1e-10, atol=1e-12)
        # Check history buffer at (step % max_history_len) matches S(y)
        s_expected = np.asarray(sigmoid_jax(lfp_jax(jnp.asarray(dyn.x)), model.e0, model.v0, model.r))
        slot = step % model.max_history_len
        np.testing.assert_allclose(np.asarray(hist)[slot], s_expected, rtol=1e-10, atol=1e-12)


def test_free_run_2d_and_3d_batched() -> None:
    """Free-run Rollout supports both 2D (T, m) and 3D batched (B, T, m) inputs."""
    params, conn = _toy_plant(3, sigma=0.0)
    leadfield = np.array([[1.0, 0.5, -0.5], [0.0, 1.0, 0.2]])
    model = JansenRitModel.from_plant_components(params, conn=conn, leadfield=leadfield, dt=_DT)

    t_steps = 10
    rng = np.random.default_rng(_SEED + 15)
    u_2d = rng.standard_normal((t_steps, model.m))

    y_2d = model.free_run(np.zeros(0), np.zeros(0), u_2d)
    assert y_2d.shape == (t_steps, 2)
    assert np.all(np.isfinite(np.asarray(y_2d)))

    # 3D batched
    batch_size = 4
    u_3d = rng.standard_normal((batch_size, t_steps, model.m))
    y_3d = model.free_run(np.zeros(0), np.zeros(0), u_3d)
    assert y_3d.shape == (batch_size, t_steps, 2)
    assert np.all(np.isfinite(np.asarray(y_3d)))

    # Verify batch consistency with individual 2D runs
    for b in range(batch_size):
        y_single = model.free_run(np.zeros(0), np.zeros(0), u_3d[b])
        np.testing.assert_allclose(np.asarray(y_3d[b]), np.asarray(y_single), rtol=1e-10, atol=1e-12)


def test_multi_step_closed_loop_solve_with_delayed_connectome() -> None:
    """Multi-step closed-loop MPC solve with delayed Connectome advances State Absorption."""
    params, conn = _toy_plant(2, sigma=0.0)
    cfg = _AnalyticalConfig(model="analytical", electrodes=["F3", "F4"])
    stim = AnalyticalStim(cfg, conn.centres)

    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, stim=stim, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)

    problem = build_jansen_rit_problem(
        model,
        horizon=5,
        u_max=2.0,
        w_y=1.0,
        w_u=0.05,
        w_y_terminal=2.0,
        w_u_l1=0.01,
        kirchhoff=True,
    )

    solver = SingleShooting(solver=Ipopt(options={"print_level": 0, "max_iter": 50}))
    controller = TrajOptMPCController(dt=_DT, problem=problem, solver=solver)

    for step in range(5):
        u_cmd, log = controller.update(step * _DT, np.zeros(dyn.x.shape), dyn.x)
        assert u_cmd.shape == (2,)
        assert np.all(np.isfinite(u_cmd))
        assert log.success
        np.testing.assert_allclose(float(np.sum(u_cmd)), 0.0, atol=1e-6)
        dyn.evaluate(step * _DT, u_cmd)

    # Controller internal state step index should be at 5
    _, _, k_final = model.unpack_state(jnp.asarray(controller._state))  # noqa: SLF001 -- inspect internal state
    np.testing.assert_allclose(float(k_final), 5.0)
