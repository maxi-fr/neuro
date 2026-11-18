from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from test_mpc import _build_checkpoint, _build_observable_checkpoint
from trajopt.mpc import MPC
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.config import StftGeometry
from neuro.control.mpc import TrajOptMPCController
from neuro.control.narx import (
    NarxShooting,
    _build_narx_sparsity,
    eval_g,
    eval_jac_g,
    narx_dimensions,
)
from neuro.control.problems import build_observable_problem, build_waveform_problem
from neuro.control.solvers import IPOPT_DEFAULTS, _build_solver
from neuro.predictor.cnn import ObservableCNNModel
from neuro.spectral import HealthyReference, ObservableEnvelope
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path


def _save_observable_cnn(path: Path, *, n_y: int = 3, n_u: int = 2, n_channels: int = 2, n_controls: int = 2) -> Path:
    """Create a small ObservableCNNModel checkpoint for NARX tests."""
    geom = StftGeometry(
        n_segment=10,
        n_hop=3,
        band_hz=(1.0, 25.0),
        window="hann",
        asymmetric_window=True,
        n_bin_pool=1,
        kernel="linear",
        kernel_width=1,
    )
    n_values = geom.n_values(50.0)
    n_outputs = n_channels * n_values
    model = ObservableCNNModel(
        n_y=n_y,
        n_u=n_u,
        horizon=4,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_outputs,
        hidden_size=4,
        depth=1,
        kernel_size=2,
        frequency_kernel_size=1,
        activation="softplus",
        dt=0.06,
        downsample=200,
        geometry=geom,
        y_std=Standardizer(center=np.zeros((n_channels, n_values)), scale=np.ones((n_channels, n_values))),
        u_std=Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls)),
        key=jax.random.PRNGKey(11),
    )
    stem = path / "obs_cnn"
    model.save(stem)
    return stem


def test_narx_dimensions_and_sparse_jacobian_match_dense(tmp_path: Path) -> None:
    """NARX counts match Chapter 5 and sparse Jacobian values match dense AD."""
    art = _build_checkpoint(tmp_path, depth=1, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=2)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    problem = build_waveform_problem(art, horizon=5, u_max=1.5, w_y=1.0, w_u=0.2, kirchhoff=True, reference=ref)
    n_vars, n_defects, n_user = narx_dimensions(problem)
    assert n_vars == 5 * (2 + 2)
    assert n_defects == 5 * 2
    assert n_user == 5

    sparsity = _build_narx_sparsity(problem, problem.model)  # ty: ignore[invalid-argument-type]
    rng = np.random.default_rng(7)
    x0 = jnp.asarray(rng.standard_normal(problem.model.n))
    v = jnp.asarray(rng.standard_normal(n_vars))

    jac_vals = np.asarray(eval_jac_g(problem, v, x0, 0.0, problem.dt, gather=sparsity.gather))
    sparse_jac = sp.coo_matrix((jac_vals, (sparsity.rows, sparsity.cols)), shape=(n_defects + n_user, n_vars)).toarray()
    dense_jac = np.asarray(jax.jacrev(lambda v_: eval_g(problem, v_, x0, 0.0, problem.dt))(v))
    np.testing.assert_allclose(sparse_jac, dense_jac, atol=1e-10, rtol=1e-10)


def test_narx_matches_single_and_multiple_shooting_waveform(tmp_path: Path) -> None:
    """NarxShooting matches SingleShooting and Ipopt optima on a nonlinear waveform problem."""
    art = _build_checkpoint(tmp_path, depth=1, n_y=3, n_u=2, horizon=6, n_channels=2, n_controls=2)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    problem = build_waveform_problem(art, horizon=6, u_max=1.5, w_y=1.0, w_u=0.5, kirchhoff=True, reference=ref)

    rng = np.random.default_rng(19)
    x0 = jnp.asarray(rng.standard_normal(problem.model.n))
    opts = {**IPOPT_DEFAULTS, "tol": 1e-6, "acceptable_tol": 1e-5}

    res_ss = MPC(problem, SingleShooting(solver=Ipopt(options=opts)), x0=x0).solve()
    res_ms = MPC(problem, Ipopt(options=opts), x0=x0).solve()
    res_nx = MPC(problem, NarxShooting(solver=Ipopt(options=opts)), x0=x0).solve()

    assert res_ss.success
    assert res_ms.success
    assert res_nx.success
    assert res_nx.constraint_violation < 1e-6
    np.testing.assert_allclose(res_nx.cost, res_ss.cost, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_nx.cost, res_ms.cost, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_nx.trajectory.U, res_ss.trajectory.U, atol=1e-3)
    np.testing.assert_allclose(res_nx.trajectory.X, res_ms.trajectory.X, atol=1e-3)


def test_narx_matches_single_shooting_observable_cnn(tmp_path: Path) -> None:
    """NarxShooting matches SingleShooting on an Observable CNN hinge problem."""
    art = _save_observable_cnn(tmp_path, n_y=3, n_u=2, n_channels=2, n_controls=2)
    geom = StftGeometry(
        n_segment=10,
        n_hop=3,
        band_hz=(1.0, 25.0),
        window="hann",
        asymmetric_window=True,
        n_bin_pool=1,
        kernel="linear",
        kernel_width=1,
    )
    n_values = geom.n_values(50.0)
    ref = HealthyReference(
        observable=ObservableEnvelope(power=np.zeros((2, n_values)), fs=50.0, geometry=geom),
    )
    problem = build_observable_problem(
        art, horizon=4, u_max=1.5, w_u=0.2, w_hinge=5.0, w_hinge_terminal=10.0, reference=ref, kirchhoff=True
    )

    rng = np.random.default_rng(23)
    x0 = jnp.asarray(rng.uniform(-0.5, 1.5, size=problem.model.n))
    opts = {**IPOPT_DEFAULTS, "tol": 1e-6, "acceptable_tol": 1e-5}

    res_ss = MPC(problem, SingleShooting(solver=Ipopt(options=opts)), x0=x0).solve()
    res_nx = MPC(problem, NarxShooting(solver=Ipopt(options=opts)), x0=x0).solve()

    assert res_ss.success
    assert res_nx.success
    assert res_nx.constraint_violation < 1e-6
    np.testing.assert_allclose(res_nx.cost, res_ss.cost, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_nx.trajectory.U, res_ss.trajectory.U, atol=1e-3)


def test_narx_closed_loop_controller_steps_with_warm_start(tmp_path: Path) -> None:
    """NarxShooting runs closed-loop updates in TrajOptMPCController via class_path."""
    art, geom = _build_observable_checkpoint(tmp_path, depth=0, n_y=2, n_u=2, horizon=4, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)
    ref = HealthyReference(
        observable=ObservableEnvelope(power=np.zeros((2, n_values)), fs=50.0, geometry=geom),
    )
    problem = build_observable_problem(art, horizon=4, u_max=2.0, w_u=0.1, w_hinge=5.0, reference=ref, kirchhoff=True)
    solver = _build_solver({"class_path": "neuro.control.narx.NarxShooting"}, problem)
    assert isinstance(solver, NarxShooting)

    dt = float(problem.model.dt)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    ctrl = TrajOptMPCController(dt=dt, problem=problem, solver=solver)
    rng = np.random.default_rng(5)
    u_prev = np.zeros(2)
    for k in range(5):
        y = rng.standard_normal((2, n_values))
        u_next, log = ctrl.update(k * dt, u_prev, y)
        if not log.warmup:
            assert log.success
            np.testing.assert_allclose(np.sum(u_next), 0.0, atol=1e-6)
        u_prev = u_next


def test_narx_validation_rejects_unsupported_configurations(tmp_path: Path) -> None:
    """NarxShooting rejects blocked controls, binary indices, and non-Ipopt solvers."""
    with pytest.raises(TypeError, match="supports only Ipopt"):
        NarxShooting(solver="not_ipopt")  # ty: ignore[invalid-argument-type]

    art = _build_checkpoint(tmp_path, depth=0, n_y=2, n_u=2, horizon=4, n_channels=2, n_controls=2)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    blocked = build_waveform_problem(art, horizon=4, control_block_steps=2, u_max=1.0, reference=ref)
    with pytest.raises(ValueError, match="control_block_steps=1"):
        narx_dimensions(blocked)
