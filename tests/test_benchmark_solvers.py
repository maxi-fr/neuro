from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from test_mpc import _build_checkpoint, _build_observable_checkpoint
from trajopt.mpc import MPC
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.benchmark import (
    format_closed_loop_table,
    format_open_loop_table,
    get_benchmark_solver,
    run_observable_benchmark,
    run_waveform_benchmark,
)
from neuro.control.mpc import (
    CanonicalDuals,
    NullspaceReducedModel,
    build_waveform_problem,
    kirchhoff_basis,
)
from neuro.predictor.inference import InferencePredictor, WaveformMLPModel

if TYPE_CHECKING:
    from pathlib import Path


def test_get_benchmark_solver_instantiation() -> None:
    """get_benchmark_solver creates configured solver instances across all names."""
    assert type(get_benchmark_solver("single_shooting")) is SingleShooting
    assert type(get_benchmark_solver("ipopt")) is Ipopt
    assert type(get_benchmark_solver("altro")) is ALTRO
    assert type(get_benchmark_solver("boxqp")) is BoxQP
    assert type(get_benchmark_solver("osqp")) is OSQP

    with pytest.raises(ValueError, match="Unknown solver name"):
        get_benchmark_solver("invalid_solver_name")


def test_osqp_exact_linear_predictor_optimality(tmp_path: Path) -> None:
    """On a linear predictor with quadratic cost, OSQP matches IPOPT Single Shooting to high precision."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    problem = build_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True)
    model = problem.model
    assert isinstance(model, InferencePredictor)

    rng = np.random.default_rng(123)
    x0 = rng.standard_normal(model.n)
    s_ipopt = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    s_osqp = OSQP(options={"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000})

    mpc_ipopt = MPC(problem, s_ipopt, x0=jnp.asarray(x0))
    mpc_osqp = MPC(problem, s_osqp, x0=jnp.asarray(x0))

    assert mpc_ipopt.solve().success
    assert mpc_osqp.solve().success

    np.testing.assert_allclose(mpc_ipopt.controls[0], mpc_osqp.controls[0], atol=1e-4)
    np.testing.assert_allclose(float(mpc_ipopt.cost()), float(mpc_osqp.cost()), rtol=1e-4)


def test_nullspace_reduced_model_methods(tmp_path: Path) -> None:
    """NullspaceReducedModel wraps dynamics, absorb, free_run, and initial_state over Z."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=3)
    base = WaveformMLPModel.load(art)
    reduced = NullspaceReducedModel(base)

    assert reduced.m == base.m - 1
    assert reduced.n == base.n
    assert reduced.n_controls == base.m - 1

    x0 = reduced.initial_state()
    assert np.isnan(x0[: base.n_y * base.n_channels]).all()
    assert not reduced.is_ready(x0)

    y_meas = np.array([0.1, -0.2])
    v = np.array([0.3, -0.4])
    u_full = np.asarray(reduced.basis) @ v
    x_primed = reduced.absorb(x0, y_meas, v)
    assert not np.isnan(x_primed[-base.n_u * base.n_controls :]).any()
    np.testing.assert_allclose(x_primed, base.absorb(x0, y_meas, u_full))

    # free_run and discrete_dynamics must agree with the base model fed the expanded currents,
    # since that equivalence is the only thing making the reduced OCP the same problem.
    x = jnp.asarray(np.random.default_rng(3).standard_normal(base.n))
    np.testing.assert_allclose(
        np.asarray(reduced.discrete_dynamics(x, jnp.asarray(v), 0.0, base.dt)),
        np.asarray(base.discrete_dynamics(x, jnp.asarray(u_full), 0.0, base.dt)),
        atol=1e-12,
    )

    rng = np.random.default_rng(4)
    y_h = rng.standard_normal((2, base.n_y, base.n_channels))
    u_h = rng.standard_normal((2, base.n_u, base.m - 1))
    u_f = rng.standard_normal((2, 3, base.m - 1))
    Z_T = np.asarray(reduced.basis).T
    np.testing.assert_allclose(
        np.asarray(reduced.free_run(y_h, u_h, u_f)),
        np.asarray(base.free_run(y_h, u_h @ Z_T, u_f @ Z_T)),
        atol=1e-12,
    )


def test_run_waveform_benchmark_integration(tmp_path: Path) -> None:
    """run_waveform_benchmark runs open-loop and closed-loop comparisons across solvers."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=2)
    open_comp, closed_comp = run_waveform_benchmark(
        art,
        horizon=3,
        u_max=0.5,
        n_repeats=2,
        num_steps=3,
    )
    # Rows are asserted by label, not by count: a solver the formulation defeats is dropped from
    # the table rather than taking it down, so the set present is the outcome under test.
    expected = {"SingleShooting(Ipopt)", "Ipopt(MultipleShooting)", "ALTRO", "OSQP"}
    assert {row.solver for row in open_comp.rows} <= expected
    assert {row.solver for row in closed_comp.rows} <= expected
    assert {"SingleShooting(Ipopt)", "ALTRO"} <= {row.solver for row in open_comp.rows}

    open_table = format_open_loop_table(open_comp)
    closed_table = format_closed_loop_table(closed_comp)
    assert "Open-Loop Solver Comparison" in open_table
    assert "Closed-Loop Receding Horizon MPC Comparison" in closed_table


def test_run_observable_benchmark_integration(tmp_path: Path) -> None:
    """run_observable_benchmark runs open-loop and closed-loop comparisons on Observable OCP."""
    art, geom = _build_observable_checkpoint(tmp_path, n_y=2, n_u=2, horizon=3, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)
    env_path = tmp_path / "obs_env.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((2, n_values), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )

    open_comp, closed_comp = run_observable_benchmark(
        art,
        env_path,
        horizon=3,
        u_max=0.5,
        w_u=1.0,
        w_hinge=2.0,
        n_repeats=2,
        num_steps=3,
    )
    expected = {"SingleShooting(Ipopt)", "Ipopt(MultipleShooting)", "ALTRO", "OSQP"}
    assert {row.solver for row in open_comp.rows} <= expected
    assert {row.solver for row in closed_comp.rows} <= expected
    assert {"SingleShooting(Ipopt)", "ALTRO"} <= {row.solver for row in open_comp.rows}


def test_kirchhoff_basis_spans_the_nullspace() -> None:
    """kirchhoff_basis returns an (m, m-1) Z whose image is exactly the sum-to-zero subspace."""
    rng = np.random.default_rng(1)
    for m in (2, 3, 5):
        Z = np.asarray(kirchhoff_basis(m))
        assert Z.shape == (m, m - 1)
        assert np.linalg.matrix_rank(Z) == m - 1
        u = rng.standard_normal((7, m - 1)) @ Z.T
        assert np.abs(u.sum(axis=-1)).max() < 1e-12

    with pytest.raises(ValueError, match="at least 2 electrodes"):
        kirchhoff_basis(1)


def test_nullspace_reduction_matches_hard_equality(tmp_path: Path) -> None:
    """The reduction reaches the hard-equality optimum with Kirchhoff exact rather than to a tolerance."""
    art = _build_checkpoint(tmp_path, depth=1, n_y=4, n_u=3, horizon=6, n_channels=4, n_controls=3)
    # u_max is tight enough that the per-electrode limit is active at the optimum: with it slack,
    # any Z-shaped polytope passes, so an active bound is what actually tests the reduced rows.
    u_max = 0.02
    hard = build_waveform_problem(art, horizon=6, u_max=u_max, w_y=1.0, w_u=0.0, kirchhoff=True)
    poly = build_waveform_problem(art, horizon=6, u_max=u_max, w_y=1.0, w_u=0.0, reduce_kirchhoff=True)

    assert poly.model.m == hard.model.m - 1
    # The exact current limit is a polytope in the reduced coordinates, carried as coupled rows.
    assert sum(poly.constraints.p) > 0

    x0 = jnp.asarray(np.random.default_rng(0).standard_normal(hard.model.n))
    costs = {}
    for label, problem in (("hard", hard), ("polytope", poly)):
        mpc = MPC(problem, CanonicalDuals(get_benchmark_solver("single_shooting")), x0=x0)
        res = mpc.solve()
        assert res.success
        costs[label] = float(mpc.cost())
        U = np.asarray(res.trajectory.U)
        if label != "hard":
            assert isinstance(problem.model, NullspaceReducedModel)
            U = U @ np.asarray(problem.model.basis).T
            assert np.abs(U.sum(axis=-1)).max() == 0.0
        assert np.abs(U).max() <= u_max + 1e-8
    assert np.abs(U).max() == pytest.approx(u_max, rel=1e-3)

    assert costs["polytope"] == pytest.approx(costs["hard"], rel=1e-4)


def test_nullspace_reduction_rejects_redundant_kirchhoff_options(tmp_path: Path) -> None:
    """Reduction satisfies Kirchhoff by construction, so the equality and L1 forms are refused."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=3)
    with pytest.raises(ValueError, match="drop kirchhoff"):
        build_waveform_problem(art, horizon=3, u_max=1.0, reduce_kirchhoff=True, kirchhoff=True)
    with pytest.raises(ValueError, match="does not support w_u_l1"):
        build_waveform_problem(art, horizon=3, u_max=1.0, reduce_kirchhoff=True, w_u_l1=1.0)
