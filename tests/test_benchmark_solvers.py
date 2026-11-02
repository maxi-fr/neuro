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
from neuro.control.problems import build_waveform_problem
from neuro.control.solvers import CanonicalDuals
from neuro.predictor.inference import InferencePredictor, WaveformMLPModel
from neuro.spectral import HealthyReference

if TYPE_CHECKING:
    from pathlib import Path


def test_get_benchmark_solver_instantiation() -> None:
    """get_benchmark_solver creates configured solver instances across all names with tuned Ipopt defaults."""
    ss = get_benchmark_solver("single_shooting")
    assert type(ss) is SingleShooting
    assert type(ss.solver) is Ipopt
    assert ss.solver.options["tol"] == 1e-3
    assert ss.solver.options["acceptable_tol"] == 1e-2
    assert ss.solver.options["acceptable_iter"] == 5

    ip = get_benchmark_solver("ipopt")
    assert type(ip) is Ipopt
    assert ip.options["tol"] == 1e-3
    assert ip.options["acceptable_tol"] == 1e-2
    assert ip.options["acceptable_iter"] == 5

    assert type(get_benchmark_solver("altro")) is ALTRO
    assert type(get_benchmark_solver("boxqp")) is BoxQP
    assert type(get_benchmark_solver("osqp")) is OSQP

    with pytest.raises(ValueError, match="Unknown solver name"):
        get_benchmark_solver("invalid_solver_name")


def test_osqp_exact_linear_predictor_optimality(tmp_path: Path) -> None:
    """On a linear Predictor with quadratic cost, OSQP matches IPOPT Single Shooting to high precision."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    problem = build_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True, reference=ref)
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


def test_run_waveform_benchmark_integration(tmp_path: Path) -> None:
    """run_waveform_benchmark runs open-loop and closed-loop comparisons across solvers."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=2)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    open_comp, closed_comp = run_waveform_benchmark(
        art,
        horizon=3,
        u_max=0.5,
        reference=ref,
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
        reference=HealthyReference.load(env_path),
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
