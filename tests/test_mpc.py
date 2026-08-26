from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting
from yaml import safe_load

from neuro.control.mpc import (
    TrajOptMPCController,
    TrajOptMPCLog,
    build_waveform_problem,
)
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_SEED = 7

# Pinned parity values from the incumbent CasADi MPCController at fd0d244 (solver="ipopt"), run
# on the depth-0 checkpoint and the _SEED + 5 measurement trajectory below, with Kirchhoff
# applied unconditionally as the incumbent always does.
_WAVEFORM_PARITY_CONTROLS = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.5000000097735464, 0.5000000097735464],
        [-0.2197725016350544, 0.2197725016350544],
        [-0.5000000099748614, 0.5000000099748614],
        [-0.5000000099693410, 0.5000000099693410],
        [-0.5000000099527547, 0.5000000099527547],
    ]
)
_WAVEFORM_PARITY_COSTS = [
    0.0,
    0.0,
    0.0,
    1.4970273984042963,
    1.6780944598038890,
    1.8600552355429352,
    1.7721805744284902,
    2.4054735588528806,
]


def _full_parity_solver() -> Ipopt:
    """The general Ipopt transcription that carries the Kirchhoff linear equality."""
    return Ipopt(
        options={
            "hessian_approximation": "limited-memory",
            "print_level": 0,
            "max_iter": 500,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 5,
            "acceptable_constr_viol_tol": 1e-4,
        }
    )


def _drive_golden(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> tuple[FloatArray, list[float]]:
    """Feed the fixed golden trajectory through ``update``, returning controls and reported costs."""
    rng = np.random.default_rng(_SEED + 5)
    controls = []
    costs = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        controls.append(np.atleast_1d(np.asarray(u, dtype=np.float64)))
        costs.append(log.cost)
    return np.array(controls), costs


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs, drawn uniformly from ``+-1/sqrt(fan_in)``."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def _build_checkpoint(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
    depth: int = 2,
) -> Path:
    """Save a tiny synthetic MLP checkpoint and return its suffix-less stem."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=depth,
        activation="relu",
        dt=0.01,
        # The golden control/cost values were pinned from the incumbent CasADi controller on the
        # plain (non-residual) semantics, so the parity artifacts keep the skip off; the residual
        # path is pinned separately by the cross-side parity tests.
        residual=False,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, [in_size, *([5] * depth), n_channels]), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "art"
    model.save(path)
    return path


def _drive(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, TrajOptMPCLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 4)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_absorb_is_ready_initial_state(tmp_path: Path) -> None:
    """The jax model's priming seam holds NaN until ``n_y`` samples are absorbed, then is ready."""
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    artifact = _build_checkpoint(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    adapter = WaveformMLPModel.load(artifact)

    state = adapter.initial_state()
    assert np.isnan(state[: n_y * n_channels]).all()
    assert not adapter.is_ready(state)

    rng = np.random.default_rng(_SEED + 3)
    y_seq = rng.standard_normal((n_y, n_channels))
    u_seq = rng.standard_normal((n_y, n_controls))
    for t in range(n_y):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == (t == n_y - 1)
    assert adapter.is_ready(state)
    assert not np.isnan(state[: n_y * n_channels]).any()


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is still NaN-padded, the controller holds off and emits zeros."""
    n_y = 4
    artifact = _build_checkpoint(tmp_path, n_y=n_y)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    results = _drive(controller, n_steps=n_y, n_channels=controller.model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(controller.model.m))

    assert not results[-1][1].warmup


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite ``(n_controls,)`` control within the box bounds."""
    u_max = 0.5
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=u_max, w_y=1.0, w_u=0.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    u, _ = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)[-1]
    assert u.shape == (controller.model.m,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_pure_effort_cost_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the cost is sum||u||^2, whose unconstrained minimizer is u=0."""
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=1.0, w_y=0.0, w_u=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    u, log = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(controller.model.m), atol=1e-4)


def test_from_config_dispatches_problem_factory(tmp_path: Path) -> None:
    """from_config routes the problem through the ``{class_path, ...}`` factory pattern."""
    artifact = _build_checkpoint(tmp_path, horizon=5)
    controller = TrajOptMPCController.from_config(
        {
            "dt": 0.01,
            "problem": {
                "class_path": "neuro.control.mpc.build_waveform_problem",
                "artifact": str(artifact),
                "horizon": 5,
                "u_max": 0.5,
                "w_y": 1.0,
            },
        }
    )
    assert controller.dt == 0.01
    assert controller.problem.N == 6  # horizon + 1 knot points
    assert controller.model.m == 2


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """A u_max length that is neither 1 nor n_controls is rejected by the box builder."""
    artifact = _build_checkpoint(tmp_path, n_controls=2)
    with pytest.raises(ValueError, match="could not be broadcast"):
        build_waveform_problem(artifact, horizon=3, u_max=[1.0, 2.0, 3.0])


def test_controller_keeps_absorbed_state_private(tmp_path: Path) -> None:
    """``update`` persists the absorbed state and ``u_last`` as its own attributes.

    ``state.x0`` after ``shift()`` is the prior solve's second knot -- a model prediction, not
    the measurement-corrected state -- so the controller must not be reading its persistent
    state back from the post-solve trajectory.
    """
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    n_z = controller.model.n_y * controller.model.n_channels
    results = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert not np.isnan(controller._state[:n_z]).any()  # noqa: SLF001 -- the test inspects the absorbed state it owns
    np.testing.assert_array_equal(controller._u_last, u_last)  # noqa: SLF001 -- the test verifies the private u_last persistence
    seed = controller.state.x0
    assert not np.array_equal(np.asarray(seed), controller._state)  # noqa: SLF001 -- absorbed state vs post-solve seed


def test_reproduces_mpc_controller_control_sequence(tmp_path: Path) -> None:
    """The trajopt controller reproduces the incumbent CasADi control sequence and reported cost.

    The golden values are pinned from the incumbent ``MPCController`` (``solver="ipopt"``) at
    fd0d244 on this same depth-0 (linear, hence convex) checkpoint and measurement trajectory.
    Both the controls and the per-step reported cost must match: the cost assertion is what
    catches the absorbed-measurement term the incumbent graph never scores.
    """
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, w_u=0.0, kirchhoff=True)
    controller = TrajOptMPCController(dt=0.01, problem=problem, solver=_full_parity_solver())

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_migrated_config_reproduces_incumbent_end_to_end(tmp_path: Path) -> None:
    """The migrated YAML, dispatched through ``from_config``, reproduces the incumbent sequence and cost.

    Loads ``mse02_psd_mpc.yaml`` and swaps in the synthetic checkpoint and the golden weights,
    keeping the config's ``kirchhoff: true``; the controller is then built through the config's
    class-path dispatch rather than direct instantiation.
    """
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    problem_cfg = {
        **controller_cfg["problem"],
        "artifact": str(_build_checkpoint(tmp_path, depth=0)),
        "horizon": 3,
        "u_max": 0.5,
        "w_y": 1.0,
        "w_u": 0.0,
    }
    controller = TrajOptMPCController.from_config(
        {"dt": controller_cfg["dt"], "problem": problem_cfg, "solver": _full_parity_solver()}
    )

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_migrated_config_reproduces_incumbent_with_default_solver(tmp_path: Path) -> None:
    """A migrated ``kirchhoff: true`` config runs through ``from_config`` with no injected solver.

    The default selection must pick the general Ipopt transcription (the incumbent's
    ``solver="ipopt"``) when the Kirchhoff linear equality is present, so the config-driven
    closed loop reproduces the incumbent control sequence and reported cost.
    """
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    problem_cfg = {
        **controller_cfg["problem"],
        "artifact": str(_build_checkpoint(tmp_path, depth=0)),
        "horizon": 3,
        "u_max": 0.5,
        "w_y": 1.0,
        "w_u": 0.0,
    }
    controller = TrajOptMPCController.from_config({"dt": controller_cfg["dt"], "problem": problem_cfg})
    assert type(controller.solver) is Ipopt  # the general transcription, not SingleShooting

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_single_shooting_solver_rejected_when_kirchhoff(tmp_path: Path) -> None:
    """An injected single-shooting solver on a Kirchhoff problem fails at construction, not solve."""
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, kirchhoff=True)
    with pytest.raises(ValueError, match="SingleShooting supports only ControlBound and GoalConstraint"):
        TrajOptMPCController(
            dt=0.01,
            problem=problem,
            solver=SingleShooting(solver=Ipopt(options={"print_level": 0})),
        )
