from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt
from yaml import safe_load

from neuro.checkpoint import ObservableCheckpoint
from neuro.config import StftGeometry
from neuro.control.trajopt_mpc import (
    ObservableModel,
    TrajOptMPCController,
    TrajOptMPCLog,
    build_observable_problem,
)
from neuro.observable import control_means
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.checkpoint import ObservableCheckpoint
    from neuro.types import FloatArray

_SEED = 7
_N_EEG, _N_CONTROLS = 2, 2
_L, _R, _FS = 8, 4, 50.0
# float32 torch runtime vs float64 checkpoint weights: the reservoir recurrence accumulates
# rounding, but on these fixtures it stays inside ticket 01's waveform bar (measured ~2e-8).


def _drive(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, TrajOptMPCLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 4)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def _envelope_npz(tmp_path: Path, power: FloatArray, *, window: int = _L, hop: int = _R) -> Path:
    """Write a healthy-envelope npz in the layout ``scripts/build_healthy_psd.py`` produces."""
    path = tmp_path / "healthy.npz"
    np.savez(
        path,
        Pref=power,
        freqs=np.fft.rfftfreq(window, 1.0 / _FS),
        fs=_FS,
        L=window,
        R=hop,
        quantile=0.9,
        n_windows=10,
        plant_fingerprint="test",
    )
    return path


def test_observable_rollout_matches_torch_rollout(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """Chaining adapter Frame steps reproduces the torch module's rollout on the same trajectory.

    The module's ``rollout`` aggregates raw future controls into Frame means; the adapter steps
    one Frame per call, so the test aggregates the same way and compares the decoded Frames
    one-for-one.
    """
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, n_u=3, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    adapter = ObservableModel.from_checkpoint(artifact)
    module = StepwiseObservableMLP.load(artifact)

    rng = np.random.default_rng(_SEED + 5)
    horizon = 16
    y_ctx = rng.standard_normal((8, ckpt.n_channels))
    u_ctx = rng.standard_normal((8, ckpt.n_controls))
    u_future = rng.standard_normal((horizon, ckpt.n_controls))

    state = module.prime(y_ctx, u_ctx)
    want = module.rollout(state, u_future)
    u_bar = control_means(ckpt.geometry, horizon, ckpt.fs) @ u_future
    x = jnp.asarray(state)
    got = []
    for u in u_bar:
        x = adapter.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.0)
        got.append(np.asarray(adapter.output(x)))
    np.testing.assert_allclose(np.stack(got), want, rtol=1e-5, atol=1e-6)


def test_observable_absorb_is_ready_initial_state_match_torch_module(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The adapter's priming seam matches the torch module's shift-register discipline."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, n_u=3, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    adapter = ObservableModel.from_checkpoint(artifact)
    module = StepwiseObservableMLP.load(artifact)

    rng = np.random.default_rng(_SEED + 6)
    state = adapter.initial_state()
    module_state = module.initial_state()
    np.testing.assert_array_equal(state, module_state)
    assert not adapter.is_ready(state)
    assert not module.is_ready(module_state)

    y_seq = rng.standard_normal((4, ckpt.n_channels))
    u_seq = rng.standard_normal((4, ckpt.n_controls))
    for t in range(4):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        module_state = module.absorb(module_state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == module.is_ready(module_state)
    # The register part is float64 on both sides and matches exactly; the lifted Frame state
    # differs by the module's float32 lift, so the whole state compares to float32 tolerance.
    np.testing.assert_allclose(state, module_state, rtol=1e-5, atol=1e-6)
    assert adapter.is_ready(state)


def test_observable_problem_drives_controller(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The observable problem runs through the ticket-01/02 controller on the Frame grid."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))
    u_max = 2.0
    problem = build_observable_problem(artifact, horizon=16, u_max=u_max, w_u=0.01, w_psd=10.0, psd_ref=envelope)
    controller = TrajOptMPCController(dt=ckpt.dt, problem=problem)

    results = _drive(controller, n_steps=8, n_channels=ckpt.n_channels)
    for u, log in results[:3]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(ckpt.n_controls))
    for u, log in results[3:]:
        assert not log.warmup
        assert log.success
        assert u.shape == (ckpt.n_controls,)
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)


def test_observable_from_config_dispatches_factory(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """from_config routes the observable problem through the factory, defaulting no horizon."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))

    controller = TrajOptMPCController.from_config(
        {
            "dt": ckpt.dt,
            "problem": {
                "class_path": "neuro.control.trajopt_mpc.build_observable_problem",
                "artifact": str(artifact),
                "horizon": 16,
                "u_max": 2.0,
                "w_u": 1.0,
                "w_psd": 10.0,
                "psd_ref": str(envelope),
            },
        }
    )
    assert controller.dt == ckpt.dt
    assert isinstance(controller.model, ObservableModel)
    assert ckpt.n_frames(16) + 1 == controller.problem.N


@pytest.mark.parametrize(
    ("kind", "factory", "weights"),
    [
        ("mlp", "build_waveform_problem", {"w_y": 1.0, "w_u": 10.0}),
        (
            "observable",
            "build_observable_problem",
            {"w_y": 0.0, "w_u": 10.0, "w_psd": 1000.0, "psd_ref": None},
        ),
    ],
    ids=["waveform", "observable"],
)
def test_migrated_yaml_controller_block_dispatches_each_kind(
    tmp_path: Path,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
    kind: str,
    factory: str,
    weights: dict[str, object],
) -> None:
    """The checked-in ``mse02_psd_mpc.yaml`` controller block dispatches each predictor kind.

    The config carries the migrated layout: ``class_path`` at ``TrajOptMPCController`` and the
    cost-weight fields (``artifact``, ``horizon``, ``u_max``, ``w_y``, ``w_u``) nested under
    ``problem`` with the per-kind factory. Swapping in a synthetic checkpoint for each kind and
    adding the observable-only weights drives it through ``from_config`` directly.
    """
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    assert controller_cfg["class_path"] == "neuro.control.trajopt_mpc.TrajOptMPCController"
    assert controller_cfg["problem"]["class_path"] == "neuro.control.trajopt_mpc.build_waveform_problem"

    if kind == "mlp":
        artifact = _mlp_checkpoint(tmp_path)
    else:
        ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=50, n_y=4, dt=0.02)
        artifact = tmp_path / "observable"
        ckpt.save(artifact)
        weights["psd_ref"] = str(_envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3)))

    problem_cfg = {
        **controller_cfg["problem"],
        "class_path": f"neuro.control.trajopt_mpc.{factory}",
        "artifact": str(artifact),
        **weights,
    }
    controller = TrajOptMPCController.from_config({"dt": controller_cfg["dt"], "problem": problem_cfg})
    assert controller.dt == controller_cfg["dt"]
    u, log = controller.update(0.0, np.array([0.0]), np.zeros(controller.model.n_channels))
    assert u.shape == (controller.model.m,)
    assert log.warmup  # nothing absorbed yet


def _mlp_checkpoint(tmp_path: Path) -> Path:
    """Save a tiny synthetic depth-0 MLP checkpoint and return its suffix-less stem."""
    rng = np.random.default_rng(_SEED)
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
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
        horizon=3,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=0,
        activation="relu",
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, n_channels]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "mlp_art"
    model.save(path)
    return path


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs, drawn uniformly from ``+-1/sqrt(fan_in)``."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def test_observable_problem_rejects_w_y_and_requires_psd(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The observable path rejects ``w_y`` and requires ``w_psd`` + ``psd_ref``, like the incumbent."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))

    with pytest.raises(ValueError, match="w_y"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_y=1.0, w_psd=1.0, psd_ref=envelope)
    with pytest.raises(ValueError, match="w_psd > 0"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_u=1.0)
    with pytest.raises(ValueError, match="w_psd > 0"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_psd=1.0)
    with pytest.raises(ValueError, match=r"no .* frame"):
        build_observable_problem(artifact, horizon=4, u_max=2.0, w_psd=1.0, psd_ref=envelope)


def test_observable_problem_rejects_native_solver_with_hinge(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """A native expansion-only solver is rejected when the objective carries a whole-horizon hinge.

    The hinge returns ``0`` from ``evaluate``, so a native backend would silently solve a problem
    missing it; the controller raises instead. The transcription path still scores the hinge and
    is covered by the parity tests.
    """
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))
    problem = build_observable_problem(artifact, horizon=16, u_max=2.0, w_u=0.1, w_psd=10.0, psd_ref=envelope)
    with pytest.raises(ValueError, match="whole-horizon"):
        TrajOptMPCController(dt=ckpt.dt, problem=problem, solver=ALTRO())


def test_observable_config_defaults_to_general_ipopt_with_kirchhoff(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The migrated observable config with ``kirchhoff: true`` and no solver picks the general Ipopt.

    The ObservableRolloutHinge is whole-horizon, but the general Ipopt transcription is not a
    native expansion-only backend, so ``ensure_solver_supports_objective`` must not reject it;
    the default selection chooses it over single shooting because of the Kirchhoff equality.
    """
    with Path("configs/simulation/observable_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    assert controller_cfg["problem"]["kirchhoff"] is True

    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=75, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))

    problem_cfg = {
        **controller_cfg["problem"],
        "artifact": str(artifact),
        "psd_ref": str(envelope),
    }
    controller = TrajOptMPCController.from_config({"dt": controller_cfg["dt"], "problem": problem_cfg})
    assert type(controller.solver) is Ipopt
