"""Pin the observable MPC: cost equivalence with the incumbent hinge, and the controller seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
from simulate.simulation import Simulation

from neuro.config import EegMsGeometry, StftGeometry
from neuro.control.nlp import MPCNlp, _observable_hinge_cost, _spectral_hinge_cost
from neuro.control.nonlinear_mpc import MPCController, MPCControllerLog
from neuro.observable import envelope_log_reference, log_observable
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.spectral import PsdEnvelope, compute_periodograms, hinge_penalty
from neuro.validation import ConfigConsistencyError, validate_simulation_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from neuro.observable import ObservableArtifact
    from neuro.types import FloatArray

_SEED = 41
_FS = 50.0
_L, _R = 8, 4


def _envelope_npz(tmp_path: Path, power: FloatArray, *, ms_power: FloatArray | None = None) -> Path:
    """Write a healthy-envelope npz in the layout ``scripts/build_healthy_psd.py`` produces."""
    path = tmp_path / "healthy.npz"
    arrays: dict[str, object] = {
        "Pref": power,
        "freqs": np.fft.rfftfreq(_L, 1.0 / _FS),
        "fs": _FS,
        "L": _L,
        "R": _R,
        "quantile": 0.9,
        "n_windows": 10,
        "plant_fingerprint": "test",
    }
    if ms_power is not None:
        arrays["Pref_ms"] = ms_power
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
    return path


def _drive(controller: MPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, MPCControllerLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 1)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


@pytest.mark.parametrize(("horizon", "length", "hop"), [(12, 8, 4), (16, 8, 4), (10, 10, 10), (20, 8, 6)])
def test_observable_hinge_is_the_same_functional_as_the_spectral_hinge(horizon: int, length: int, hop: int) -> None:
    """Fed the *true* log-spectrogram, the observable hinge equals the incumbent rollout hinge.

    Both must also equal the NumPy reference, so ``w_psd`` transfers between the two controllers and
    a comparison between them is a comparison of Predictors rather than of objectives.
    """
    n_channels = 3
    rng = np.random.default_rng(_SEED + 2)
    y_traj = rng.standard_normal((horizon, n_channels)) * 2.0

    windows = compute_periodograms(y_traj, fs=_FS, window=length, hop=hop)
    envelope = PsdEnvelope(power=np.median(windows, axis=0), fs=_FS, window=length, hop=hop)
    expected = hinge_penalty(windows[..., 1:], envelope.power[:, 1:])
    assert expected > 0.0, "the envelope must actually bite for the comparison to mean anything"

    geometry = StftGeometry(n_segment=length, n_hop=hop)
    l_true = log_observable(y_traj, geometry, _FS)
    l_flat = l_true.reshape(l_true.shape[0], -1).T  # (channels * values, frames)
    observable_cost = _observable_hinge_cost(ca.MX(l_flat), envelope_log_reference(envelope, geometry, _FS))

    y_nodes = [ca.MX(y_traj[t].reshape(-1, 1)) for t in range(horizon)]
    spectral_cost = _spectral_hinge_cost(y_nodes, envelope, horizon)

    assert float(ca.evalf(observable_cost)) == pytest.approx(expected, rel=1e-9)
    assert float(ca.evalf(spectral_cost)) == pytest.approx(expected, rel=1e-9)


def test_observable_hinge_is_exactly_zero_when_under_the_envelope() -> None:
    """One-sided: a forecast everywhere below the envelope costs exactly 0, not merely a little."""
    log_reference = np.full((3, 4), 5.0)
    l_hat = ca.MX(np.full((12, 2), -3.0))
    assert float(ca.evalf(_observable_hinge_cost(l_hat, log_reference))) == 0.0


def test_observable_nlp_rejects_a_stagewise_output_weight(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """``w_y`` has no meaning without per-step outputs, so it raises rather than being ignored."""
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=16)
    model = ObservableSymbolicModel(art)

    with pytest.raises(ValueError, match="w_y"):
        MPCNlp.build(
            model,
            horizon=16,
            shooting_depth=16,
            n_controls=model.n_controls,
            u_max=np.full(model.n_controls, 2.0),
            w_y=1.0,
            w_psd=1.0,
            log_reference=np.zeros((art.n_channels, art.n_values)),
        )


def test_observable_nlp_rejects_multiple_shooting(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """There is no per-sample state to introduce shooting roots on, so a short depth raises."""
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=16)
    model = ObservableSymbolicModel(art)

    with pytest.raises(ValueError, match="shooting_depth"):
        MPCNlp.build(
            model,
            horizon=16,
            shooting_depth=4,
            n_controls=model.n_controls,
            u_max=np.full(model.n_controls, 2.0),
            w_y=0.0,
            w_psd=1.0,
            log_reference=np.zeros((art.n_channels, art.n_values)),
        )


def test_observable_nlp_has_no_shooting_variables(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """The decision vector is the controls alone: no ``phi`` roots are constructed at all."""
    horizon = 16
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=horizon)
    model = ObservableSymbolicModel(art)

    mpc_nlp = MPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=horizon,
        n_controls=model.n_controls,
        u_max=np.full(model.n_controls, 2.0),
        w_y=0.0,
        w_u=1.0,
        w_psd=1.0,
        log_reference=np.zeros((art.n_channels, art.n_values)),
    )
    assert mpc_nlp.nlp["x"].numel() == horizon * model.n_controls
    assert mpc_nlp.nlp["g"].numel() == horizon  # the Kirchhoff equality only


def _controller(art: ObservableArtifact, envelope_path: Path, *, horizon: int = 16, **kwargs: object) -> MPCController:
    """An MPC driven by the observable model, wired to the healthy envelope on disk."""
    defaults = {"u_max": 2.0, "w_y": 0.0, "w_u": 0.01, "w_psd": 1.0}
    return MPCController(
        dt=art.dt,
        model=ObservableSymbolicModel(art),
        horizon=horizon,
        psd_ref=str(envelope_path),
        **{**defaults, **kwargs},  # ty: ignore[invalid-argument-type]
    )


def test_warmup_emits_zero_until_window_filled(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """While the EEG window is still NaN-padded, the MPC holds off and emits zeros."""
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4)
    controller = _controller(art, _envelope_npz(tmp_path, np.full((art.n_channels, _L // 2 + 1), 1e-3)))

    results = _drive(controller, n_steps=art.n_y, n_channels=art.n_channels)
    for u, log in results[: art.n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(art.n_controls))
    assert not results[-1][1].warmup


def test_solved_controls_respect_bounds_and_kirchhoff(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """Past warm-up the control is finite, inside the box, and sums to zero across electrodes."""
    u_max = 2.0
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4)
    # A biting envelope, so the solver has a reason to stimulate rather than sit at zero.
    envelope = _envelope_npz(tmp_path, np.full((art.n_channels, _L // 2 + 1), 1e-6))
    controller = _controller(art, envelope, u_max=u_max, w_u=0.0, w_psd=100.0)

    for u, log in _drive(controller, n_steps=8, n_channels=art.n_channels)[art.n_y :]:
        assert not log.warmup
        assert u.shape == (art.n_controls,)
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)
        assert abs(float(np.sum(u))) < 1e-6


def test_from_config_loads_an_observable_artifact_and_defaults_the_horizon(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """from_config dispatches on ``model_type`` and defaults the horizon to the artifact's."""
    art = make_observable_artifact(StftGeometry(n_segment=_L, n_hop=_R), horizon=16)
    art.save(tmp_path / "observable")
    envelope = _envelope_npz(tmp_path, np.full((art.n_channels, _L // 2 + 1), 1e-3))

    controller = MPCController.from_config(
        {
            "dt": art.dt,
            "artifact": str(tmp_path / "observable"),
            "u_max": 2.0,
            "w_y": 0.0,
            "w_u": 1.0,
            "w_psd": 10.0,
            "psd_ref": str(envelope),
        }
    )

    assert controller.horizon == 16
    assert controller.is_observable
    assert controller.psd_envelope is None
    assert controller.log_reference is not None
    assert controller.log_reference.shape == (art.n_channels, art.n_values)


def test_closed_loop_simulation_runs(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """The observable MPC closes the loop through the orchestrator and keeps controls in bounds."""
    n_channels, u_max = 3, 3.0
    art = make_observable_artifact(
        StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, n_u=3, n_channels=n_channels, dt=0.01
    )
    art.save(tmp_path / "observable")
    envelope_path = tmp_path / "healthy.npz"
    np.savez(
        envelope_path,
        Pref=np.full((n_channels, _L // 2 + 1), 1e-4),
        freqs=np.fft.rfftfreq(_L, art.dt),
        fs=1.0 / art.dt,
        L=_L,
        R=_R,
        quantile=0.9,
        n_windows=10,
        plant_fingerprint="test",
    )

    config = {
        "t_end": 0.05,
        "dynamics": {
            "class_path": "neuro.jansen_rit.JansenRitDynamics",
            "dt": 1e-4,
            "seed": 69,
            "connectome": {"speed": 50.0, "K": 0.5357},
            "stimulation": {"model": "analytical", "electrodes": ["CP5", "T7"], "spread": 20.0},
            "params": {"A": 3.25},
        },
        "reference": {"class_path": "simulate.reference.StepReference", "dt": 1e-4, "step_value": 0.0},
        "sensors": {
            "class_path": "simulate.sensor.GaussianSensor",
            "dt": 1e-4,
            "std_dev": 0.0,
            "measurement": {"class_path": "neuro.eeg.EEGMeasurement", "selected_channels": [0, 1, 2]},
        },
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.nonlinear_mpc.MPCController",
            "dt": art.dt,
            "artifact": str(tmp_path / "observable"),
            "horizon": 16,
            "u_max": u_max,
            "w_y": 0.0,
            "w_u": 0.01,
            "w_psd": 10.0,
            "psd_ref": str(envelope_path),
        },
    }

    sim = Simulation.from_config(config)
    sim.run()

    assert sim.logger is not None
    us = sim.logger.signal("controller", "u")
    assert us.shape[1] == art.n_controls
    assert np.all(np.abs(us) <= u_max + 1e-6)


def test_geometry_mismatch_between_artifact_and_envelope_raises(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """An artifact recording one Frame grid and an envelope measured on another is a hard error."""
    art = make_observable_artifact(StftGeometry(n_segment=_L + 2, n_hop=_R), horizon=16, n_channels=2)
    art.save(tmp_path / "observable")
    envelope = _envelope_npz(tmp_path, np.full((2, _L // 2 + 1), 1e-3))

    config = {
        "dynamics": {"dt": 1e-4},
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": 1e-4},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.nonlinear_mpc.MPCController",
            "dt": art.dt,
            "artifact": str(tmp_path / "observable"),
            "u_max": 2.0,
            "psd_ref": str(envelope),
        },
    }
    with pytest.raises(ConfigConsistencyError, match="segment length"):
        validate_simulation_config(config)


def test_eeg_ms_artifact_needs_the_mean_square_envelope(
    tmp_path: Path, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """An ``eeg_ms`` artifact pointed at an npz without ``Pref_ms`` fails loudly, not silently."""
    art = make_observable_artifact(EegMsGeometry(window_s=_L * 0.02, hop_s=_R * 0.02), horizon=16, n_channels=2)
    art.save(tmp_path / "observable")

    config = {
        "dynamics": {"dt": 1e-4},
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": 1e-4},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.nonlinear_mpc.MPCController",
            "dt": art.dt,
            "artifact": str(tmp_path / "observable"),
            "u_max": 2.0,
            "psd_ref": str(_envelope_npz(tmp_path, np.full((2, _L // 2 + 1), 1e-3))),
        },
    }
    with pytest.raises(ConfigConsistencyError, match="Pref_ms"):
        validate_simulation_config(config)

    art_path = tmp_path / "observable"
    config["controller"]["psd_ref"] = str(
        _envelope_npz(tmp_path, np.full((2, _L // 2 + 1), 1e-3), ms_power=np.full(2, 1e-3))
    )
    validate_simulation_config(config)
    assert art_path.with_suffix(".npz").exists()
