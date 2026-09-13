from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from neuro.control.mpc import NullspaceReducedModel, _build_problem
from neuro.predictor.inference import InferencePredictor
from neuro.run_view import PredictionSeries

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.run_view import Run
    from neuro.types import FloatArray


@eqx.filter_jit
def predict(model: InferencePredictor, state: jax.Array, controls: jax.Array, t: float, dt: float) -> jax.Array:
    """Return outputs ``(H+1, p)`` under recorded future Control Currents without State Absorption."""

    def step(x: jax.Array, item: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        u, time = item
        next_x = model.discrete_dynamics(x, u, time, dt)
        return next_x, model.output(next_x, u, time + dt)

    _, future = jax.lax.scan(step, state, (controls, t + jnp.arange(len(controls)) * dt))
    return jnp.concatenate((model.output(state, controls[0], t)[None], future))


def replay_predictions(
    model: InferencePredictor, measurements: FloatArray, controls: FloatArray, times: FloatArray, *, horizon: int
) -> tuple[FloatArray, FloatArray]:
    """Replay causal decision measurements and predict only futures fully covered by the recording."""
    dt = float(times[1] - times[0])
    state = model.initial_state()
    previous = np.zeros(model.m)
    predictions = np.full((len(times), horizon + 1, model.p), np.nan)
    observed = np.full((len(times), model.p), np.nan)
    for i, t in enumerate(times):
        state = model.absorb(state, measurements[i], previous)
        previous = controls[i]
        if not model.is_ready(state):
            continue
        observed[i] = np.asarray(model.output(jnp.asarray(state), jnp.asarray(previous), t))
        if i + horizon >= len(times):
            continue
        predictions[i] = np.asarray(
            predict(model, jnp.asarray(state), jnp.asarray(controls[i : i + horizon]), float(t), dt)
        )
    return predictions, observed


def prepare_replay(run: Run, config: dict[str, Any], destination: Path) -> None:
    """Prepare Rollouts from component observations and recorded Control Currents."""
    dt = float(config["controller"]["dt"])
    if config["estimator"] != run.config["estimator"] or config["sensors"] != run.config["sensors"]:
        msg = "Replay requires the same sensor and Estimator configuration as the recording."
        raise ValueError(msg)
    if dt != float(run.config["controller"]["dt"]):
        msg = "Replay requires the recording's controller period."
        raise ValueError(msg)
    if config["dynamics"].get("stimulation") != run.config["dynamics"].get("stimulation"):
        msg = "Replay requires the recording's stimulation montage."
        raise ValueError(msg)
    problem = _build_problem(config["controller"]["problem"])
    model = problem.model
    if isinstance(model, NullspaceReducedModel):
        model = model.base_model
    if not isinstance(model, InferencePredictor):
        msg = "The configured Predictor does not implement replay."
        raise TypeError(msg)
    times, controls = run.signal("controller", "u")
    measurements = run.measurements()
    expected = times[0] + np.arange(len(times)) * dt
    if not np.allclose(times, expected, rtol=0, atol=1e-9):
        msg = "Replay requires uniformly spaced controller decisions."
        raise ValueError(msg)
    predictions, observed = replay_predictions(model, measurements, controls, times, horizon=problem.N - 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    PredictionSeries(times, predictions, observed, dt, run.frequencies()).save(destination)
    destination.with_suffix(".yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    destination.with_suffix(".json").write_text(
        json.dumps({"source": str(run.directory.resolve()), "conditioning": "applied currents"}, indent=2),
        encoding="utf-8",
    )
