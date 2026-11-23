import csv
from functools import cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from neuro.filtering import antialias_filter
from neuro.plant.eeg import EEGMeasurement
from neuro.plant.jansen_rit import JansenRitDynamics
from neuro.predictor.base import InferencePredictor

ROOT = Path(__file__).resolve().parents[3]
ARM = ROOT / "artifacts/thesis/sweep_2026_11/waveform_matched_geometry/final/waveform_mlp"
RUN = ARM / "evaluation/runs/waveform_mlp_s7104"
CHECK = ROOT / "artifacts/thesis/seed_7104_forecast_check"


def replay_seed(currents, end_s):
    """Reconstruct native regional LFP and filtered 50 Hz scalp EEG under recorded or branched currents."""
    config = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
    dynamics = config["dynamics"]
    dynamics.pop("class_path")
    stim = dynamics["stimulation"]
    stim["field_projection_path"] = str(ROOT / stim["field_projection_path"])
    plant = JansenRitDynamics.from_config(dynamics)
    stride = config["estimator"]["downsample"]
    samples = round(end_s / plant.dt) + 1
    lfp = np.empty((samples, plant.x.shape[1]))
    for tick in range(samples):
        _, log = plant.evaluate(tick * plant.dt, currents[tick // stride])
        lfp[tick] = log.lfp
    eeg = lfp @ EEGMeasurement().leadfield.T
    return lfp, antialias_filter(eeg, 1 / plant.dt, stride)[::stride]


def wave_cost(eeg):
    """Evaluate waveform tracking error at each measured or predicted sample."""
    with np.load(ARM / "reference.npz") as reference:
        return np.sum((eeg - reference["eeg_mean"].reshape(-1)) ** 2, axis=-1)


@cache
def waveform_predictor():
    """Load the frozen predictor once for the early decisions on this seed."""
    return InferencePredictor.load(ARM / "artifact.npz")


@eqx.filter_jit
def _forecast(model, state, plans):
    """Advance the model at controller precision for every candidate plan."""
    def advance(history, current):
        next_history = model.discrete_dynamics(history, current, 0.0, model.dt)
        return next_history, model.output(next_history)

    return jax.vmap(lambda plan: jax.lax.scan(advance, state, plan)[1])(plans)


def forecast_costs(eeg, currents, decision, plans):
    """Predict each stage's tracking cost after absorbing measurements through the decision tick."""
    model = waveform_predictor()
    history = model.initial_state()
    for tick in range(decision - max(model.n_history, model.n_u) + 1, decision + 1):
        history = model.absorb(history, eeg[tick], currents[tick - 1])
    predictions = _forecast(model, jnp.asarray(history), jnp.asarray(plans))
    return wave_cost(np.asarray(predictions))


def compare_plans() -> None:
    """Evaluate feedback forecasts and compare the two controllers' plans from the same 1.98 s history."""
    CHECK.mkdir(parents=True, exist_ok=True)
    snapshot = CHECK / "base_simulation.yaml"
    archived = (RUN / "config.yaml").read_bytes()
    if not snapshot.exists():
        snapshot.write_bytes(archived)
    assert snapshot.read_bytes() == archived
    with np.load(RUN / "trace.npz") as trace:
        times = trace["controller.t"]
        decision = np.flatnonzero(np.isclose(times, 1.98)).item()
        assert trace["controller.solve_attempted"][decision]
        currents = trace["controller.u"]
        plan = trace["controller.planned_u"][decision]
        tracking = trace["controller.cost_tracking"][decision]
        issued = trace["controller.solve_attempted"] & (times < 3)
        issued_plans = trace["controller.planned_u"][issued]
        issued_costs = trace["controller.cost_tracking"][issued]
        end = decision + len(plan)
        original_lfp, original_eeg = replay_seed(currents, times[end])
        for window_index, centre in enumerate(trace["spread_t"]):
            if centre + 0.5 > times[end] + 1e-9:
                break
            start = round((centre - 0.5) / 0.0001)
            ptp = np.ptp(original_lfp[start:start + 10000], axis=0)
            np.testing.assert_allclose(ptp, trace["spread_ptp"][:, window_index], atol=1e-8)
    with (CHECK / "feedback.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("decision_s", "endpoint_s", "next_3_predicted", "next_3_measured", "planned_horizon_predicted"))
        for tick, chosen, archived in zip(np.flatnonzero(issued), issued_plans, issued_costs, strict=True):
            pred = forecast_costs(original_eeg, currents, tick, chosen[None])[0]
            np.testing.assert_allclose(pred.mean(), archived, rtol=1e-9)
            measured = wave_cost(original_eeg[tick + 1:tick + 4])
            writer.writerow((times[tick], times[tick + 3], pred[:3].mean(), measured.mean(), archived))
    with np.load(CHECK / "cnn_handover" / "switch_1.98_12s.npz") as switch:
        cnn_plan = np.repeat(switch["first_plan"], 3, axis=0)
    alternatives = np.stack((plan, np.zeros_like(plan), cnn_plan))
    predictions = forecast_costs(original_eeg, currents, decision, alternatives)
    np.testing.assert_allclose(predictions[0].mean(), tracking, rtol=1e-6)
    future = slice(decision + 1, end + 1)
    rows = []
    counts = []
    summary = []
    for name, alternative, prediction in zip(("selected", "zero", "cnn_plan"), alternatives, predictions, strict=True):
        holds = currents.copy()
        holds[decision:end] = alternative
        lfp, measured = replay_seed(holds, times[end])
        np.testing.assert_allclose(measured[:decision + 1], original_eeg[:decision + 1], atol=1e-8)
        costs = wave_cost(measured[future])
        effort = 100 * np.sum(alternative**2, axis=1).mean()
        summary.append((name, prediction.mean(), costs.mean(), effort))
        if name == "selected":
            np.testing.assert_allclose(measured[decision+1:decision+4], original_eeg[decision+1:decision+4], atol=1e-8)
        for t, predicted, actual in zip(times[future], prediction, costs, strict=True):
            rows.append((name, f"{t:.2f}", predicted, actual))
        for endpoint in np.arange(2.0, 3.01, 0.25):
            stop = round(endpoint / 0.0001)
            count = (np.ptp(lfp[stop - 10000:stop], axis=0) > 5).sum()
            counts.append((name, endpoint, count))
    with (CHECK / "plan_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("plan", "predicted_tracking", "measured_tracking", "weighted_effort"))
        writer.writerows(summary)
    with (CHECK / "costs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("plan", "time_s", "predicted_tracking", "plant_tracking"))
        writer.writerows(rows)
    with (CHECK / "recruitment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("plan", "endpoint_s", "seizing_regions"))
        writer.writerows(counts)


if __name__ == "__main__":
    compare_plans()
