import csv

import numpy as np
import yaml

from seed_7104_forecasts import CHECK, ROOT, RUN, forecast_costs, replay_seed, wave_cost


def compare_release_currents() -> None:
    """Score both controllers' issued plans and delivered sequences from the common release."""
    config = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
    cnn_log = ROOT / "artifacts/sweep_2026_11/observable_benchmark/runs/obs_cnn_s7104/log.npz"
    with np.load(RUN / "trace.npz") as waveform, np.load(cnn_log) as spectral:
        times = waveform["controller.t"]
        first = np.flatnonzero(waveform["controller.solve_attempted"])[0]
        np.testing.assert_allclose(times[first], config["controller"]["control_start_s"])
        issued = waveform["controller.planned_u"][first]
        logged_cost = waveform["controller.cost_tracking"][first]
        stop = first + len(issued)
        waveform_currents = waveform["controller.u"]
        stride = round(np.diff(spectral["controller.t"])[0] / np.diff(times)[0])
        np.testing.assert_allclose(times[::stride], spectral["controller.t"])
        cnn_first = np.flatnonzero(spectral["controller.solve_attempted"])[0]
        np.testing.assert_allclose(spectral["controller.t"][cnn_first], times[first])
        cnn_issued = np.repeat(spectral["controller.planned_u"][cnn_first], stride, axis=0)
        cnn_currents = spectral["controller.u"][np.arange(len(times)) // stride]
        cnn_lfp = spectral["dynamics.lfp"][: round(times[stop] / config["dynamics"]["dt"]) + 1]
    actions = np.stack((waveform_currents[first:stop], cnn_currents[first:stop], issued, cnn_issued))
    np.testing.assert_allclose(actions[:2, :stride], actions[2:, :stride])
    np.testing.assert_allclose(waveform_currents[:first], cnn_currents[:first])
    eeg_histories = []
    measured = []
    final_counts = []
    for i, action in enumerate(actions):
        currents = waveform_currents.copy()
        currents[first:stop] = action
        lfp, eeg = replay_seed(currents, times[stop])
        if i == 1:
            np.testing.assert_array_equal(lfp, cnn_lfp)
        eeg_histories.append(eeg[:first + 1])
        measured.append(wave_cost(eeg[first + 1:stop + 1]).mean())
        window_samples = round(1 / config["dynamics"]["dt"])
        final_counts.append((np.ptp(lfp[-window_samples-1:-1], axis=0) > 5).sum())
    for history in eeg_histories[1:]:
        np.testing.assert_allclose(eeg_histories[0], history, atol=1e-9)
    predicted = forecast_costs(eeg_histories[0], waveform_currents, first, actions).mean(axis=1)
    np.testing.assert_allclose(predicted[2], logged_cost, rtol=1e-9)
    effort = config["controller"]["problem"]["w_u"] * np.sum(actions**2, axis=2).mean(axis=1)
    with (CHECK / "release_currents.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "start_s", "end_s", "predicted_tracking", "measured_tracking",
                         "weighted_effort", "predicted_total", "measured_total", "seizing_regions_at_end"))
        for i, name in enumerate(("waveform_delivered", "cnn_delivered", "waveform_issued_open_loop",
                                  "cnn_issued_open_loop")):
            writer.writerow((name, round(times[first], 2), round(times[stop], 2), predicted[i], measured[i],
                             effort[i], predicted[i] + effort[i], measured[i] + effort[i], final_counts[i]))


if __name__ == "__main__":
    compare_release_currents()
