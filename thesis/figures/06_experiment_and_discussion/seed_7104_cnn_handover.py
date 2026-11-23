import sys

import numpy as np
import yaml
from neuro.control.mpc import TrajOptMPCController
from neuro.filtering import ObservableEstimator
from neuro.plant.eeg import EEGMeasurement
from neuro.plant.jansen_rit import JansenRitDynamics

from seed_7104_forecasts import CHECK, ROOT, RUN

CNN = ROOT / "artifacts/thesis/sweep_2026_11/observable_benchmark/runs/obs_cnn_s7104"


def simulate_switch(switch_s: float, duration: float):
    """Apply archived waveform holds until handover, then run the fitted spectral CNN in feedback."""
    snapshot = CHECK / "cnn_handover" / "spectral_cnn.yaml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    archived = (CNN / "config.yaml").read_bytes()
    if not snapshot.exists():
        snapshot.write_bytes(archived)
    assert snapshot.read_bytes() == archived
    cfg = yaml.safe_load(snapshot.read_text())
    dyn = yaml.safe_load((RUN / "config.yaml").read_text())["dynamics"]
    dyn.pop("class_path")
    stim = dyn["stimulation"]
    stim["field_projection_path"] = str(ROOT / stim["field_projection_path"])
    plant = JansenRitDynamics.from_config(dyn)
    sensor = EEGMeasurement()
    estimator = ObservableEstimator.from_config(cfg["estimator"])
    ctrl = cfg["controller"]
    for key in ("reference", "artifact"):
        ctrl["problem"][key] = str(ROOT / ctrl["problem"][key])
    if switch_s == 1.38:
        # Allow the reference check to compile the solver before the timed handover run.
        ctrl["solver"]["solver"]["options"]["max_cpu_time"] = 30.0
    ctrl["control_start_s"] = switch_s
    controller = TrajOptMPCController.from_config(ctrl)
    with np.load(RUN / "trace.npz") as trace:
        waveform = trace["controller.u"]
        reference_ptp = trace["spread_ptp"]
    dt = plant.dt
    frame_ticks = round(ctrl["dt"] / dt)
    waveform_ticks = round(0.02 / dt)
    release = round(switch_s / dt)
    last = round(duration / dt)
    lfp = np.empty((last + 1, len(plant.conn.region_labels)))
    currents = []
    solves = []
    first_plan = None
    current = np.zeros(3)
    for tick in range(last + 1):
        t = tick * dt
        lfp[tick] = plant.x[1] - plant.x[2]
        frame, _ = estimator.update(t, sensor.leadfield @ lfp[tick], current)
        if tick % frame_ticks == 0:
            if 0 < tick <= release:
                # The spectral predictor samples applied current at the preceding frame boundary.
                controller._u_last = waveform[(tick - frame_ticks) // waveform_ticks].copy()
            output, log = controller.update(t, np.zeros(1), frame)
            if tick >= release:
                if not log.success or tick == release:
                    print("solve", t, log.status, log.solve_time, output, flush=True)
                if tick == release:
                    first_plan = log.planned_u.copy()
                assert log.solve_attempted
                current = output
                solves.append((t, log.success, *output))
        if tick < release:
            current = waveform[tick // waveform_ticks]
        if tick % waveform_ticks == 0:
            currents.append(current.copy())
        if tick < last:
            plant.evaluate(t, current)
    ends = np.arange(1, duration + 1e-9, 0.25)
    ptp = np.array([np.ptp(lfp[i-10000:i], axis=0) for i in np.rint(ends / dt).astype(int)])
    n = (ends <= switch_s).sum()
    np.testing.assert_allclose(ptp[:n], reference_ptp[:, :n].T, atol=1e-8)
    return ends, ptp, np.asarray(currents), np.array(solves), first_plan


if __name__ == "__main__":
    CHECK.mkdir(parents=True, exist_ok=True)
    with np.load(CNN / "log.npz") as archive:
        first = np.flatnonzero(archive["controller.solve_attempted"])[0]
        logged = archive["controller.u"][first]
    _, _, _, calibration, _ = simulate_switch(1.38, 1.44)
    print("Control check", logged, calibration[0, 2:], flush=True)
    np.testing.assert_allclose(calibration[0, 2:], logged, atol=1e-4, rtol=1e-4)
    switch = float(sys.argv[1]) if len(sys.argv) > 1 else 1.98
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    ends, ptp, currents, solves, plan = simulate_switch(switch, duration)
    folder = CHECK / "cnn_handover"
    folder.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(folder / f"switch_{switch:.2f}_{duration:g}s.npz", endpoints=ends, ptp=ptp, currents=currents, solves=solves, first_plan=plan)
    print("Results", switch, ends, (ptp > 5).sum(axis=1), "accepted", solves[:,1].sum(), "/", len(solves), flush=True)
