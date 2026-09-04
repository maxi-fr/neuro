"""Probe how far each arm's model thinks stimulation can move the hinge, and write the json.

Rolls every arm's model over its own Control Horizon under a held control and records the hinge it
predicts, each in the units its own MPC minimizes, so the closed-loop result can be read against the
control authority the models actually believe they have. Arms B and C score the Observable hinge over
1550 log-power bins; arm A scores the windowed spectral hinge its waveform cost is written in. The
hinge values are therefore not comparable across the A/BC split, but the break-even amplitudes are:
each is that arm's own linear benefit set against the same quadratic control penalty.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from neuro.config import StftGeometry
from neuro.control.costs import jax_compute_log_power_frames
from neuro.predictor.data import reduce_trajectory_to_frames
from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.spectral import ObservableEnvelope, PsdEnvelope

GEOM = StftGeometry(n_segment=50, n_hop=5, band_hz=None, n_bin_pool=1, kernel="boxcar", kernel_width=1)
FS = 50.0
W_HINGE, W_U, U_MAX = 10.0, 10.0, 2.0
OBS_HORIZON, WAVE_HORIZON = 15, 75
T_PROBE = 2.2  # EZ and PZ both recruited
DIRS = {"e0-e1": [1.0, -1.0, 0.0], "e0-e2": [1.0, 0.0, -1.0], "e1-e2": [0.0, 1.0, -1.0]}
OUT = Path("results/predictor_comparison/final_t3/control_authority.json")

y = np.load(min(Path("data/experiment_excited_long/test").glob("*.npz")))["sensor_0.y_mea"][::200]


def observable_hinge_fn(artifact: str):  # noqa: ANN201 -- a closure over one arm's model
    """Build the Observable arm's held-control hinge, and the hinge it predicts at zero control."""
    power = np.asarray(ObservableEnvelope.load("data/healthy_psd_hop5.npz").power).reshape(-1)
    model = ObservableMLPModel.load(artifact)
    y_c, y_s = np.asarray(model.y_center).reshape(-1), np.asarray(model.y_scale).reshape(-1)
    seizing = reduce_trajectory_to_frames(y, GEOM, FS)[int(T_PROBE / 0.1)]
    x0 = jnp.concatenate([jnp.asarray((seizing - y_c) / y_s), jnp.zeros(model.n_u * model.n_controls)])

    def hinge(u: np.ndarray) -> float:
        """Observable hinge over the Control Horizon when ``u`` is held for all of it."""
        x, traj = x0, []
        for _ in range(OBS_HORIZON):
            x = model.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.1)
            traj.append(np.asarray(x[: model.n_outputs]) * y_s + y_c)
        excess = np.maximum(0.0, np.stack(traj) - power)
        return W_HINGE * float((excess**2).sum()) / (OBS_HORIZON * power.shape[0])

    return hinge, float((seizing > power).mean()), int(power.shape[0])


def waveform_hinge_fn(artifact: str):  # noqa: ANN201 -- a closure over arm A's model
    """Build arm A's held-control spectral hinge over its 75-step Control Horizon."""
    env = PsdEnvelope.load("data/healthy_psd.npz")
    model = WaveformMLPModel.load(artifact)
    y_c, y_s = np.asarray(model.y_center).reshape(-1), np.asarray(model.y_scale).reshape(-1)
    start = int(T_PROBE * FS)
    hist = (y[start - model.n_y : start] - y_c) / y_s
    x0 = jnp.concatenate([jnp.asarray(hist.reshape(-1)), jnp.zeros(model.n_u * model.n_controls)])
    log_power_ref = jnp.log(jnp.asarray(env.power)[None, :, 1:])

    def hinge(u: np.ndarray) -> float:
        """Spectral hinge over the Control Horizon when ``u`` is held for all of it."""
        x, traj = x0, []
        for _ in range(WAVE_HORIZON):
            x = model.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.02)
            traj.append(np.asarray(x[: model.n_outputs]) * y_s + y_c)
        frames = jax_compute_log_power_frames(jnp.asarray(np.stack(traj)), fs=env.fs, window=env.window, hop=env.hop)
        return W_HINGE * float(jnp.mean(jnp.maximum(0.0, frames - log_power_ref) ** 2))

    return hinge


def sweep(hinge) -> dict:  # noqa: ANN001 -- any held-control hinge closure
    """Slope, break-even amplitude and full-box hinge of every electrode pair, from one hinge closure."""
    base = hinge(np.zeros(3))
    out = {"hinge_at_zero": base, "directions": {}}
    for name, v in DIRS.items():
        vec = np.array(v)
        sign = 1 if hinge(0.5 * vec) < hinge(-0.5 * vec) else -1
        slope = (base - hinge(sign * 0.5 * vec)) / 0.5
        at_box = hinge(sign * U_MAX * vec)
        out["directions"][name] = {
            "sign": sign,
            "slope": slope,
            "break_even": slope / (2 * W_U * float((vec**2).sum())),
            "hinge_at_box": at_box,
            "rel_at_box": (at_box - base) / base,
        }
    return out


obs_dmd, over, n_bins = observable_hinge_fn("artifacts/cmp_observable_dmd_hop5/model")
obs_mlp, _, _ = observable_hinge_fn("artifacts/cmp_observable_mlp_hop5/model")
result = {
    "frame_time_s": T_PROBE,
    "bins": n_bins,
    "bins_over_envelope": over,
    "w_u": W_U,
    "w_hinge": W_HINGE,
    "u_max": U_MAX,
    "arms": {
        "observable_dmd": sweep(obs_dmd),
        "observable_mlp": sweep(obs_mlp),
        "waveform_mlp": sweep(waveform_hinge_fn("artifacts/cmp_waveform_mlp_1p5s/model")),
    },
}
OUT.write_text(json.dumps(result, indent=2))
for arm, rec in result["arms"].items():
    print(
        f"{arm:16s} hinge0 {rec['hinge_at_zero']:7.2f}  "
        + "  ".join(
            f"{n} slope {d['slope']:5.2f} a* {d['break_even']:.3f} box {100 * d['rel_at_box']:+5.1f}%"
            for n, d in rec["directions"].items()
        )
    )
