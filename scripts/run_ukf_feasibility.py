"""Feasibility test of tracking a whole-brain seizure with a reduced region-space UKF.

Loads the 62-channel seizure EEG from a capture run -- simulate a seizure plant through the
orchestrator (``scripts/configs/jansen_rit_seizure.yaml`` via ``scripts/run_simulation.py``) and
point ``--data`` at the resulting ``log.npz`` -- and drives a *reduced* ``M``-region Jansen-Rit
UKF. Unlike a channel-space surrogate, the reduced nodes are
actual brain regions -- the EZ/PZ plus their strongest structural neighbours -- so a node sits on
the seizure source. The measurement model is the real ``M``-column leadfield restricted to the
``M`` EEG channels that best see those regions (``estimate_eeg_gains=False``), and the per-node
excitatory gain ``A`` (the bifurcation knob), global coupling ``K`` and weights ``W`` are
estimated online from smart, plant-derived initial values (``A`` per region, ``K = 0.75``,
``W`` = structural sub-connectome).

Because the leadfield carries physical units, channels are only rescaled (not z-scored away):
each channel ``c`` and its leadfield row are divided by the channel's standard deviation, so the
amplitude that ``A`` controls survives and ``R`` stays well-scaled.

The script reports the one-step UKF innovation and, more importantly for MPC, the multi-step
free-run prediction error against a persistence baseline, plus the learned ``A``/``K``.

Usage
-----
    uv run python scripts/run_ukf_feasibility.py
    uv run python scripts/run_ukf_feasibility.py --steps 20000 --n-select 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from neuro.connectome import Connectome, load_connectome
from neuro.estimator import UKFEstimator
from neuro.jansen_rit import JansenRitParams
from neuro.seizure import A_EZ, A_HEALTHY, A_PZ, DT, EZ_REGIONS, PZ_REGIONS, K, focus_indices
from utils.channel_selection import leadfield_focus_scores

DATA_PATH = Path("results/seizure_capture/log.npz")
OUTPUT_DIR = Path("simulations/ukf_feasibility")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the UKF feasibility runner."""
    parser = argparse.ArgumentParser(description="Reduced region-space UKF feasibility test on seizure EEG.")
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Capture log.npz from a seizure run_simulation run."
    )
    parser.add_argument("--n-select", type=int, default=10, help="Number of reduced regions and EEG channels.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of dt steps to filter (1e-4 s each).")
    parser.add_argument("--q-x5", type=float, default=1e-3, help="Process-noise variance on x5.")
    parser.add_argument("--q-a", type=float, default=1e-7, help="Process-noise variance on the estimated A (tamed).")
    parser.add_argument("--p-a", type=float, default=1e-2, help="Initial variance on the estimated A (tamed).")
    parser.add_argument(
        "--q-input", type=float, default=1e-2, help="Process-noise variance on the estimated mean input."
    )
    parser.add_argument("--p-input", type=float, default=1.0, help="Initial variance on the estimated mean input.")
    parser.add_argument("--r-channel", type=float, default=1e-2, help="Measurement-noise variance per channel.")
    parser.add_argument("--horizon", type=int, default=10000, help="Max free-run prediction horizon in steps.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs.")
    return parser.parse_args()


def select_regions_and_channels(connectome: Connectome, n_select: int) -> tuple[list[int], list[int]]:
    """Pick the reduced regions (EZ/PZ + strongest neighbours) and the channels that observe them.

    Regions are the EZ/PZ nodes plus the top structural neighbours by bidirectional weight to the
    focus; channels are the ``n_select`` EEG sensors with the largest leadfield onto those regions.
    """
    focus = focus_indices(connectome)
    weights = connectome.weights
    coupling_to_focus = weights[:, focus].sum(axis=1) + weights[focus, :].sum(axis=0)
    coupling_to_focus[focus] = -np.inf  # do not re-pick the focus regions
    n_extra = max(0, n_select - len(focus))
    extra = np.argsort(coupling_to_focus)[::-1][:n_extra].tolist()
    regions = focus + extra

    focus_scores = leadfield_focus_scores(connectome.gain, regions)
    channels = np.argsort(focus_scores)[::-1][:n_select].tolist()
    return regions, channels


def main(  # noqa: PLR0913, PLR0915
    *,
    data: Path = DATA_PATH,
    n_select: int = 10,
    steps: int = 10000,
    q_x5: float = 1e-3,
    q_a: float = 1e-7,
    p_a: float = 1e-2,
    q_input: float = 1e-2,
    p_input: float = 1.0,
    r_channel: float = 1e-2,
    horizon: int = 10000,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Build the reduced region-space model, run the UKF, and report prediction quality."""
    output_dir.mkdir(parents=True, exist_ok=True)

    connectome = load_connectome(speed=50.0)

    # Seizure EEG comes from a capture run: simulate a seizure plant through the orchestrator
    # (scripts/configs/jansen_rit_seizure.yaml -> scripts/run_simulation.py) and point --data at
    # the resulting log.npz; the 62-channel EEG is the universally-logged measurement.
    blob = np.load(data, allow_pickle=True)
    eeg = np.asarray(blob["universal_y_mea"], dtype=np.float64).T  # (62, T)
    labels = connectome.channel_labels
    regions, channels = select_regions_and_channels(connectome, n_select)
    region_labels = [str(connectome.region_labels[r]) for r in regions]
    a0 = np.array(
        [A_EZ if rl in EZ_REGIONS else A_PZ if rl in PZ_REGIONS else A_HEALTHY for rl in region_labels],
        dtype=np.float64,
    )
    w0 = connectome.weights[np.ix_(regions, regions)].copy()
    np.fill_diagonal(w0, 0.0)

    print(f"Reduced regions: {', '.join(region_labels)}")
    print(f"Measured channels: {', '.join(labels[c] for c in channels)}")
    print(f"Init A per region (from plant): {np.array2string(a0, precision=2)}")

    steps = min(steps, eeg.shape[1] - 1)

    # Physical-unit scaling: divide each channel and its leadfield row by the channel std, so the
    # measurement and model output share units and amplitude (which A controls) is preserved.
    chan_std = eeg[:, :steps].std(axis=1)
    chan_mean = eeg[:, :steps].mean(axis=1)
    gain_scaled = connectome.gain[:, regions] / chan_std[:, None]
    y_meas = (eeg - chan_mean[:, None]) / chan_std[:, None]  # (62, T), train-window stats

    print(f"Filtering {steps} steps ({steps * DT:.2f} s) of a {len(regions)}-region reduced model...")

    est = UKFEstimator(
        dt=DT,
        n_nodes=len(regions),
        gain=gain_scaled,  # real M-column leadfield, per-channel scaled
        delays=None,  # instantaneous coupling so the free-run stays history-free
        selected_channels=channels,
        params=JansenRitParams(),
        estimate_eeg_gains=False,
        estimate_a=True,
        estimate_input=True,  # per-node effective input absorbs the omitted 66-region drive
        q_x5=q_x5,
        q_a=q_a,
        p_a=p_a,
        q_input=q_input,
        p_input=p_input,
        r_channel=r_channel,
        initial_k=K,
        initial_w=w0,
        initial_a=a0,
        # Fix K at the plant value: zero process noise and a negligible prior variance so the
        # tiny-spread sigma points and ~zero Kalman gain leave K essentially constant at 0.75.
        q_k=0.0,
        p_k=1e-12,
    )

    innovations = np.empty((steps, n_select), dtype=np.float64)
    a_hist = np.empty((steps, len(regions)), dtype=np.float64)
    input_hist = np.empty((steps, len(regions)), dtype=np.float64)
    k_hist = np.empty(steps, dtype=np.float64)
    diverged_at = -1

    t_start = time.perf_counter()
    for step in range(steps):
        z = y_meas[channels, step]
        _x_hat, log = est.update(step * DT, z, 0.0)
        innovations[step] = est.ukf.y
        a_hist[step] = log.a_est
        input_hist[step] = log.input_est
        k_hist[step] = log.K_est
        if not np.all(np.isfinite(est.ukf.x)):
            diverged_at = step
            break
    elapsed = time.perf_counter() - t_start

    if diverged_at >= 0:
        print(f"\n*** Filter DIVERGED at step {diverged_at} ({diverged_at * DT:.3f} s) -- non-finite state. ***")
        return

    print(f"Done in {elapsed:.1f} s ({1e3 * elapsed / steps:.2f} ms/step).")

    # One-step prediction error (UKF innovation) vs a persistence baseline z[t] ~ z[t-1].
    z_win = y_meas[channels, :steps]
    pred_rel = np.linalg.norm(innovations) / np.linalg.norm(z_win)
    persist_rel = np.linalg.norm(np.diff(z_win, axis=1)) / np.linalg.norm(z_win[:, 1:])
    per_channel = np.linalg.norm(innovations, axis=0) / np.linalg.norm(z_win, axis=1)

    print("\n--- One-step prediction (lower is better) ---")
    print(f"UKF relative innovation RMS : {pred_rel:.4f}")
    print(f"Persistence baseline        : {persist_rel:.4f}")
    print(
        "Per-channel UKF innovation  : "
        + ", ".join(f"{labels[channels[i]]}={per_channel[i]:.3f}" for i in range(n_select))
    )

    print("\n--- Estimated bifurcation gain A (true EZ/PZ seizure ~3.4-3.6, healthy 3.25), K fixed ---")
    a_final = a_hist[-1]
    print(
        f"A start={a_hist[0].mean():.3f}  ->  end mean={a_final.mean():.3f}  (min {a_final.min():.3f}, max {a_final.max():.3f})"
    )
    print(f"K (fixed at plant value): {k_hist[-1]:.4f}")
    input_final = input_hist[-1]
    print(
        f"Mean input I (init {input_hist[0].mean():.1f}): end mean={input_final.mean():.1f}  "
        f"(min {input_final.min():.1f}, max {input_final.max():.1f})"
    )

    # Free-run prediction vs horizon: freeze the final estimate and roll the model forward with no
    # measurement updates, then compare to the true future channels over several horizons (the
    # seizure rhythm is ~3-10 Hz, so a meaningful test must span several 100-300 ms cycles).
    h_max = min(horizon, eeg.shape[1] - steps)
    free_rel = persist_free = float("nan")
    preds = future = np.empty((0, n_select), dtype=np.float64)
    if h_max > 0:
        x_free = est.ukf.x.copy()
        preds = np.empty((h_max, n_select), dtype=np.float64)
        for h in range(h_max):
            x_free = est.ukf.fx(x_free, DT, 0.0, steps + h)
            preds[h] = est.ukf.hx(x_free)
        future = y_meas[channels, steps : steps + h_max].T
        hold = y_meas[channels, steps - 1]
        finite = "ok" if np.all(np.isfinite(preds)) else "NON-FINITE"
        print(f"\n--- Free-run prediction vs horizon [{finite}] (model / persistence; lower is better) ---")
        for h in (500, 1000, 2000, 5000, 10000):
            if h > h_max:
                break
            fut_h = future[:h]
            model_h = float(np.linalg.norm(preds[:h] - fut_h) / np.linalg.norm(fut_h))
            persist_h = float(np.linalg.norm(fut_h - hold) / np.linalg.norm(fut_h))
            verdict = "model wins" if model_h < persist_h else "persistence wins"
            print(f"  {h * DT * 1e3:5.0f} ms : model {model_h:.4f} / persistence {persist_h:.4f}  ({verdict})")
        free_rel = float(np.linalg.norm(preds - future) / np.linalg.norm(future))
        persist_free = float(np.linalg.norm(future - hold) / np.linalg.norm(future))

    out_path = output_dir / "ukf_feasibility.npz"
    np.savez(
        out_path,
        regions=np.array(regions),
        region_labels=np.array(region_labels),
        channels=np.array(channels),
        channel_labels=labels[channels],
        innovations=innovations,
        a_hist=a_hist,
        input_hist=input_hist,
        k_hist=k_hist,
        pred_rel=pred_rel,
        persist_rel=persist_rel,
        free_rel=free_rel,
        persist_free=persist_free,
        # Standardized waveforms for plotting: filtering window (true), and the free-run horizon
        # (true vs model). One-step model output = meas_filt - innovations.
        meas_filt=z_win.T,
        meas_future=future,
        free_preds=preds,
        dt=DT,
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        data=cli_args.data,
        n_select=cli_args.n_select,
        steps=cli_args.steps,
        q_x5=cli_args.q_x5,
        q_a=cli_args.q_a,
        p_a=cli_args.p_a,
        q_input=cli_args.q_input,
        p_input=cli_args.p_input,
        r_channel=cli_args.r_channel,
        horizon=cli_args.horizon,
        output_dir=cli_args.output_dir,
    )
