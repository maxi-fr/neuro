"""Fit and persist a reduced-region UKF predictor for the seizure + tES regime.

The reduced UKF (``neuro.estimator.UKFEstimator``) is an *online* estimator with no
offline-trained weights, so "training" here means: filter a bounded segment of the
**train** seeds (with the recorded tES applied) so the per-node excitatory gain ``A``,
the connection weights ``W`` and the per-node mean input converge from their smart
plant-derived initial values, then snapshot those converged parameters. The comparison
notebook (:class:`neuro.prediction.UKFPredictor`) reloads them, holds them fixed, and
re-estimates only the dynamic state from each test context before free-running.

The reduced regions are the EZ/PZ seizure focus plus their strongest structural
neighbours (:func:`neuro.seizure.focus_indices`); the measurement is the real
``M``-column leadfield onto all 62 EEG channels, per-channel standardised so the
amplitude ``A`` controls survives.

Usage
-----
    uv run python scripts/run_ukf_predictor.py
    uv run python scripts/run_ukf_predictor.py --n-select 10 --steps 10000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from neuro.connectome import Connectome, compute_gamma, load_connectome
from neuro.estimator import UKFEstimator
from neuro.jansen_rit import JansenRitParams
from neuro.seizure import DT, SPEED, K, build_seizure_a_gains, focus_indices

TRAIN_GLOB = "results/simulation_2026-06-23_18-49-20/train/*.npz"
ELECTRODES = ("CP5", "T7")  # matches configs/simulation/jansen_rit_seizure_excited.yaml
GAMMA_SIGMA = 20.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the UKF predictor trainer."""
    parser = argparse.ArgumentParser(description="Fit and save a reduced-region UKF predictor.")
    parser.add_argument("--train-glob", type=str, default=TRAIN_GLOB, help="Glob for train-split seed npz files.")
    parser.add_argument("--n-select", type=int, default=10, help="Number of reduced regions.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of dt steps to filter for convergence.")
    parser.add_argument("--q-x5", type=float, default=1e-3, help="Process-noise variance on x5.")
    parser.add_argument("--q-a", type=float, default=1e-7, help="Process-noise variance on the estimated A.")
    parser.add_argument("--p-a", type=float, default=1e-2, help="Initial variance on the estimated A.")
    parser.add_argument("--q-input", type=float, default=1e-2, help="Process-noise variance on the mean input.")
    parser.add_argument("--p-input", type=float, default=1.0, help="Initial variance on the mean input.")
    parser.add_argument("--r-channel", type=float, default=1e-2, help="Measurement-noise variance per channel.")
    parser.add_argument("--out", type=Path, default=Path("results/ukf_predictor.npz"), help="Artifact path.")
    return parser.parse_args()


def select_regions(connectome: Connectome, n_select: int) -> list[int]:
    """EZ/PZ focus regions plus the strongest structural neighbours by bidirectional weight."""
    focus = focus_indices(connectome)
    weights = connectome.weights
    coupling = weights[:, focus].sum(axis=1) + weights[focus, :].sum(axis=0)
    coupling[focus] = -np.inf  # do not re-pick the focus regions
    n_extra = max(0, n_select - len(focus))
    extra = np.argsort(coupling)[::-1][:n_extra].tolist()
    return focus + extra


def main() -> None:
    """Fit the reduced-region UKF parameters on the train seeds and save the artifact."""
    args = parse_args()

    files = sorted(glob.glob(args.train_glob))  # noqa: PTH207
    if not files:
        print(f"No train files matched {args.train_glob!r}")
        return
    train_file = files[0]
    print(f"Fitting UKF parameters on {train_file} ({args.steps} steps, {args.steps * DT:.2f} s)...")

    connectome = load_connectome(speed=SPEED)
    regions = select_regions(connectome, args.n_select)
    region_labels = [str(connectome.region_labels[r]) for r in regions]
    print(f"Reduced regions: {', '.join(region_labels)}")

    gain_m = connectome.gain[:, regions]  # (62, M), raw units
    gamma_full = np.atleast_2d(compute_gamma(connectome.centres, list(ELECTRODES), GAMMA_SIGMA))  # (n_elec, 76)
    gamma_m = gamma_full[:, regions]  # (n_elec, M)
    a0 = build_seizure_a_gains(connectome)[regions]
    w0 = connectome.weights[np.ix_(regions, regions)].copy()
    np.fill_diagonal(w0, 0.0)

    with np.load(train_file) as data:
        eeg = np.asarray(data["universal_y_mea"], dtype=np.float64).T  # (62, T)
        u = np.asarray(data["universal_u"], dtype=np.float64)  # (T, n_elec)
    steps = min(args.steps, eeg.shape[1] - 1)
    chan_mean = eeg[:, :steps].mean(axis=1)
    chan_std = eeg[:, :steps].std(axis=1)
    y_std = (eeg - chan_mean[:, None]) / chan_std[:, None]
    gain_scaled = gain_m / chan_std[:, None]

    est = UKFEstimator(
        dt=DT,
        n_nodes=len(regions),
        gain=gain_scaled,
        gamma=gamma_m,
        delays=None,
        params=JansenRitParams(),
        estimate_a=True,
        estimate_input=True,
        q_x5=args.q_x5,
        q_a=args.q_a,
        p_a=args.p_a,
        q_input=args.q_input,
        p_input=args.p_input,
        r_channel=args.r_channel,
        initial_k=K,
        initial_w=w0,
        initial_a=a0,
        q_k=0.0,
        p_k=1e-12,
    )

    for k in range(steps):
        est.update(k * DT, y_std[:, k], u[k])
        if not np.all(np.isfinite(est.ukf.x)):
            print(f"*** Filter diverged at step {k} ({k * DT:.3f} s); aborting. ***")
            return

    n_q = 6 * len(regions)
    xf = est.ukf.x
    a_conv = xf[est.a_start : est.a_start + len(regions)].copy()
    input_conv = xf[est.input_start : est.input_start + len(regions)].copy()
    w_conv = xf[n_q + 1 : n_q + 1 + len(regions) ** 2].reshape(len(regions), len(regions)).copy()
    print(f"Converged A: mean={a_conv.mean():.3f} (min {a_conv.min():.3f}, max {a_conv.max():.3f})")
    print(f"Converged mean input: mean={input_conv.mean():.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        regions=np.array(regions),
        region_labels=np.array(region_labels),
        gain=gain_m,
        gamma_m=gamma_m,
        chan_mean=chan_mean,
        chan_std=chan_std,
        initial_a=a_conv,
        initial_w=w_conv,
        initial_k=float(K),
        initial_input=input_conv,
        dt=float(DT),
        q_x5=float(args.q_x5),
        q_a=float(args.q_a),
        p_a=float(args.p_a),
        q_input=float(args.q_input),
        p_input=float(args.p_input),
        r_channel=float(args.r_channel),
    )
    print(f"Saved UKF predictor artifact -> {args.out}")


if __name__ == "__main__":
    main()
