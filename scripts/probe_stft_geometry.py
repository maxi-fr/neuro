from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from neuro.artifacts import load_rollout_artifact
from neuro.config import EegMsSpec, StftSpec
from neuro.filtering import antialias_filter
from neuro.predictor.data import load_trajectory
from neuro.predictor.losses import EegMsLoss, LossContext, StftLoss, spectrogram
from neuro.spectral import LOG_FLOOR, PsdEnvelope

if TYPE_CHECKING:
    from neuro.artifacts import RolloutArtifact
    from neuro.types import FloatArray

# Seizure branches only: the healthy branch runs a different A vector than the predictor's
# training plant, so its rollouts would score distribution shift rather than geometry.
_BRANCHES = ("pre_onset", "ez_ignited", "mid_spread", "saturated")
_ACF_SEGMENTS = (15, 25, 37, 50)
_DECAY = float(np.exp(-1.0))
# Var[log X - log Y] for two independent chi-squared-2 periodogram cells: 2 * polygamma(1, 1).
_CHI2_FLOOR = 2.0 * float(np.pi**2 / 6.0)


@dataclass(frozen=True)
class Candidate:
    """One loss geometry to score, named for the report."""

    name: str
    spec: StftSpec | EegMsSpec

    def build(self, fs: float) -> StftLoss | EegMsLoss:
        """Instantiate the loss this candidate declares at sampling rate ``fs``."""
        if isinstance(self.spec, StftSpec):
            return StftLoss.from_spec(self.spec, fs)
        return EegMsLoss.from_spec(self.spec, fs, "eeg_ms")


def candidates() -> list[Candidate]:
    """Geometries spanning the section 6 trade table, plus the broadband and Welch endpoints."""
    band = (3.0, 12.0)
    return [
        Candidate("welch_50", StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=50)),
        Candidate("welch_75", StftSpec(weight=1.0, n_span=75, n_segment=75, n_hop=75)),
        Candidate("W50_H25_s75", StftSpec(weight=1.0, n_span=75, n_segment=50, n_hop=25)),
        Candidate("W50_H12_s75", StftSpec(weight=1.0, n_span=75, n_segment=50, n_hop=12)),
        Candidate("W25_H25_s50", StftSpec(weight=1.0, n_span=50, n_segment=25, n_hop=25)),
        Candidate("W25_H12_s50", StftSpec(weight=1.0, n_span=50, n_segment=25, n_hop=12)),
        Candidate("W25_H12_s75", StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=12)),
        Candidate("W25_H6_s75", StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=6)),
        Candidate("W15_H8_s75", StftSpec(weight=1.0, n_span=75, n_segment=15, n_hop=8)),
        Candidate("W37_H19_s75", StftSpec(weight=1.0, n_span=75, n_segment=37, n_hop=19)),
        Candidate("W25_H12_s75_band", StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=12, band_hz=band)),
        Candidate("welch_50_band", StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=50, band_hz=band)),
        Candidate("W25_H12_s75_pool2", StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=12, n_bin_pool=2)),
        Candidate(
            "W25_H6_s75_hann3",
            StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=6, kernel="hann", kernel_width=3),
        ),
        Candidate(
            "W25_H6_s75_box3",
            StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=6, kernel="boxcar", kernel_width=3),
        ),
        Candidate(
            "W25_H6_s75_box2",
            StftSpec(weight=1.0, n_span=75, n_segment=25, n_hop=6, kernel="boxcar", kernel_width=2),
        ),
        Candidate("eeg_ms_s50", EegMsSpec(weight=1.0, span_s=1.0)),
        Candidate("eeg_ms_s75", EegMsSpec(weight=1.0, span_s=1.5)),
    ]


def raw_context(fs: float, n_channels: int) -> LossContext:
    """Context whose ``to_raw`` is the identity, so the probe can feed raw millivolts directly."""
    return LossContext(
        y_center=torch.zeros(n_channels, dtype=torch.float64),
        y_scale=torch.ones(n_channels, dtype=torch.float64),
        fs=fs,
        epoch=None,
    )


def per_sample(loss: StftLoss | EegMsLoss, pred: FloatArray, true: FloatArray, ctx: LossContext) -> FloatArray:
    """Score every rollout of ``(B, T, C)`` separately, reducing every axis but the batch."""
    a = torch.as_tensor(pred, dtype=torch.float64)
    b = torch.as_tensor(true, dtype=torch.float64)
    if isinstance(loss, StftLoss):
        residual = loss.log_spectrogram(a, ctx) - loss.log_spectrogram(b, ctx)
    else:
        residual = torch.log(loss.windowed_power(a, ctx) + LOG_FLOOR) - torch.log(
            loss.windowed_power(b, ctx) + LOG_FLOOR
        )
    return np.asarray(torch.mean(residual.flatten(1) ** 2, dim=1), dtype=np.float64)


def decimate_store(path: Path, downsample: int, dt_plant: float) -> FloatArray:
    """Load one ``(rollout, channel, sample)`` EEG store and decimate it the way training data is loaded.

    Rows are filtered one at a time, so a 1 GB store never becomes resident.
    """
    store = np.load(path, mmap_mode="r")
    rows = [
        antialias_filter(np.asarray(store[row], dtype=np.float64).T, 1.0 / dt_plant, downsample)[::downsample]
        for row in range(store.shape[0])
    ]
    return np.stack(rows)


def phase_randomise(y: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Surrogate with the same per-channel PSD and no time-varying envelope."""
    spectrum = np.fft.rfft(y, axis=0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phase[0] = 0.0
    return np.asarray(np.fft.irfft(np.abs(spectrum) * np.exp(1j * phase), n=y.shape[0], axis=0), dtype=np.float64)


def frame_autocorrelation(y: FloatArray, n_segment: int, fs: float, max_lag: int) -> FloatArray:
    """Autocorrelation of ``log P(m, c, f)`` along the frame axis, hop 1, averaged over channels and bins."""
    power = spectrogram(torch.as_tensor(y.T), n_segment, 1, fs=fs)[..., 1:]
    log_power = np.asarray(torch.log(power + LOG_FLOOR), dtype=np.float64).transpose(1, 0, 2)
    centred = log_power - log_power.mean(axis=0, keepdims=True)
    variance = np.mean(centred**2, axis=0)
    acf = np.stack([np.mean(centred[lag:] * centred[: len(centred) - lag], axis=0) for lag in range(max_lag + 1)])
    return np.asarray(np.mean(acf / variance, axis=(1, 2)), dtype=np.float64)


def decay_lag(acf: FloatArray) -> float:
    """First lag where the autocorrelation falls below ``1/e``, interpolated between samples."""
    below = np.flatnonzero(acf < _DECAY)
    if below.size == 0:
        return float(len(acf) - 1)
    hi = int(below[0])
    lo = hi - 1
    return lo + (acf[lo] - _DECAY) / (acf[lo] - acf[hi])


def report_correlation_width(data_dir: Path, art: RolloutArtifact, max_lag: int, seed: int) -> None:
    """Print the frame-axis correlation width of the log-power trajectory, against a stationary surrogate."""
    fs = 1.0 / art.dt
    rng = np.random.default_rng(seed)
    files = sorted(str(p) for p in data_dir.glob("*.npz"))
    trajectories = [load_trajectory(f, None, art.downsample, art.dt / art.downsample)[1] for f in files]

    print(f"\nFrame-axis correlation of log P(m, c, f), hop 1, {len(trajectories)} trajectories at {fs:g} Hz")
    print(f"{'W':>4} {'measured':>10} {'surrogate':>10} {'measured':>10} {'surrogate':>10}")
    print(f"{'':>4} {'(frames)':>10} {'(frames)':>10} {'(s)':>10} {'(s)':>10}")
    for n_segment in _ACF_SEGMENTS:
        acf = np.mean([frame_autocorrelation(y, n_segment, fs, max_lag) for y in trajectories], axis=0)
        surrogate = np.mean(
            [frame_autocorrelation(phase_randomise(y, rng), n_segment, fs, max_lag) for y in trajectories], axis=0
        )
        w_meas, w_surr = decay_lag(acf), decay_lag(surrogate)
        print(f"{n_segment:>4} {w_meas:>10.1f} {w_surr:>10.1f} {w_meas / fs:>10.3f} {w_surr / fs:>10.3f}")


def load_children(ensemble_dir: Path, arm: str, art: RolloutArtifact) -> dict[str, FloatArray]:
    """Decimated children per seizure branch, shaped ``(parent, child, sample, channel)``."""
    manifest = json.loads((ensemble_dir / "manifest.json").read_text(encoding="utf-8"))
    n_children = int(manifest["n_children"])
    dt_plant = float(manifest["dt"])
    stores: dict[str, FloatArray] = {}
    for branch in _BRANCHES:
        path = ensemble_dir / "eeg" / f"{branch}_{arm}.npy"
        if not path.exists():
            continue
        rollouts = decimate_store(path, art.downsample, dt_plant)
        stores[branch] = rollouts.reshape(-1, n_children, *rollouts.shape[1:])
        print(f"  {branch}_{arm}: {stores[branch].shape}", flush=True)
    if not stores:
        msg = f"no {arm} stores found under {ensemble_dir / 'eeg'}"
        raise SystemExit(msg)
    return stores


@dataclass(frozen=True)
class Triplet:
    """One aligned batch of rollouts: the prediction, its own realisation, a sibling, and a stranger."""

    pred: FloatArray
    true: FloatArray
    sibling: FloatArray
    stranger: FloatArray
    t0: int


def build_triplets(
    children: dict[str, FloatArray], art: RolloutArtifact, arm_current: FloatArray, span: int, stride: int
) -> list[Triplet]:
    """Roll the predictor out on every child and pair each rollout with a sibling and a stranger.

    Siblings share a parent snapshot and differ only in noise seed, so ``L(true, sibling)`` is the
    floor; strangers come from another parent, so ``L(true, stranger)`` is the no-information level.
    """
    k = art.priming_steps
    triplets: list[Triplet] = []
    for y in children.values():
        n_parents, n_children, n_samples, _ = y.shape
        flat = y.reshape(-1, n_samples, y.shape[-1])
        u = np.tile(arm_current, (flat.shape[0], n_samples, 1))
        sibling = np.roll(y, 1, axis=1).reshape(flat.shape)
        stranger = np.roll(y, 1, axis=0).reshape(flat.shape)
        for t0 in range(k, n_samples - span + 1, stride):
            states = art.prime_many(flat[:, t0 - k : t0], u[:, t0 - k : t0])
            pred = art.rollout_many(states, u[:, t0 : t0 + span])
            window = slice(t0, t0 + span)
            triplets.append(Triplet(pred, flat[:, window], sibling[:, window], stranger[:, window], t0))
        print(f"  rolled out {n_parents * n_children} children x {len(triplets)} start offsets", flush=True)
    return triplets


def discriminability(signal: FloatArray, floor: FloatArray) -> float:
    """Separation between the model's score and a perfect model's, in pooled standard deviations."""
    spread = np.sqrt(0.5 * (signal.var(ddof=1) + floor.var(ddof=1)))
    return float((signal.mean() - floor.mean()) / spread) if spread > 0 else float("inf")


def rank_separation(signal: FloatArray, floor: FloatArray) -> float:
    """Probability that a scored rollout is ranked worse than a floor pair, ties counted half.

    The rank twin of :func:`discriminability`: both distributions are heavy-tailed, so a mean
    shift in pooled standard deviations understates how reliably the two are told apart.
    """
    order = np.sort(floor)
    below = np.searchsorted(order, signal, side="left")
    ties = np.searchsorted(order, signal, side="right") - below
    return float((below + 0.5 * ties).sum() / (signal.size * floor.size))


def report_discriminability(triplets: list[Triplet], fs: float, n_channels: int) -> None:
    """Print floor, signal and no-information level per candidate geometry, ranked by discriminability."""
    ctx = raw_context(fs, n_channels)
    print(f"\nPer-geometry floor and signal ({triplets[0].pred.shape[0] * len(triplets)} rollouts each)")
    print(f"chi-squared-2 floor of a single unpooled cell: {_CHI2_FLOOR:.2f} nats^2")
    header = f"{'candidate':>20} {'floor':>8} {'signal':>8} {'stranger':>9} {'d':>6} {'AUC':>6} {'AUC_str':>8}"
    print(header)
    rows = []
    for candidate in candidates():
        span = candidate.spec.span_steps(fs)
        built = candidate.build(fs)
        usable = [t for t in triplets if t.pred.shape[1] >= span]
        floor = np.concatenate([per_sample(built, t.sibling, t.true, ctx) for t in usable])
        signal = np.concatenate([per_sample(built, t.pred, t.true, ctx) for t in usable])
        stranger = np.concatenate([per_sample(built, t.stranger, t.true, ctx) for t in usable])
        rows.append(
            (
                candidate.name,
                float(floor.mean()),
                float(signal.mean()),
                float(stranger.mean()),
                discriminability(signal, floor),
                rank_separation(signal, floor),
                rank_separation(stranger, floor),
            )
        )
    for name, floor_m, signal_m, stranger_m, d, auc, auc_str in sorted(rows, key=lambda r: -r[5]):
        print(f"{name:>20} {floor_m:>8.2f} {signal_m:>8.2f} {stranger_m:>9.2f} {d:>6.2f} {auc:>6.3f} {auc_str:>8.3f}")


def report_hinge_split(triplets: list[Triplet], fs: float, n_channels: int, envelope_path: Path) -> None:
    """Split the squared log residual by whether the controller's hinge is active on that cell.

    Scored on the envelope's own geometry, since the mask is only defined where the envelope is.
    """
    envelope = PsdEnvelope.load(envelope_path)
    reference = torch.as_tensor(envelope.power[None, :, None, 1:], dtype=torch.float64)
    ctx = raw_context(fs, n_channels)
    spec = StftSpec(weight=1.0, n_span=envelope.window, n_segment=envelope.window, n_hop=envelope.window)
    built = StftLoss.from_spec(spec, fs)

    def split(a: FloatArray, b: FloatArray) -> tuple[float, float, float]:
        residual = built.log_spectrogram(torch.as_tensor(a), ctx) - built.log_spectrogram(torch.as_tensor(b), ctx)
        power = spectrogram(
            torch.as_tensor(b[:, : envelope.window]).transpose(1, 2), envelope.window, envelope.window, fs=fs
        )
        active = power[..., 1:] > reference
        squared = residual**2
        return (
            float(squared[active].mean()),
            float(squared[~active].mean()),
            float(active.to(torch.float64).mean()),
        )

    print(f"\nHinge split on the envelope geometry (W={envelope.window}), active vs inactive squared residual")
    print(f"{'pairing':>10} {'active':>9} {'inactive':>9} {'frac active':>12}")
    for label, getter in (("floor", lambda t: t.sibling), ("signal", lambda t: t.pred)):
        stats = np.mean([split(getter(t), t.true) for t in triplets], axis=0)
        print(f"{label:>10} {stats[0]:>9.2f} {stats[1]:>9.2f} {stats[2]:>12.3f}")


def report_floor_vs_offset(triplets: list[Triplet], fs: float, n_channels: int) -> None:
    """Print how the floor moves with time since the branch, which bounds how tight the floor is."""
    ctx = raw_context(fs, n_channels)
    spec = StftSpec(weight=1.0, n_span=50, n_segment=25, n_hop=12)
    built = StftLoss.from_spec(spec, fs)
    print(f"\nFloor vs time since the branch (W25 H12 span 50)\n{'t0 (s)':>11} {'floor':>9}")
    by_offset: dict[int, list[float]] = {}
    for triplet in triplets:
        by_offset.setdefault(triplet.t0, []).extend(per_sample(built, triplet.sibling, triplet.true, ctx))
    for offset, values in sorted(by_offset.items()):
        print(f"{offset / fs:>11.2f} {np.mean(values):>9.3f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the STFT geometry probe."""
    parser = argparse.ArgumentParser(
        description="Measure frame-axis correlation width and per-geometry discriminability, without training."
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Predictor artifact basename.")
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument("--ensemble", type=Path, required=True, help="Branch-ensemble directory with an eeg/ store.")
    parser.add_argument("--arm", default="zero", help="Stimulation arm whose stores are scored.")
    parser.add_argument("--span", type=int, default=75, help="Longest candidate span in steps.")
    parser.add_argument("--stride", type=int, default=15, help="Stride between rollout start offsets, in steps.")
    parser.add_argument("--max-lag", type=int, default=60, help="Largest frame lag in the autocorrelation.")
    parser.add_argument(
        "--envelope", type=Path, default=Path("data/healthy_psd.npz"), help="Healthy envelope for the hinge split."
    )
    parser.add_argument("--seed", type=int, default=69, help="Seed of the surrogate phase randomisation.")
    return parser.parse_args()


def main() -> None:
    """Run the section 8 measurement: correlation width, then floor and signal per geometry."""
    args = parse_args()
    art = load_rollout_artifact(args.artifact)
    fs = 1.0 / art.dt

    report_correlation_width(args.data, art, args.max_lag, args.seed)

    manifest = json.loads((args.ensemble / "manifest.json").read_text(encoding="utf-8"))
    arm_current = next(a["current"] for a in manifest["arms"] if a["name"] == args.arm)
    print(f"\nLoading {args.arm} children from {args.ensemble}...", flush=True)
    children = load_children(args.ensemble, args.arm, art)
    triplets = build_triplets(children, art, np.asarray(arm_current, dtype=np.float64), args.span, args.stride)

    report_floor_vs_offset(triplets, fs, art.n_channels)
    report_discriminability(triplets, fs, art.n_channels)
    report_hinge_split(triplets, fs, art.n_channels, args.envelope)


if __name__ == "__main__":
    main()
