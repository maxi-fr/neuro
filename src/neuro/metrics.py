from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.signal import detrend

from neuro.filtering import causal_filter, design_bandpass_sos, design_lowpass_sos
from utils.processing import band_energy, compute_psd, synchronization

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from neuro.types import FloatArray, IntArray

DEFAULT_HOP_S = 0.05
SEIZURE_BAND_HZ = (3.0, 12.0)
CENTROID_BAND_HZ = (1.0, 45.0)

ENVELOPE_SMOOTH_HZ = 3.0
BASELINE_FINE_HOP_S = 0.005
BASELINE_FINE_UNTIL_S = 0.5


def windowed(
    signals: FloatArray,
    fs: float,
    reduce: Callable[[FloatArray, float], float],
    *,
    window_s: float,
    hop_s: float = DEFAULT_HOP_S,
) -> tuple[FloatArray, FloatArray]:
    """Slide a **trailing** window over ``signals`` and reduce each block to a scalar.

    Windows are causal: the point reported at time ``t`` is computed from the samples in
    ``(t - window_s, t]``, and ``t`` is the instant the value first becomes available. So the
    grid starts at exactly ``window_s`` and a value never depends on the future. (This differs
    from :func:`neuro.seizure.spread_profile`, which timestamps window *centres* -- that is a
    labelling job, not a control one.)

    Parameters
    ----------
    signals
        Input signals, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in Hz.
    reduce
        Maps one ``(n_channels, n_window)`` block and ``fs`` to a scalar.
    window_s
        Window length in seconds.
    hop_s
        Spacing between consecutive windows in seconds.

    Returns
    -------
    times
        Window-end times in seconds since the start of ``signals``, shape ``(n_windows,)``.
        Empty if ``signals`` is shorter than one window.
    values
        The reduction of each window, shape ``(n_windows,)``.
    """
    n_window = round(window_s * fs)
    n_hop = round(hop_s * fs)
    starts = np.arange(0, signals.shape[1] - n_window + 1, n_hop)

    times = (starts + n_window) / fs
    values = np.array([reduce(signals[:, s : s + n_window], fs) for s in starts], dtype=np.float64)
    return times.astype(np.float64), values


@dataclass(frozen=True)
class Metric:
    """A candidate control observable: one windowed scalar read off multi-channel signals.

    Attributes
    ----------
    name
        Registry key.
    window_s
        The metric's own window length in seconds -- also its *feasibility* score, since a
        metric needing a window longer than the control period is unusable whatever else it
        scores.
    reduce
        The per-window reduction, ``(block, fs) -> float``.

    Every reduction pools over channels with a plain mean, so the pooling convention is shared
    and the caller's choice of channel slice is a clean second axis.
    """

    name: str
    window_s: float
    reduce: Callable[[FloatArray, float], float]

    def __call__(
        self,
        signals: FloatArray,
        fs: float,
        *,
        hop_s: float = DEFAULT_HOP_S,
    ) -> tuple[FloatArray, FloatArray]:
        """Score ``signals`` on this metric's own window, returning ``(times, values)``."""
        return windowed(signals, fs, self.reduce, window_s=self.window_s, hop_s=hop_s)


def _block_ptp(block: FloatArray, _fs: float) -> float:
    """Mean over channels of the block's peak-to-peak amplitude."""
    return float(np.mean(np.ptp(block, axis=1)))


def _line_length(block: FloatArray, _fs: float) -> float:
    """Mean absolute sample-to-sample difference -- line length as a per-sample density."""
    return float(np.mean(np.abs(np.diff(block, axis=1))))


def _eeg_ms(block: FloatArray, _fs: float) -> float:
    """Mean square over channels and samples -- the incumbent MPC objective."""
    return float(np.mean(block**2))


def _band_power(block: FloatArray, fs: float) -> float:
    """Mean over channels of the un-normalised PSD integral over :data:`SEIZURE_BAND_HZ`."""
    return float(np.mean(band_energy(block, 1000.0 / fs, band=SEIZURE_BAND_HZ, normalize=False)))


def _synchronization(block: FloatArray, _fs: float) -> float:
    """Mean off-diagonal channel correlation -- the scalar reduction of the FC matrix."""
    return synchronization(block)


def _spectral_centroid(block: FloatArray, fs: float) -> float:
    """PSD-weighted mean frequency in Hz over :data:`CENTROID_BAND_HZ`.

    Bounded well below Nyquist on purpose: the plant's noise is white on ``x5'``, so a
    full-band centroid at 10 kHz would track that tail rather than any physiology.
    """
    freqs, psd = compute_psd(detrend(block, axis=1), 1000.0 / fs)
    low, high = CENTROID_BAND_HZ
    mask = (freqs >= low) & (freqs <= high)
    weight = psd[:, mask].sum(axis=1)
    centroid = np.divide(
        psd[:, mask] @ freqs[mask],
        weight,
        out=np.zeros_like(weight),
        where=weight > 0,
    )
    return float(np.mean(centroid))


METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric("block_ptp", 0.1, _block_ptp),
        Metric("line_length", 0.1, _line_length),
        Metric("eeg_ms", 0.1, _eeg_ms),
        Metric("band_power", 0.5, _band_power),
        Metric("synchronization", 0.5, _synchronization),
        Metric("spectral_centroid", 0.5, _spectral_centroid),
    )
}


@dataclass(frozen=True)
class Ensemble:
    """One metric's value across one branch and arm, on the lookahead grid.

    Attributes
    ----------
    times
        Lookahead `h` since the branch, in seconds, shape ``(n_times,)``.
    values
        Metric value per rollout, shape ``(n_parents * n_children, n_times)``, in the store's
        row order.
    n_children
        Rollouts per parent, which is what makes the parent grouping recoverable.
    """

    times: FloatArray
    values: FloatArray
    n_children: int

    def __post_init__(self) -> None:
        """Reject a rollout count that is not a whole number of equal-sized parents."""
        if self.values.shape[0] % self.n_children:
            msg = f"{self.values.shape[0]} rollouts is not a whole number of parents at {self.n_children} children each"
            raise ValueError(msg)

    @property
    def n_parents(self) -> int:
        """Parents in this ensemble."""
        return self.values.shape[0] // self.n_children

    @property
    def by_parent(self) -> FloatArray:
        """Values regrouped as ``(n_parents, n_children, n_times)``."""
        return self.values.reshape(self.n_parents, self.n_children, -1)


def variance_ratio(values: FloatArray, n_children: int) -> FloatArray:
    """Share of variance that knowing the parent removes, over the leading rollout axis.

    ``1 - Var_within_ensemble / Var_total``: the numerator is what is still unknown once ``x0``
    is fixed, the denominator what was unknown before. This is a ceiling on any predictor, and
    being model-free it does not depend on the state of the predictor pipeline.

    Trailing axes are carried through untouched, so the one formula serves both a scalar metric
    series ``(n_rollouts, n_times)`` and the per-channel raw-signal baselines
    ``(n_rollouts, n_channels, n_times)``.

    Population variances (``ddof=0``) on purpose: the design is balanced, so within-parent and
    between-parent variance then sum exactly to the total and the score lands in ``[0, 1]``.
    Use :func:`sigma_ens` for the spread as an *estimate* in native units.
    """
    by_parent = values.reshape(values.shape[0] // n_children, n_children, *values.shape[1:])
    within = by_parent.var(axis=1).mean(axis=0)
    total = values.var(axis=0)
    return 1.0 - np.divide(within, total, out=np.full_like(total, np.nan), where=total > 0)


def predictability_r2(ens: Ensemble) -> FloatArray:
    """Share of the metric's variance that knowing the parent state removes, per lookahead."""
    return variance_ratio(ens.values, ens.n_children)


def sigma_ens(ens: Ensemble) -> FloatArray:
    """Within-ensemble standard deviation in the metric's native units, per lookahead.

    The spread that survives knowing the branch state -- the noise a causal controller cannot
    resolve, and so the honest denominator for :func:`controllability`.
    """
    return np.sqrt(ens.by_parent.var(axis=1, ddof=1).mean(axis=0))


def spread_reference(ens: Ensemble) -> tuple[FloatArray, FloatArray]:
    """Compute the p5 and p95 of the metric across all rollouts, as a scale reference for `sigma_ens`."""
    return (
        np.percentile(ens.values, 5, axis=0),
        np.percentile(ens.values, 95, axis=0),
    )


class Separability(NamedTuple):
    """How far apart the healthy and saturated branches sit on this metric.

    Attributes
    ----------
    cohens_d
        Standardised healthy-to-saturated difference over the **parent means**, so ``n`` is the
        parent count; children within a parent are not independent draws from a class.
    gap
        Raw signed difference ``mean_saturated - mean_healthy`` in native units.
    direction
        ``-1`` where the seizure raises the metric, so lowering it is the recovery direction;
        ``+1`` where the seizure lowers it. Anchored to a measurement rather than assumed --
        a metric can be driven confidently in a direction that is not recovery.
    """

    cohens_d: FloatArray
    gap: FloatArray
    direction: FloatArray


def separability(healthy: Ensemble, saturated: Ensemble) -> Separability:
    """Score a metric's healthy-against-saturated contrast, and read off its recovery direction."""
    h = healthy.by_parent.mean(axis=1)
    s = saturated.by_parent.mean(axis=1)
    n_h, n_s = h.shape[0], s.shape[0]

    pooled = np.sqrt(((n_h - 1) * h.var(axis=0, ddof=1) + (n_s - 1) * s.var(axis=0, ddof=1)) / (n_h + n_s - 2))
    gap = s.mean(axis=0) - h.mean(axis=0)
    return Separability(
        cohens_d=np.divide(gap, pooled, out=np.full_like(gap, np.nan), where=pooled > 0),
        gap=gap,
        direction=-np.sign(gap),
    )


class Controllability(NamedTuple):
    """What a sustained open-loop command does to a metric, per lookahead.

    Attributes
    ----------
    d_ctrl
        Signed ``direction * delta_bar / sigma_ens``; positive means toward healthy. The
        headline score, dimensionless like R2 so the two share commensurate axes.
    delta_bar
        Mean paired difference ``M_stim - M_zero`` in native units.
    paired_sd
        Standard deviation of the per-rollout paired difference. Far more sensitive than
        ``sigma_ens`` because the arms share their noise -- which is what makes it the right
        significance check and the wrong controllability score.
    relative
        Signed fraction of the healthy-to-saturated gap the command closes; ``1.0`` is a full
        traverse toward healthy.
    """

    d_ctrl: FloatArray
    delta_bar: FloatArray
    paired_sd: FloatArray
    relative: FloatArray


def controllability(
    zero: Ensemble,
    stim: Ensemble,
    *,
    direction: FloatArray,
    gap: FloatArray,
) -> Controllability:
    """Score how far a stimulation arm moves a metric against the spread a controller faces.

    ``zero`` and ``stim`` must be row-aligned -- same parents, same children, same seeds -- so
    the difference is paired. The ``sigma_ens`` of the **unstimulated** arm is the denominator:
    a real controller cannot know the noise realisation, so scoring on the much smaller paired
    spread would credit effects no causal controller can act on.
    """
    delta = stim.values - zero.values
    delta_bar = delta.mean(axis=0)
    spread = sigma_ens(zero)
    return Controllability(
        d_ctrl=direction * np.divide(delta_bar, spread, out=np.full_like(delta_bar, np.nan), where=spread > 0),
        delta_bar=delta_bar,
        paired_sd=delta.std(axis=0, ddof=1),
        relative=direction
        * np.divide(delta_bar, np.abs(gap), out=np.full_like(delta_bar, np.nan), where=np.abs(gap) > 0),
    )


def scalp_region_correlation(scalp: Ensemble, region: Ensemble) -> FloatArray:
    """Per-rollout correlation between the scalp and region readings of one metric, over `h`.

    The observability axis: it catches metrics that volume conduction has smeared away by the
    time they reach the scalp. Region space is only ever this reference, never the primary.
    """
    a = scalp.values - scalp.values.mean(axis=1, keepdims=True)
    b = region.values - region.values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), norm, out=np.full(a.shape[0], np.nan), where=norm > 0)


def envelope(
    signals: FloatArray,
    fs: float,
    *,
    band: tuple[float, float] = SEIZURE_BAND_HZ,
    smooth_hz: float = ENVELOPE_SMOOTH_HZ,
) -> FloatArray:
    """Causal amplitude envelope of ``signals`` within ``band``: band-pass, rectify, low-pass.

    A detector rather than a Hilbert transform, and that is the whole point. The analytic signal
    at time ``t`` depends on the entire record, so a Hilbert envelope leaks the future into every
    sample -- the same failure as filtering zero-phase, which would inflate exactly the
    predictability this is a baseline for.

    Being window-free is what distinguishes it from ``band_power``: it is an instantaneous
    amplitude, so its only latency is the group delay of its two filters. ``smooth_hz`` defaults
    to the band's lower edge, which puts the rectified carrier (from ``2 * band[0]`` upward) at
    least an octave into the stop band.

    Scaled by ``pi / 2`` so the result is the tone's amplitude rather than its mean-rectified
    value, which a plain detector would leave a factor ``2 / pi`` short.
    """
    passband = causal_filter(signals, design_bandpass_sos(fs, band))
    return 0.5 * np.pi * causal_filter(np.abs(passband), design_lowpass_sos(fs, smooth_hz))


def baseline_grid(duration_s: float, *, hop_s: float = DEFAULT_HOP_S) -> FloatArray:
    """Lookahead grid for the raw-signal baselines: fine early, the shared metric grid later.

    The plant is chaotic in phase, so waveform predictability is expected to collapse within a
    couple hundred milliseconds -- below the resolution of the shared 50 ms grid, on which the
    baseline would read as a flat zero and look like a bug rather than a result.
    """
    fine = np.arange(BASELINE_FINE_HOP_S, BASELINE_FINE_UNTIL_S, BASELINE_FINE_HOP_S)
    coarse = np.arange(BASELINE_FINE_UNTIL_S, duration_s + 0.5 * hop_s, hop_s)
    return np.concatenate([fine, coarse])


def sample_at(signals: FloatArray, fs: float, times: FloatArray) -> FloatArray:
    """Read ``signals`` at the instants ``times``, on :func:`windowed`'s timestamp convention.

    Sample ``j`` covers ``(j / fs, (j + 1) / fs]`` and is timestamped at its right edge, so the
    sample carrying time ``t`` is at index ``round(t * fs) - 1``.
    """
    return signals[..., np.rint(np.asarray(times) * fs).astype(np.int64) - 1]


def score_store(  # noqa: PLR0913
    store: FloatArray,
    fs: float,
    *,
    metrics: Mapping[str, Metric],
    channel_sets: Mapping[str, IntArray],
    n_children: int,
    hop_s: float = DEFAULT_HOP_S,
    cutoff_hz: float | None = None,
) -> tuple[dict[str, FloatArray], dict[tuple[str, str], Ensemble]]:
    """Score every rollout of a ``(n_rollouts, n_channels, n_samples)`` store, in one pass.

    Row-major on purpose: every metric and channel set is read off the same rollout, so a
    memory-mapped 1.9 GB store is traversed once rather than once per metric -- which is what
    makes the bandwidth sweep affordable, since each cutoff re-reads the store.

    ``cutoff_hz`` applies the causal low-pass of the bandwidth axis before scoring; ``None`` is
    the raw signal. Filtering happens before the channel slice, so the two axes stay independent.

    Returns
    -------
    times
        Lookahead grid per metric, keyed by metric name.
    ensembles
        One :class:`Ensemble` per ``(metric, channel_set)``.
    """
    sos = None if cutoff_hz is None else design_lowpass_sos(fs, cutoff_hz)
    times: dict[str, FloatArray] = {}
    rows: dict[tuple[str, str], list[FloatArray]] = {(m, s): [] for m in metrics for s in channel_sets}

    for row in range(store.shape[0]):
        signals = np.asarray(store[row], dtype=np.float64)
        if sos is not None:
            signals = causal_filter(signals, sos)
        for set_name, channels in channel_sets.items():
            block = signals[channels]
            for name, metric in metrics.items():
                times[name], values = metric(block, fs, hop_s=hop_s)
                rows[name, set_name].append(values)

    return times, {
        key: Ensemble(times=times[key[0]], values=np.stack(values), n_children=n_children)
        for key, values in rows.items()
    }


def baseline_r2(
    store: FloatArray,
    fs: float,
    *,
    times: FloatArray,
    n_children: int,
    cutoff_hz: float | None = None,
) -> dict[str, FloatArray]:
    """Per-channel predictability of the signal itself, as the reference the metrics are ranked against.

    The rungs the scored metrics sit above: ``waveform`` bounds a predictor forecasting ``y(t)``
    sample-wise, which is what the existing MLP/ESN pipeline does, and ``envelope`` bounds one
    forecasting band amplitude with phase discarded. The gap between them is the cost of phase
    divergence, which is the number that says whether moving the objective off the waveform buys
    anything at all.

    ``envelope`` is returned only for the raw signal: its band is fixed at
    :data:`SEIZURE_BAND_HZ`, so the bandwidth axis would redefine it rather than filter it, and
    recomputing it per cutoff would be waste.

    Returns
    -------
    dict[str, FloatArray]
        R2 of shape ``(n_channels, n_times)`` per baseline name. Channel sets are applied
        afterwards by averaging rows, so scoring never has to run twice.
    """
    sos = None if cutoff_hz is None else design_lowpass_sos(fs, cutoff_hz)
    waveform: list[FloatArray] = []
    band: list[FloatArray] = []

    for row in range(store.shape[0]):
        raw = np.asarray(store[row], dtype=np.float64)
        waveform.append(sample_at(raw if sos is None else causal_filter(raw, sos), fs, times))
        if sos is None:
            band.append(sample_at(envelope(raw, fs), fs, times))

    scores = {"waveform": variance_ratio(np.stack(waveform), n_children)}
    if sos is None:
        scores["envelope"] = variance_ratio(np.stack(band), n_children)
    return scores
