from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.signal import detrend

from neuro.filtering import causal_filter, design_bandpass_sos, design_lowpass_sos, group_delay_s
from neuro.seizure_calibration import SEIZURE_PTP_MV, SPREAD_WINDOW_S
from utils.processing import band_energy, compute_psd

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
    reduce: Callable[[FloatArray, float], FloatArray],
    *,
    window_s: float,
    hop_s: float = DEFAULT_HOP_S,
) -> tuple[FloatArray, FloatArray]:
    """Slide a **trailing** window over ``signals`` and reduce each block per channel.

    Windows are causal: the point reported at time ``t`` is computed from the samples in
    ``(t - window_s, t]``, and ``t`` is the instant the value first becomes available.

    Parameters
    ----------
    signals
        Input signals, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in Hz.
    reduce
        Maps one ``(n_channels, n_window)`` block and ``fs`` to one value per channel.
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
        The reduction of each window, shape ``(n_channels, n_windows)``.
    """
    n_window = round(window_s * fs)
    n_hop = round(hop_s * fs)
    starts = np.arange(0, signals.shape[1] - n_window + 1, n_hop)

    times = (starts + n_window) / fs
    columns = [np.asarray(reduce(signals[:, s : s + n_window], fs), dtype=np.float64) for s in starts]
    values = np.stack(columns, axis=-1) if columns else np.empty((signals.shape[0], 0), dtype=np.float64)
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
        The per-window reduction, ``(block, fs) -> (n_channels,)``.

    Reductions are **per channel**, and pooling to a scalar is a downstream choice
    (:meth:`pooled`). Metrics are read off a channel at a time because that is how they are
    defined, and collapsing them at source would throw away the spatial structure a focal
    seizure lives in.
    """

    name: str
    window_s: float
    reduce: Callable[[FloatArray, float], FloatArray]

    def __call__(
        self,
        signals: FloatArray,
        fs: float,
        *,
        hop_s: float = DEFAULT_HOP_S,
    ) -> tuple[FloatArray, FloatArray]:
        """Score ``signals`` per channel, returning ``(times, (n_channels, n_windows))``."""
        return windowed(signals, fs, self.reduce, window_s=self.window_s, hop_s=hop_s)


def _block_ptp(block: FloatArray, _fs: float) -> FloatArray:
    """Per-channel peak-to-peak amplitude of the block."""
    return np.ptp(block, axis=1)


def _line_length(block: FloatArray, _fs: float) -> FloatArray:
    """Per-channel mean absolute sample-to-sample difference -- line length as a density."""
    return np.abs(np.diff(block, axis=1)).mean(axis=1)


def _eeg_ms(block: FloatArray, _fs: float) -> FloatArray:
    """Per-channel mean square -- the incumbent MPC objective, before pooling."""
    return (block**2).mean(axis=1)


def _band_power(block: FloatArray, fs: float) -> FloatArray:
    """Per-channel un-normalised PSD integral over :data:`SEIZURE_BAND_HZ`."""
    return band_energy(block, 1000.0 / fs, band=SEIZURE_BAND_HZ, normalize=False)


def _fc_strength(block: FloatArray, _fs: float) -> FloatArray:
    """Per-channel FC node strength: mean ``|correlation|`` with the other channels.

    The magnitude is taken because the EEG forward operator is signed, so two channels
    straddling a source see the *same* synchronous activity with opposite sign. Averaging the
    signed correlation -- as :func:`utils.processing.synchronization` does -- cancels exactly
    the synchrony a seizure is characterised by.

    Pooling this over channels gives the magnitude analogue of that scalar, so the metric is
    both a per-channel quantity and a drop-in for the network synchrony index.
    """
    if block.shape[0] < 2:  # noqa: PLR2004
        return np.full(block.shape[0], np.nan)
    corr = np.abs(np.corrcoef(block))
    np.fill_diagonal(corr, np.nan)
    return np.nanmean(corr, axis=1)


def _spectral_centroid(block: FloatArray, fs: float) -> FloatArray:
    """Per-channel PSD-weighted mean frequency in Hz over :data:`CENTROID_BAND_HZ`.

    Bounded well below Nyquist on purpose: the plant's noise is white on ``x5'``, so a
    full-band centroid at 10 kHz would track that tail rather than any physiology.
    """
    freqs, psd = compute_psd(detrend(block, axis=1), 1000.0 / fs)
    low, high = CENTROID_BAND_HZ
    mask = (freqs >= low) & (freqs <= high)
    weight = psd[:, mask].sum(axis=1)
    return np.divide(
        psd[:, mask] @ freqs[mask],
        weight,
        out=np.zeros_like(weight),
        where=weight > 0,
    )


METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric("block_ptp", 0.1, _block_ptp),
        Metric("line_length", 0.1, _line_length),
        Metric("eeg_ms", 0.1, _eeg_ms),
        Metric("band_power", 0.5, _band_power),
        Metric("fc_strength", 0.5, _fc_strength),
        Metric("spectral_centroid", 0.5, _spectral_centroid),
    )
}


def seizure_state(
    region_lfp: FloatArray,
    fs: float,
    *,
    window_s: float = SPREAD_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    threshold: float = SEIZURE_PTP_MV,
) -> tuple[FloatArray, FloatArray]:
    """Fraction of regions seizing over time -- the ground truth the metrics are scored against.

    The network reduction of :class:`~neuro.seizure.SpreadProfile`'s criterion, on the causal
    grid of :func:`windowed` rather than that class's window-*centre* one, so it lines up sample
    for sample with a metric read off the scalp at the same instant.

    ``region_lfp`` is expected to start ``window_s`` before the branch, so the first window ends
    at the branch and the returned times are branch-referenced, starting at 0.

    ``threshold`` and ``window_s`` are calibrated together -- :data:`~neuro.seizure.SEIZURE_PTP_MV`
    is the peak-to-peak a seizing region reaches over :data:`~neuro.seizure.SPREAD_WINDOW_S`, and a
    shorter window measures less of it, so moving one without the other re-tunes the criterion.

    Parameters
    ----------
    region_lfp
        Region LFP ``y = x2 - x3`` in mV, shape ``(n_regions, n_samples)``.
    fs
        Sampling rate of ``region_lfp`` in Hz.
    window_s
        Window length in seconds.
    hop_s
        Spacing between consecutive windows in seconds.
    threshold
        PTP threshold in mV for declaring a region seizing.

    Returns
    -------
    times
        Window-end times in seconds since the branch, starting at 0.
    state
        Fraction of regions above ``threshold`` in each window, in ``[0, 1]``.
    """
    times, ptp = windowed(region_lfp, fs, lambda block, _: np.ptp(block, axis=1), window_s=window_s, hop_s=hop_s)
    return times - window_s, (ptp > threshold).mean(axis=0)


@dataclass(frozen=True)
class Ensemble:
    """One metric's value across one branch and arm, on the lookahead grid.

    Vocabulary, used throughout the scoring layer:

    - **trajectory** -- one independent realisation of the disease course (one plant seed);
    - **state** ``x0`` -- a trajectory frozen at a branch time, the thing a predictor conditions on;
    - **replicate** -- one noise realisation rolled out from a state.

    Rows are ordered ``trajectory * n_replicates + replicate``, so one branch and arm has one
    state per trajectory and :attr:`by_state` recovers the grouping.

    Attributes
    ----------
    times
        Lookahead `h` since the branch, in seconds, shape ``(n_times,)``.
    values
        Metric value per rollout, shape ``(n_states * n_replicates, ..., n_times)``, in the
        store's row order. Trailing axes are free: a pooled metric is ``(n_rollouts, n_times)``
        and a per-channel one ``(n_rollouts, n_channels, n_times)``.
    n_replicates
        Rollouts per state, which is what makes the state grouping recoverable.
    """

    times: FloatArray
    values: FloatArray
    n_replicates: int

    def __post_init__(self) -> None:
        """Reject a rollout count that is not a whole number of equal-sized states."""
        if self.values.shape[0] % self.n_replicates:
            msg = (
                f"{self.values.shape[0]} rollouts is not a whole number of states "
                f"at {self.n_replicates} replicates each"
            )
            raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Distinct branch states in this ensemble, one per trajectory."""
        return self.values.shape[0] // self.n_replicates

    @property
    def by_state(self) -> FloatArray:
        """Values regrouped as ``(n_states, n_replicates, ...)``, trailing axes preserved."""
        return self.values.reshape(self.n_states, self.n_replicates, *self.values.shape[1:])


def variance_ratio(values: FloatArray, n_replicates: int) -> FloatArray:
    """Share of variance that knowing the state removes, over the leading rollout axis.

    ``1 - Var_replicate / Var_total``: the numerator is what is still unknown once ``x0`` is
    fixed, the denominator what was unknown before. This is a ceiling on any predictor, and
    being model-free it does not depend on the state of the predictor pipeline.

    Trailing axes are carried through untouched, so the one formula serves a pooled metric
    series ``(n_rollouts, n_times)``, a per-channel one ``(n_rollouts, n_channels, n_times)``
    and the raw-signal baselines alike.

    Population variances (``ddof=0``) on purpose: the design is balanced, so within-state and
    between-state variance then sum exactly to the total and the score lands in ``[0, 1]``.
    Use :func:`sigma_ens` for the spread as an *estimate* in native units.
    """
    by_state = values.reshape(values.shape[0] // n_replicates, n_replicates, *values.shape[1:])
    within = by_state.var(axis=1).mean(axis=0)
    total = values.var(axis=0)
    return 1.0 - np.divide(within, total, out=np.full_like(total, np.nan), where=total > 0)


def sigma_ens(ens: Ensemble) -> FloatArray:
    """Within-state standard deviation in the metric's native units, per lookahead.

    The spread that survives knowing the branch state -- the noise a causal controller cannot
    resolve, and so the honest denominator for :func:`controllability`.
    """
    return np.sqrt(ens.by_state.var(axis=1, ddof=1).mean(axis=0))


class StateReadout(NamedTuple):
    """How much of a metric's variation the seizure state accounts for.

    Attributes
    ----------
    r2
        ``1 - E[Var(M | s)] / Var(M)``: the share of the metric's variance that knowing the
        seizure state removes. Trailing axes of the metric are carried through, so a
        per-channel metric gives one score per channel.
    explained_var
        ``Var(M) - E[Var(M | s)]`` in the metric's **native units squared** -- the numerator's
        counterpart, and the stated denominator :func:`state_predictability_r2` references a
        forecast error against.
    bin_state
        Mean seizure state in each occupied bin, shape ``(n_filled,)`` -- at most ``n_bins``.
    bin_metric
        Mean metric value in each bin, trailing axes carried through as in ``r2`` -- with
        ``bin_state``, the calibration curve.
    """

    r2: FloatArray
    explained_var: FloatArray
    bin_state: FloatArray
    bin_metric: FloatArray


def state_readout_r2(metric: FloatArray, state: FloatArray, *, n_bins: int = 10) -> StateReadout:
    """Score how well a metric reads the seizure state, by conditional variance reduction.

    Binned rather than fitted, so no functional form is imposed: a metric that tracks the state
    non-monotonically is scored on what it resolves, not on how linear it is. Bins are
    **quantiles** of ``state``, so each carries a comparable number of observations however the
    states are distributed -- which they are not evenly, the branches being clustered.

    ``state`` carries a point mass at zero -- every pre-onset window -- so several quantile edges
    coincide there. Duplicate edges are dropped rather than kept as empty bins, and ``n_bins`` is
    therefore an upper bound: the returned curve has one point per *distinct* edge.

    This is the answer to "does the controller's observable see the seizure at all", and it is
    the axis that :func:`separability`'s two-point Cohen's d approximates with only the healthy
    and saturated ends.

    Parameters
    ----------
    metric
        Metric values, shape ``(n_obs, ...)``, pooled over rollouts and times.
    state
        Matching seizure state per observation, shape ``(n_obs,)``.
    n_bins
        Quantile bins of ``state`` to condition on.
    """
    edges = np.unique(np.quantile(state, np.linspace(0.0, 1.0, n_bins + 1)))
    n_filled = len(edges) - 1
    index = np.clip(np.searchsorted(edges[1:-1], state, side="right"), 0, n_filled - 1)

    counts = np.bincount(index, minlength=n_filled)
    occupied = counts > 0
    grouped = [metric[index == b] for b in range(n_filled)]
    unfilled = np.full(metric.shape[1:], np.nan)

    within = sum((counts[b] / len(state)) * grouped[b].var(axis=0) for b in range(n_filled) if occupied[b])
    total = metric.var(axis=0)
    return StateReadout(
        r2=1.0 - np.divide(within, total, out=np.full_like(total, np.nan), where=total > 0),
        explained_var=total - within,
        bin_state=np.array([state[index == b].mean() if occupied[b] else np.nan for b in range(n_filled)]),
        bin_metric=np.stack([grouped[b].mean(axis=0) if occupied[b] else unfilled for b in range(n_filled)]),
    )


class Separability(NamedTuple):
    """How far apart the healthy and saturated branches sit on this metric.

    Attributes
    ----------
    cohens_d
        Standardised healthy-to-saturated difference over the **state means**, so ``n`` is the
        state count; replicates of a state are not independent draws from a class.
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
    h = healthy.by_state.mean(axis=1)
    s = saturated.by_state.mean(axis=1)
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

    ``zero`` and ``stim`` must be row-aligned -- same trajectories, replicates and seeds -- so
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


def state_predictability_r2(ens: Ensemble, explained_var: FloatArray | float) -> FloatArray:
    """Forecast error at each lookahead, referenced to the metric's seizure-state range.

    ``1 - Var_replicate(h) / explained_var``. The numerator is the spread that survives fixing
    ``x0``, which is the irreducible forecast error, against a **stated** denominator instead of
    an incidental one. It is :func:`sigma_ens` squared, so ``ddof=1`` rather than
    :func:`predictability_r2`'s ``ddof=0``: nothing here has to sum to a total, so the unbiased
    estimate is the right one and the two scores differ slightly beyond their denominators.
    ``predictability_r2`` divides by the
    spread of whatever states this branch's trajectories happened to land in, so its value is
    set by the trajectory draw and is not comparable between branches; ``explained_var`` from
    :func:`state_readout_r2` is the same number at every branch.

    That also makes the score operational rather than descriptive: it asks whether the forecast
    error is small against the state difference a controller is trying to steer between, which
    is what an MPC needs, rather than whether the metric is accurately forecastable in its own
    units, which nothing needs.

    Unbounded below, and negative is a real answer: the metric's noise floor is wider than the
    span of seizure states it is supposed to distinguish.

    ``explained_var`` carries the metric's trailing axes and no lookahead axis -- a scalar for a
    pooled metric, one value per channel for a per-channel one -- and the score is shaped like
    :func:`sigma_ens`, one curve over `h` per trailing axis.
    """
    var_rep = sigma_ens(ens) ** 2
    # the denominator is one number per channel, so it needs the lookahead axis the score runs over
    denominator = np.asarray(explained_var)[..., np.newaxis]
    return 1.0 - np.divide(
        var_rep,
        denominator,
        out=np.full_like(var_rep, np.nan),
        where=denominator > 0,
    )


def state_predictability_snr_db(ens: Ensemble, explained_var: FloatArray | float) -> FloatArray:
    """Predictability signal-to-noise ratio in dB, referenced to seizure-state range.

    ``10 * log10(explained_var / Var_replicate(h))``
    """
    var_rep = sigma_ens(ens) ** 2
    denominator = np.asarray(explained_var)[..., np.newaxis]
    valid = (var_rep > 0) & (denominator > 0)
    ratio = np.divide(denominator, var_rep, out=np.full_like(var_rep, np.nan), where=valid)
    return np.multiply(10.0, np.log10(ratio, out=np.full_like(ratio, np.nan), where=valid & (ratio > 0)))


def coupling(zero_metric: Ensemble, stim_metric: Ensemble, zero_state: Ensemble, stim_state: Ensemble) -> FloatArray:
    """Correlate the per-rollout metric shift against the per-rollout seizure-state shift.

    ``corr_i(delta M_i, delta s_i)`` over rollouts, where each ``delta`` is the paired
    stimulated-minus-unstimulated difference of one noise realisation. Positive means the
    rollouts in which stimulation pushed the metric furthest are the rollouts in which it
    pushed the seizure furthest. ``d_ctrl`` scores that a metric *can* be driven and ``d_state``
    that the seizure *can* be suppressed; only their covariation says the first buys the second.

    Both deltas are centred, so this is blind to the mean effect that ``d_ctrl`` and ``d_state``
    already report: a command shifting every rollout by the same constant scores ``NaN``, not
    zero. Read it **with** those two, never alone -- alone it cannot tell a metric that tracks
    the seizure from one whose trial-to-trial jitter merely shares a noise source.

    The state is one scalar per rollout, so a per-channel metric scores one correlation per
    channel. All four ensembles must be row-aligned -- same trajectories, replicates and seeds.
    """
    delta_m = stim_metric.values - zero_metric.values
    delta_s = stim_state.values - zero_state.values
    # the state is one scalar per rollout, so a per-channel metric needs it broadcast over channels
    delta_s = delta_s.reshape(delta_s.shape[0], *(1,) * (delta_m.ndim - delta_s.ndim), delta_s.shape[-1])

    a = delta_m - delta_m.mean(axis=0)
    b = delta_s - delta_s.mean(axis=0)
    norm = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    return np.divide((a * b).sum(axis=0), norm, out=np.full_like(norm, np.nan), where=norm > 0)


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


def envelope_latency_s(
    fs: float,
    *,
    band: tuple[float, float] = SEIZURE_BAND_HZ,
    smooth_hz: float = ENVELOPE_SMOOTH_HZ,
) -> float:
    """Group delay of :func:`envelope`'s two causal filters -- its analogue of a metric window."""
    return group_delay_s(design_bandpass_sos(fs, band), fs) + group_delay_s(design_lowpass_sos(fs, smooth_hz), fs)


@dataclass(frozen=True)
class RawSeries:
    """An observable read straight off the rollout rather than reduced over a window.

    The window-free counterpart of :class:`Metric`, and scored on exactly the same axes: these are
    the signals the existing predictor pipeline is trained against, so a candidate metric only
    earns its window by beating them.

    Attributes
    ----------
    name
        Registry key.
    transform
        Causal whole-signal map, ``(signals, fs) -> (n_channels, n_samples)``.
    latency
        Seconds before the first valid sample, given ``fs`` -- what :attr:`Metric.window_s` is for
        a windowed metric.
    """

    name: str
    transform: Callable[[FloatArray, float], FloatArray]
    latency: Callable[[float], float]


RAW_SERIES: dict[str, RawSeries] = {
    series.name: series
    for series in (
        RawSeries("waveform", lambda signals, _fs: signals, lambda _fs: 0.0),
        RawSeries("envelope", envelope, envelope_latency_s),
    )
}


def latency_s(name: str, fs: float) -> float:
    """Seconds of signal an observable needs before its first valid sample, windowed or not."""
    return METRICS[name].window_s if name in METRICS else RAW_SERIES[name].latency(fs)


def raw_series_grid(duration_s: float, *, hop_s: float = DEFAULT_HOP_S) -> FloatArray:
    """Lookahead grid for the window-free series: fine early, the shared metric grid later.

    The plant is chaotic in phase, so waveform predictability is expected to collapse within a
    couple hundred milliseconds -- below the resolution of the shared 50 ms grid, on which it
    would read as a flat zero and look like a bug rather than a result.
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
    n_replicates: int,
    hop_s: float = DEFAULT_HOP_S,
    cutoff_hz: float | None = None,
    pool: bool = True,
) -> tuple[dict[str, FloatArray], dict[tuple[str, str], Ensemble]]:
    """Score every rollout of a ``(n_rollouts, n_channels, n_samples)`` store, in one pass.

    Row-major on purpose: every metric and channel set is read off the same rollout, so a
    memory-mapped 1.9 GB store is traversed once rather than once per metric -- which is what
    makes the bandwidth sweep affordable, since each cutoff re-reads the store.

    ``cutoff_hz`` applies the causal low-pass of the bandwidth axis before scoring; ``None`` is
    the raw signal. Filtering happens before the channel slice, so the two axes stay independent.

    ``pool`` averages each metric over its channel set, giving ``Ensemble.values`` of shape
    ``(n_rollouts, n_times)``; with ``pool=False`` the channel axis is kept and the values are
    ``(n_rollouts, n_channels, n_times)``. Every score function carries trailing axes through
    untouched, so both shapes score identically.

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
                rows[name, set_name].append(values.mean(axis=0) if pool else values)

    return times, {
        key: Ensemble(times=times[key[0]], values=np.stack(values), n_replicates=n_replicates)
        for key, values in rows.items()
    }


def state_store(
    store: FloatArray,
    fs: float,
    *,
    hop_s: float = DEFAULT_HOP_S,
    window_s: float = SPREAD_WINDOW_S,
) -> tuple[FloatArray, FloatArray]:
    """Read the seizure state off every rollout of a ``(n_rollouts, n_regions, n_samples)`` store.

    Each rollout is expected to start ``window_s`` before the branch, per :func:`seizure_state`.

    Returns
    -------
    times
        Lookahead grid in seconds since the branch, starting at 0.
    state
        Fraction of regions seizing, shape ``(n_rollouts, n_times)``.
    """
    times = np.empty(0, dtype=np.float64)
    rows: list[FloatArray] = []
    for row in range(store.shape[0]):
        times, values = seizure_state(np.asarray(store[row], dtype=np.float64), fs, window_s=window_s, hop_s=hop_s)
        rows.append(values)
    return times, np.stack(rows)


def score_raw_store(  # noqa: PLR0913
    store: FloatArray,
    fs: float,
    *,
    times: FloatArray,
    series: Mapping[str, RawSeries],
    channel_sets: Mapping[str, IntArray],
    n_replicates: int,
) -> dict[tuple[str, str], Ensemble]:
    """Sample every window-free observable of a ``(n_rollouts, n_channels, n_samples)`` store.

    The counterpart of :func:`score_store` for the observables that are not a windowed reduction:
    each is a causal transform of the whole rollout, read at ``times`` rather than over a window.
    They come out as :class:`Ensemble` like any metric, so ``waveform`` -- what the predictor
    pipeline is actually trained on -- is scored on the same axes as the candidates that would
    replace it, rather than only on predictability.

    Kept per channel, and pooled on read: a channel mean of a signed waveform cancels, so the
    honest reading of that one is per-channel.

    Neither is scored under a low-pass. The envelope's band is fixed at :data:`SEIZURE_BAND_HZ`,
    so a cutoff redefines it rather than denoising it, and a filtered waveform is a question the
    bandwidth sweep asks of the metrics that survive it.
    """
    rows: dict[tuple[str, str], list[FloatArray]] = {(n, s): [] for n in series for s in channel_sets}

    for row in range(store.shape[0]):
        signals = np.asarray(store[row], dtype=np.float64)
        for name, raw in series.items():
            sampled = sample_at(raw.transform(signals, fs), fs, times)
            for set_name, channels in channel_sets.items():
                rows[name, set_name].append(sampled[channels])

    return {
        key: Ensemble(times=times, values=np.stack(values), n_replicates=n_replicates) for key, values in rows.items()
    }
