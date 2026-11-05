from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from scipy.signal import sosfilt

from neuro.filtering import design_lowpass_sos
from neuro.plant.seizure_calibration import SEIZURE_PTP_MV, SPREAD_WINDOW_S
from neuro.spectral import (
    HealthyReference,
    ObservableEnvelope,
    compute_forecast_log_power_frames,
    compute_log_power_frames,
    compute_segment_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.config import StftGeometry
    from neuro.types import BoolArray, FloatArray, IntArray

from observable_predictability.metrics import SEIZURE_BAND_HZ, RolloutIdentity, windowed

DEFAULT_FRAME_HOP_S = 0.06
DEFAULT_WAVEFORM_FS = 50.0
DEFAULT_CUTOFF_HZ = 25.0
HEALTHY_SPECTRAL_QUANTILE = 0.90
CONTROL_HORIZONS_S: tuple[float, ...] = (0.60, 0.84, 1.02, 1.20, 1.50)
REPRESENTATIONS: tuple[str, ...] = ("waveform", "spectral", "connectivity")
CONNECTIVITY_CANDIDATE_WINDOWS_S: tuple[float, ...] = (1.02, 1.50, 2.04, 3.00)
MIN_CONNECTIVITY_SEGMENTS: int = 4
DEFAULT_PSD_FLOOR: float = 1e-12
CONNECTIVITY_VALID_PAIR_THRESHOLD: float = 0.80


def baseline_stft_geometry() -> StftGeometry:
    """Return the baseline 50 Hz Observable STFT geometry for the November 2026 protocol."""
    from neuro.config import StftGeometry  # noqa: PLC0415 -- deferred to prevent circular import with neuro.config

    return StftGeometry(
        n_segment=25,
        n_hop=3,
        band_hz=(1.0, 25.0),
        n_bin_pool=1,
        kernel="linear",
        kernel_width=1,
        window="hann",
        asymmetric_window=True,
    )


def condition_waveform_pair(
    continuation_raw: FloatArray,
    fs_raw: float,
    *,
    history_raw: FloatArray | None = None,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    target_fs: float = DEFAULT_WAVEFORM_FS,
) -> tuple[FloatArray | None, FloatArray]:
    """Causally low-pass filter and decimate ``(history_raw, continuation_raw)`` on the branch-aligned ``target_fs`` grid."""
    cont = np.asarray(continuation_raw, dtype=np.float64)
    if cont.ndim != 2:  # noqa: PLR2004 -- waveform arrays are (steps, n_channels)
        msg = f"continuation_raw must have shape (steps, n_channels), got {cont.shape}."
        raise ValueError(msg)
    hist = None if history_raw is None else np.asarray(history_raw, dtype=np.float64)
    if hist is not None and (hist.ndim != 2 or hist.shape[1] != cont.shape[1]):  # noqa: PLR2004 -- (history_steps, n_channels)
        msg = f"history_raw shape {hist.shape} does not match continuation_raw channels ({cont.shape[1]})."
        raise ValueError(msg)

    downsample = max(1, round(fs_raw / target_fs))
    if not np.isclose(fs_raw / downsample, target_fs, rtol=1e-6):
        msg = f"fs_raw ({fs_raw:g} Hz) is not an integer multiple of target_fs ({target_fs:g} Hz)."
        raise ValueError(msg)

    if downsample == 1 and cutoff_hz >= 0.5 * fs_raw:
        return hist, cont

    effective_cutoff = min(cutoff_hz, 0.5 * fs_raw / downsample) if downsample > 1 else cutoff_hz
    sos = design_lowpass_sos(fs_raw, effective_cutoff)
    zi = np.zeros((sos.shape[0], 2, cont.shape[1]), dtype=np.float64)
    if hist is not None and hist.shape[0] > 0:
        hist_filt, zi = sosfilt(sos, hist, axis=0, zi=zi)
        hist_dec = np.asarray(hist_filt[(hist.shape[0] - 1) % downsample :: downsample], dtype=np.float64)
    else:
        hist_dec = np.empty((0, cont.shape[1]), dtype=np.float64) if hist is not None else None

    cont_filt, _ = sosfilt(sos, cont, axis=0, zi=zi)
    cont_dec = np.asarray(cont_filt[downsample - 1 :: downsample], dtype=np.float64)
    return hist_dec, cont_dec


def extract_spectral_frames(  # noqa: PLR0913 -- continuation, fs, history, geometry, cutoff_hz, and target_fs
    continuation: FloatArray,
    fs: float = DEFAULT_WAVEFORM_FS,
    *,
    history: FloatArray | None = None,
    geometry: StftGeometry | None = None,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    target_fs: float = DEFAULT_WAVEFORM_FS,
) -> tuple[FloatArray, FloatArray]:
    """Extract causal spectral log-power Frames across the branch boundary.

    Parameters
    ----------
    continuation
        Post-branch EEG array of shape ``(steps, n_channels)`` at ``fs``.
    fs
        Input sampling frequency in Hz.
    history
        Optional pre-branch EEG history of shape ``(history_steps, n_channels)`` at ``fs``.
    geometry
        Observable STFT geometry; defaults to :func:`baseline_stft_geometry`.
    cutoff_hz
        Anti-aliasing low-pass cutoff in Hz when conditioning from ``fs > target_fs``.
    target_fs
        Conditioned EEG sampling frequency in Hz.

    Returns
    -------
    times
        Frame endpoint times in seconds, shape ``(n_frames,)``.
    frames
        Log-power Frames of shape ``(n_frames, n_channels, n_bins)``.
    """
    geom = baseline_stft_geometry() if geometry is None else geometry
    hist_50, cont_50 = condition_waveform_pair(
        continuation,
        fs,
        history_raw=history,
        cutoff_hz=cutoff_hz,
        target_fs=target_fs,
    )
    support = geom.sample_support_steps(target_fs)
    if hist_50 is not None:
        frames = compute_forecast_log_power_frames(hist_50, cont_50, geom, fs=target_fs)
        times = np.arange(1, frames.shape[0] + 1, dtype=np.float64) * (geom.n_hop / target_fs)
        return times, frames

    frames = compute_log_power_frames(cont_50, geom, fs=target_fs)
    times = (np.arange(frames.shape[0], dtype=np.float64) * geom.n_hop + support) / target_fs
    return times, frames


def _orient_lfp_channels_first(
    lfp_continuation: FloatArray,
    lfp_history: FloatArray | None,
    n_window: int,
) -> tuple[FloatArray, FloatArray | None]:
    """Normalize regional LFP arrays to ``(n_regions, n_samples)`` regardless of input axis order."""
    cont = np.asarray(lfp_continuation, dtype=np.float64)
    if cont.ndim != 2:  # noqa: PLR2004 -- 2D (steps, n_regions) or (n_regions, steps)
        msg = f"lfp_continuation must be 2D, got shape {cont.shape}."
        raise ValueError(msg)
    if lfp_history is None:
        if cont.shape[0] >= n_window and (cont.shape[1] < n_window or cont.shape[0] >= cont.shape[1]):
            return cont.T, None
        return cont, None

    hist = np.asarray(lfp_history, dtype=np.float64)
    if hist.ndim != 2:  # noqa: PLR2004 -- 2D history array
        msg = f"lfp_history must be 2D, got shape {hist.shape}."
        raise ValueError(msg)
    if hist.shape[1] == cont.shape[1] and (hist.shape[0] != cont.shape[0] or hist.shape[0] + cont.shape[0] >= n_window):
        return cont.T, hist.T
    if hist.shape[0] == cont.shape[0]:
        return cont, hist
    msg = f"lfp_history shape {hist.shape} is incompatible with lfp_continuation shape {cont.shape}."
    raise ValueError(msg)


def compute_seizure_state(  # noqa: PLR0913 -- LFP continuation, fs, history, hop, window, and threshold
    lfp_continuation: FloatArray,
    fs_lfp: float,
    *,
    lfp_history: FloatArray | None = None,
    hop_s: float = DEFAULT_FRAME_HOP_S,
    window_s: float = SPREAD_WINDOW_S,
    threshold: float = SEIZURE_PTP_MV,
) -> tuple[FloatArray, FloatArray]:
    """Compute causal regional LFP Seizure State across the branch boundary.

    Parameters
    ----------
    lfp_continuation
        Post-branch regional LFP array of shape ``(steps, n_regions)`` or ``(n_regions, steps)``.
    fs_lfp
        Regional LFP sampling frequency in Hz.
    lfp_history
        Optional pre-branch regional LFP history of matching region dimension.
    hop_s
        Spacing between consecutive causal windows in seconds.
    window_s
        Trailing peak-to-peak window duration in seconds.
    threshold
        Regional peak-to-peak seizure threshold in mV.

    Returns
    -------
    times
        Window-end times in seconds, shape ``(n_frames,)``.
    state
        Instantaneous fraction of seizing brain regions in ``[0, 1]``, shape ``(n_frames,)``.
    """
    n_window = round(window_s * fs_lfp)
    n_hop = round(hop_s * fs_lfp)
    cont_rt, hist_rt = _orient_lfp_channels_first(lfp_continuation, lfp_history, n_window)

    if hist_rt is not None:
        prefix_len = max(0, n_window - n_hop)
        if hist_rt.shape[1] < prefix_len:
            msg = f"lfp_history length ({hist_rt.shape[1]}) must be >= {prefix_len} samples at {fs_lfp:g} Hz."
            raise ValueError(msg)
        stitched = np.concatenate([hist_rt[:, -prefix_len:], cont_rt], axis=1) if prefix_len > 0 else cont_rt
        _, ptp = windowed(stitched, fs_lfp, lambda block, _: np.ptp(block, axis=1), window_s=window_s, hop_s=hop_s)
        state = np.asarray((ptp > threshold).mean(axis=0), dtype=np.float64)
        times = np.arange(1, state.shape[0] + 1, dtype=np.float64) * (n_hop / fs_lfp)
        return times, state

    times, ptp = windowed(cont_rt, fs_lfp, lambda block, _: np.ptp(block, axis=1), window_s=window_s, hop_s=hop_s)
    return times, np.asarray((ptp > threshold).mean(axis=0), dtype=np.float64)


def _validate_reference_split(
    identities: Sequence[RolloutIdentity] | None,
    split: str,
) -> None:
    """Enforce that healthy reference fitting uses only ``split='reference'`` healthy runs."""
    if split != "reference":
        msg = f"Healthy references must be fitted exclusively on split='reference', got {split!r}."
        raise ValueError(msg)
    if identities is not None:
        if len(identities) == 0:
            msg = "identities cannot be empty when fitting a healthy reference."
            raise ValueError(msg)
        for ident in identities:
            if ident.split != "reference" or ident.plant != "healthy":
                msg = (
                    "Healthy references must be fitted exclusively on healthy reference runs "
                    f"(plant='healthy', split='reference'), got {ident}."
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class HealthyWaveformReference:
    """Healthy per-channel EEG mean and variance fitted on ``split='reference'`` runs."""

    mean: FloatArray
    var: FloatArray
    std: FloatArray
    fs: float = DEFAULT_WAVEFORM_FS

    @classmethod
    def fit(
        cls,
        reference_waveforms: FloatArray | Sequence[FloatArray],
        *,
        fs: float = DEFAULT_WAVEFORM_FS,
        identities: Sequence[RolloutIdentity] | None = None,
        split: str = "reference",
    ) -> HealthyWaveformReference:
        """Fit per-channel mean and variance from healthy reference 50 Hz Waveform runs.

        Parameters
        ----------
        reference_waveforms
            Healthy reference waveforms of shape ``(steps, n_channels)`` or ``(n_runs, steps, n_channels)``.
        fs
            Sampling rate in Hz.
        identities
            Optional rollout identities validated for ``plant='healthy'`` and ``split='reference'``.
        split
            Dataset split label; must be ``'reference'``.
        """
        _validate_reference_split(identities, split)
        if isinstance(reference_waveforms, np.ndarray):
            arr = np.asarray(reference_waveforms, dtype=np.float64)
            if arr.ndim < 2:  # noqa: PLR2004 -- need at least (steps, n_channels)
                msg = f"reference_waveforms must have at least 2 dimensions, got {arr.shape}."
                raise ValueError(msg)
            flat = arr.reshape(-1, arr.shape[-1])
        else:
            blocks = [np.asarray(w, dtype=np.float64).reshape(-1, np.asarray(w).shape[-1]) for w in reference_waveforms]
            if not blocks:
                msg = "reference_waveforms cannot be empty."
                raise ValueError(msg)
            flat = np.concatenate(blocks, axis=0)

        mean = np.mean(flat, axis=0)
        var = np.var(flat, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        return cls(
            mean=np.asarray(mean, dtype=np.float64),
            var=np.asarray(var, dtype=np.float64),
            std=np.asarray(std, dtype=np.float64),
            fs=float(fs),
        )

    def to_healthy_reference(self) -> HealthyReference:
        """Convert to a :class:`~neuro.spectral.HealthyReference` for MPC problem assembly."""
        return HealthyReference(eeg_mean=self.mean)

    @property
    def scale(self) -> FloatArray:
        """Alias for per-channel standard deviation ``std``."""
        return self.std


@dataclass(frozen=True)
class HealthySpectralReference:
    """Healthy per-bin, per-channel spectral log-power target fitted on ``split='reference'`` runs."""

    mean_log_power: FloatArray
    channel_bin_mean: FloatArray
    envelope_power: FloatArray
    std_log_power: FloatArray
    fs: float
    geometry: StftGeometry

    @classmethod
    def fit(
        cls,
        reference_data: FloatArray | Sequence[FloatArray],
        *,
        fs: float = DEFAULT_WAVEFORM_FS,
        geometry: StftGeometry | None = None,
        identities: Sequence[RolloutIdentity] | None = None,
        split: str = "reference",
    ) -> HealthySpectralReference:
        """Fit mean log power and the MPC healthy envelope from reference Frames.

        Parameters
        ----------
        reference_data
            Pre-extracted log-power Frames of shape ``(..., n_channels, n_bins)`` or 50 Hz Waveforms
            of shape ``(steps, n_channels)`` / ``(n_runs, steps, n_channels)``.
        fs
            Conditioned EEG sampling frequency in Hz.
        geometry
            Observable STFT geometry; defaults to :func:`baseline_stft_geometry`.
        identities
            Optional rollout identities validated for ``plant='healthy'`` and ``split='reference'``.
        split
            Dataset split label; must be ``'reference'``.
        """
        _validate_reference_split(identities, split)
        geom = baseline_stft_geometry() if geometry is None else geometry
        n_bins = geom.n_values(fs)

        if isinstance(reference_data, np.ndarray):
            arr = np.asarray(reference_data, dtype=np.float64)
            if arr.ndim == 2:  # noqa: PLR2004 -- (steps, n_channels) waveform
                frames = compute_log_power_frames(arr, geom, fs=fs)
            elif arr.ndim >= 3 and arr.shape[-1] == n_bins:  # noqa: PLR2004 -- (..., n_channels, n_bins) frames
                frames = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
            elif arr.ndim == 3:  # noqa: PLR2004 -- (n_runs, steps, n_channels) waveforms
                frames = np.concatenate([compute_log_power_frames(run, geom, fs=fs) for run in arr], axis=0)
            else:
                msg = f"Unsupported reference_data shape {arr.shape} for spectral reference fitting."
                raise ValueError(msg)
        else:
            collected: list[FloatArray] = []
            for item in reference_data:
                arr = np.asarray(item, dtype=np.float64)
                if arr.ndim == 2:  # noqa: PLR2004 -- (steps, n_channels)
                    collected.append(compute_log_power_frames(arr, geom, fs=fs))
                elif arr.ndim >= 3 and arr.shape[-1] == n_bins:  # noqa: PLR2004 -- (..., n_channels, n_bins)
                    collected.append(arr.reshape(-1, arr.shape[-2], arr.shape[-1]))
                else:
                    msg = f"Unsupported item shape {arr.shape} in reference_data."
                    raise ValueError(msg)
            if not collected:
                msg = "reference_data cannot be empty."
                raise ValueError(msg)
            frames = np.concatenate(collected, axis=0)

        if frames.shape[0] == 0:
            msg = "reference_data produced 0 spectral Frames."
            raise ValueError(msg)

        channel_bin_mean = np.asarray(np.mean(frames, axis=0), dtype=np.float64)
        channel_bin_std = np.asarray(np.std(frames, axis=0), dtype=np.float64)
        return cls(
            mean_log_power=np.asarray(channel_bin_mean.T, dtype=np.float64),
            channel_bin_mean=channel_bin_mean,
            envelope_power=np.quantile(frames, HEALTHY_SPECTRAL_QUANTILE, axis=0),
            std_log_power=np.asarray(channel_bin_std.T, dtype=np.float64),
            fs=float(fs),
            geometry=geom,
        )

    def to_observable_envelope(self) -> ObservableEnvelope:
        """Convert to an :class:`~neuro.spectral.ObservableEnvelope` for MPC cost assembly."""
        return ObservableEnvelope(power=self.envelope_power, fs=self.fs, geometry=self.geometry)

    @property
    def mean(self) -> FloatArray:
        """Alias for ``mean_log_power`` of shape ``(n_bins, n_channels)``."""
        return self.mean_log_power

    @property
    def scale(self) -> FloatArray:
        """Alias for ``std_log_power`` of shape ``(n_bins, n_channels)``."""
        return self.std_log_power


def compute_waveform_penalty(
    waveform: FloatArray,
    reference: HealthyWaveformReference | HealthyReference | FloatArray,
    *,
    scale: float = 1.0,
    frame_hop: int | None = None,
) -> FloatArray:
    """Compute sample-level or Frame-aggregated causal Waveform suppression penalty.

    Matches :class:`~trajopt.costs.quadratic.DiagonalCost` channel-mean squared tracking deviation
    from the healthy reference operating point.

    Parameters
    ----------
    waveform
        Conditioned 50 Hz Waveform samples of shape ``(..., steps, n_channels)``.
    reference
        Healthy Waveform reference carrying per-channel means ``(n_channels,)``.
    scale
        Multiplicative scale factor (``1.0`` for mean squared deviation, ``0.5`` for ``0.5 * MSE``).
    frame_hop
        When provided (e.g. ``3`` at 50 Hz for 60 ms Frames), average each contiguous ``frame_hop``
        block along the step axis to align with the 60 ms Frame grid.

    Returns
    -------
    FloatArray
        Waveform suppression penalty of shape ``(..., steps)`` or ``(..., n_frames)``.
    """
    y = np.asarray(waveform, dtype=np.float64)
    if isinstance(reference, HealthyWaveformReference):
        mean = reference.mean
    elif isinstance(reference, HealthyReference):
        if reference.eeg_mean is None:
            msg = "HealthyReference contains no eeg_mean vector."
            raise ValueError(msg)
        mean = reference.eeg_mean
    else:
        mean = np.asarray(reference, dtype=np.float64)

    if mean.shape != (y.shape[-1],):
        msg = f"reference mean shape {mean.shape} does not match channel count ({y.shape[-1]})."
        raise ValueError(msg)

    sample_penalty = np.asarray(scale * np.mean((y - mean) ** 2, axis=-1), dtype=np.float64)
    if frame_hop is None:
        return sample_penalty
    if frame_hop < 1:
        msg = f"frame_hop must be >= 1, got {frame_hop}."
        raise ValueError(msg)
    n_frames = sample_penalty.shape[-1] // frame_hop
    trimmed = sample_penalty[..., : n_frames * frame_hop]
    reshaped = trimmed.reshape(*sample_penalty.shape[:-1], n_frames, frame_hop)
    return np.asarray(np.mean(reshaped, axis=-1), dtype=np.float64)


def compute_spectral_penalty(
    frames: FloatArray,
    reference: HealthySpectralReference | ObservableEnvelope | FloatArray,
    *,
    one_sided: bool = False,
    scale: float = 1.0,
) -> FloatArray:
    """Compute Frame-level Spectral suppression penalty against a healthy log-power reference.

    With ``one_sided=False``, computes mean squared log-power deviation from ``reference`` across
    frequency bins and channels (matching quadratic spectral tracking). With ``one_sided=True``,
    computes mean squared positive log-power excess matching :class:`~neuro.control.costs.ObservableHingeCost`
    and :class:`~neuro.control.costs.ObservableFrameHingeCost`.

    Parameters
    ----------
    frames
        Spectral log-power Frames of shape ``(..., n_frames, n_channels, n_bins)`` or
        ``(..., n_frames, n_bins, n_channels)``.
    reference
        Healthy Spectral reference of shape ``(n_bins, n_channels)`` or ``(n_channels, n_bins)``.
    one_sided
        Whether to hinge only positive excess above ``reference``.
    scale
        Multiplicative scale factor (``1.0`` default, ``0.5`` for ``0.5 * MSE``).

    Returns
    -------
    FloatArray
        Frame-level spectral penalty of shape ``(..., n_frames)``.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim < 3:  # noqa: PLR2004 -- (..., n_frames, dim1, dim2)
        msg = f"frames must have at least 3 dimensions (..., n_frames, n_channels, n_bins), got {arr.shape}."
        raise ValueError(msg)

    if isinstance(reference, HealthySpectralReference):
        power = reference.envelope_power if one_sided else reference.channel_bin_mean
        target = power if arr.shape[-2:] == power.shape else power.T
    elif isinstance(reference, ObservableEnvelope):
        target = reference.power if arr.shape[-2:] == reference.power.shape else reference.power.T
    else:
        raw_ref = np.asarray(reference, dtype=np.float64)
        if raw_ref.shape == arr.shape[-2:]:
            target = raw_ref
        elif raw_ref.T.shape == arr.shape[-2:]:
            target = raw_ref.T
        else:
            msg = f"reference shape {raw_ref.shape} is incompatible with frame dimensions {arr.shape[-2:]}."
            raise ValueError(msg)

    diff = np.maximum(0.0, arr - target) if one_sided else (arr - target)
    return np.asarray(scale * np.mean(diff**2, axis=(-2, -1)), dtype=np.float64)


def compute_horizon_penalty(
    times: FloatArray,
    penalty_series: FloatArray,
    horizon_s: float,
    *,
    weight: float = 1.0,
) -> FloatArray:
    """Average a sample- or Frame-level penalty series over ``0 < t <= horizon_s``.

    Parameters
    ----------
    times
        Lookahead timestamps in seconds, shape ``(n_steps,)``.
    penalty_series
        Penalty values along the last axis, shape ``(..., n_steps)``.
    horizon_s
        Control Horizon duration in seconds.
    weight
        Optional MPC weight multiplier ``w``.

    Returns
    -------
    FloatArray
        Horizon-averaged penalty of shape ``penalty_series.shape[:-1]``.
    """
    t = np.asarray(times, dtype=np.float64)
    series = np.asarray(penalty_series, dtype=np.float64)
    mask = (t > 0.0) & (t <= horizon_s + 1e-9)
    if not np.any(mask):
        msg = f"No timestamps lie in (0, {horizon_s:g}] s."
        raise ValueError(msg)
    sliced = series[..., mask]
    valid_counts = np.sum(np.isfinite(sliced), axis=-1)
    sums = np.nansum(sliced, axis=-1)
    out = np.divide(
        sums,
        valid_counts,
        out=np.full(series.shape[:-1], np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    return np.asarray(weight * out, dtype=np.float64)


def connectivity_pair_indices(n_channels: int) -> tuple[tuple[int, int], ...]:
    """Return the ``C * (C - 1) // 2`` unique off-diagonal channel pairs ``(i, j)`` with ``i < j``."""
    if n_channels < 2:  # noqa: PLR2004 -- connectivity requires at least 2 channels
        msg = f"n_channels must be >= 2 to form off-diagonal pairs, got {n_channels}."
        raise ValueError(msg)
    return tuple((i, j) for i in range(n_channels) for j in range(i + 1, n_channels))


class ConnectivityExtraction(NamedTuple):
    """Causal multi-segment signed imaginary coherency Frames and validity diagnostics.

    Attributes
    ----------
    times
        Frame endpoint times in seconds, shape ``(n_frames,)``.
    values
        Signed imaginary coherency in ``[-1, 1]`` for valid entries and ``NaN`` where masked,
        shape ``(n_frames, n_pairs)``.
    valid_mask
        Boolean validity mask per Frame and channel pair, shape ``(n_frames, n_pairs)``.
    pairs
        Off-diagonal channel index pairs ``(i, j)`` with ``i < j``, length ``n_pairs``.
    n_segments
        Welch Segment count available in causal support at each Frame, shape ``(n_frames,)``.
    window_s
        Causal trailing support duration in seconds.
    hop_s
        Frame hop in seconds.
    band_hz
        Averaged frequency band ``(low_hz, high_hz)`` in Hz.
    psd_floor
        Minimum channel auto-power threshold required for a valid pair estimate.
    """

    times: FloatArray
    values: FloatArray
    valid_mask: BoolArray
    pairs: tuple[tuple[int, int], ...]
    n_segments: IntArray
    window_s: float
    hop_s: float
    band_hz: tuple[float, float]
    psd_floor: float

    @property
    def pair_valid_fraction(self) -> float:
        """Fraction of valid ``(frame, pair)`` entries across the extracted series."""
        if self.valid_mask.size == 0:
            return 0.0
        return float(np.mean(self.valid_mask))

    @property
    def frame_valid_mask(self) -> BoolArray:
        """Boolean mask of Frames with at least :data:`CONNECTIVITY_VALID_PAIR_THRESHOLD` valid pairs."""
        if self.valid_mask.size == 0:
            return np.zeros(0, dtype=bool)
        return np.asarray(np.mean(self.valid_mask, axis=1) >= CONNECTIVITY_VALID_PAIR_THRESHOLD, dtype=bool)

    @property
    def frame_valid_fraction(self) -> float:
        """Fraction of Frames meeting the valid-pair coverage threshold."""
        mask = self.frame_valid_mask
        if mask.size == 0:
            return 0.0
        return float(np.mean(mask))


def _evaluate_connectivity_frame(  # noqa: PLR0913 -- per-Frame spectral slice, taper, and pair indices
    stitched: FloatArray,
    end_idx: int,
    *,
    m_count: int,
    n_segment: int,
    n_welch_hop: int,
    w_seg: FloatArray,
    freq_mask: BoolArray,
    fold_band: FloatArray,
    scale_norm: float,
    pair_i: IntArray,
    pair_j: IntArray,
    psd_floor: float,
) -> tuple[FloatArray, BoolArray]:
    """Average per-bin signed imaginary coherency over the band for one full Frame."""
    segments = np.stack(
        [stitched[end_idx - m * n_welch_hop - n_segment : end_idx - m * n_welch_hop] for m in range(m_count)],
        axis=0,
    )
    spec = np.fft.rfft(segments * w_seg[None, :, None], axis=1)[:, freq_mask, :]
    weighted_spec = spec.transpose(1, 0, 2) * np.sqrt(fold_band / scale_norm)[:, None, None]
    auto_psd = np.mean(np.abs(weighted_spec) ** 2, axis=1)
    real = np.ascontiguousarray(weighted_spec.real)
    imag = np.ascontiguousarray(weighted_spec.imag)
    im_mat = imag.transpose(0, 2, 1) @ real / m_count
    cross_im = im_mat[:, pair_i, pair_j] - im_mat[:, pair_j, pair_i]

    psd_i = auto_psd[:, pair_i]
    psd_j = auto_psd[:, pair_j]
    pair_ok = np.all(np.isfinite(psd_i) & np.isfinite(psd_j) & (psd_i > psd_floor) & (psd_j > psd_floor), axis=0)
    frame_vals = np.full(len(pair_i), np.nan, dtype=np.float64)
    if np.any(pair_ok):
        denom = np.sqrt(psd_i[:, pair_ok] * psd_j[:, pair_ok])
        frame_vals[pair_ok] = np.clip(np.mean(cross_im[:, pair_ok] / denom, axis=0), -1.0, 1.0)
    return frame_vals, pair_ok


def extract_connectivity_frames(  # noqa: PLR0913 -- connectivity window, segment, hop, band, taper, and floor knobs
    continuation: FloatArray,
    fs: float = DEFAULT_WAVEFORM_FS,
    *,
    history: FloatArray | None = None,
    window_s: float = CONNECTIVITY_CANDIDATE_WINDOWS_S[0],
    hop_s: float = DEFAULT_FRAME_HOP_S,
    n_segment: int = 25,
    n_welch_hop: int = 3,
    band_hz: tuple[float, float] = SEIZURE_BAND_HZ,
    window: Literal["hann", "hann_poisson"] = "hann",
    asymmetric_window: bool = False,
    min_segments: int = MIN_CONNECTIVITY_SEGMENTS,
    psd_floor: float = DEFAULT_PSD_FLOOR,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    target_fs: float = DEFAULT_WAVEFORM_FS,
) -> ConnectivityExtraction:
    """Extract causal multi-segment signed imaginary coherency Frames across the branch boundary.

    Parameters
    ----------
    continuation
        Post-branch EEG array of shape ``(steps, n_channels)`` at ``fs``.
    fs
        Input sampling frequency in Hz.
    history
        Optional pre-branch EEG history of shape ``(history_steps, n_channels)`` at ``fs``.
    window_s
        Causal trailing support duration ``W_conn`` in seconds.
    hop_s
        Frame hop interval in seconds.
    n_segment
        STFT Segment length in conditioned samples (``25`` samples = 0.50 s at 50 Hz).
    n_welch_hop
        Internal Welch step in conditioned samples between overlapping Segments (``3`` samples = 0.06 s).
    band_hz
        Frequency band ``(low_hz, high_hz)`` in Hz over which signed imaginary coherency is averaged;
        DC (0 Hz) is always excluded.
    window
        Segment taper name (``'hann'`` or ``'hann_poisson'``).
    asymmetric_window
        Whether to use an asymmetric causal taper on each Segment.
    min_segments
        Minimum number of overlapping Welch Segments ``M_min`` required for a valid Frame.
    psd_floor
        Minimum auto-spectrum power required in each band bin for both channels of a pair.
    cutoff_hz
        Anti-aliasing low-pass cutoff in Hz when conditioning from ``fs > target_fs``.
    target_fs
        Conditioned EEG sampling frequency in Hz.

    Returns
    -------
    ConnectivityExtraction
        Extracted signed imaginary coherency Frames, timestamps, pair table, and validity masks.
    """
    hist_50, cont_50 = condition_waveform_pair(
        continuation, fs, history_raw=history, cutoff_hz=cutoff_hz, target_fs=target_fs
    )
    pairs = connectivity_pair_indices(cont_50.shape[1])
    n_pairs = len(pairs)
    pair_i = np.array([p[0] for p in pairs], dtype=np.intp)
    pair_j = np.array([p[1] for p in pairs], dtype=np.intp)
    n_support = round(window_s * target_fs)
    n_frame_hop = max(1, round(hop_s * target_fs))

    if hist_50 is not None:
        stitched = np.concatenate([hist_50, cont_50], axis=0)
        n_frames = cont_50.shape[0] // n_frame_hop
        endpoints = hist_50.shape[0] + np.arange(1, n_frames + 1, dtype=np.intp) * n_frame_hop
        times = np.arange(1, n_frames + 1, dtype=np.float64) * (n_frame_hop / target_fs)
    else:
        stitched = cont_50
        starts = (
            np.arange(0, stitched.shape[0] - n_support + 1, n_frame_hop, dtype=np.intp)
            if stitched.shape[0] >= n_support
            else np.empty(0, dtype=np.intp)
        )
        endpoints = starts + n_support
        times = endpoints.astype(np.float64) / target_fs
        n_frames = len(endpoints)

    freqs = np.fft.rfftfreq(n_segment, d=1.0 / target_fs)
    freq_mask = np.asarray((freqs > 0.0) & (freqs >= band_hz[0]) & (freqs <= band_hz[1]), dtype=bool)
    if not np.any(freq_mask):
        msg = (
            f"No non-DC frequency bins fall within band_hz={band_hz} for n_segment={n_segment} at fs={target_fs:g} Hz."
        )
        raise ValueError(msg)

    w_seg = compute_segment_window(n_segment, window=window, asymmetric_window=asymmetric_window)
    fold = np.full(n_segment // 2 + 1, 2.0, dtype=np.float64)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    fold_band = fold[freq_mask]
    scale_norm = float(target_fs * np.sum(w_seg**2))

    values = np.full((n_frames, n_pairs), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_frames, n_pairs), dtype=bool)
    n_segments_arr = np.zeros(n_frames, dtype=np.intp)

    for frame_idx, end_idx in enumerate(endpoints):
        avail = int(end_idx) - max(0, int(end_idx) - n_support)
        m_count = 0 if avail < n_segment else (avail - n_segment) // n_welch_hop + 1
        n_segments_arr[frame_idx] = m_count
        if avail == n_support and m_count >= min_segments:
            values[frame_idx], valid_mask[frame_idx] = _evaluate_connectivity_frame(
                stitched,
                int(end_idx),
                m_count=m_count,
                n_segment=n_segment,
                n_welch_hop=n_welch_hop,
                w_seg=w_seg,
                freq_mask=freq_mask,
                fold_band=fold_band,
                scale_norm=scale_norm,
                pair_i=pair_i,
                pair_j=pair_j,
                psd_floor=psd_floor,
            )

    return ConnectivityExtraction(
        times=times,
        values=values,
        valid_mask=valid_mask,
        pairs=pairs,
        n_segments=n_segments_arr,
        window_s=float(window_s),
        hop_s=float(hop_s),
        band_hz=(float(band_hz[0]), float(band_hz[1])),
        psd_floor=float(psd_floor),
    )


@dataclass(frozen=True)
class ConnectivitySupportSelection:
    """Summary of development-set causal support window selection for connectivity."""

    selected_window_s: float
    candidate_windows_s: tuple[float, ...]
    candidate_spreads: dict[float, float]
    candidate_normalized_spreads: dict[float, float]
    candidate_valid_fractions: dict[float, float]
    support_to_horizon_ratio: dict[float, float]
    repeatability_target: float
    qualified: bool


def _validate_dev_identities(identities: Sequence[RolloutIdentity] | None, split: str) -> None:
    """Enforce that development support-window selection runs on ``split='dev'`` only."""
    if split != "dev":
        msg = f"Support-window selection must run on split='dev' only, got {split!r}."
        raise ValueError(msg)
    if identities is not None:
        if len(identities) == 0:
            msg = "identities cannot be empty."
            raise ValueError(msg)
        for ident in identities:
            if ident.split != "dev":
                msg = f"Support-window selection requires split='dev' identities, got {ident}."
                raise ValueError(msg)


def _group_dev_trajectories(
    dev_waveforms: FloatArray,
    *,
    n_replicates: int | None = None,
    identities: Sequence[RolloutIdentity] | None = None,
    split: str = "dev",
    dev_histories: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray | None]:
    """Validate ``split='dev'`` and reshape waveforms into ``(n_states, n_replicates, steps, n_channels)``."""
    _validate_dev_identities(identities, split)
    arr = np.asarray(dev_waveforms, dtype=np.float64)
    hist_arr = None if dev_histories is None else np.asarray(dev_histories, dtype=np.float64)
    if arr.ndim == 4:  # noqa: PLR2004 -- already (n_states, n_replicates, steps, n_channels)
        return arr, hist_arr
    if arr.ndim != 3:  # noqa: PLR2004 -- (n_rollouts, steps, n_channels)
        msg = (
            "dev_waveforms must have shape (n_states, n_replicates, steps, n_channels) "
            f"or (n_rollouts, steps, n_channels), got {arr.shape}."
        )
        raise ValueError(msg)

    if identities is not None:
        groups: dict[tuple[str, int, str, str], list[tuple[int, int]]] = {}
        for idx, ident in enumerate(identities):
            groups.setdefault((*ident.snapshot_key, ident.arm), []).append((ident.replicate, idx))
        if len({len(v) for v in groups.values()}) != 1:
            msg = "Unbalanced replicate counts across development snapshots."
            raise ValueError(msg)
        ordered_indices = [
            [idx for _, idx in sorted(items, key=lambda pair: pair[0])]
            for _, items in sorted(groups.items(), key=lambda kv: kv[0])
        ]
        idx_grid = np.asarray(ordered_indices, dtype=np.intp)
        return arr[idx_grid], (None if hist_arr is None else hist_arr[idx_grid])

    if n_replicates is None or n_replicates < 1 or arr.shape[0] % n_replicates != 0:
        msg = f"Cannot group {arr.shape[0]} rollouts into equal states with n_replicates={n_replicates}."
        raise ValueError(msg)
    n_states = arr.shape[0] // n_replicates
    grouped = arr.reshape(n_states, n_replicates, arr.shape[1], arr.shape[2])
    grouped_hist = (
        None if hist_arr is None else hist_arr.reshape(n_states, n_replicates, hist_arr.shape[1], hist_arr.shape[2])
    )
    return grouped, grouped_hist


def _score_connectivity_support_candidate(  # noqa: PLR0913 -- grouped dev arrays and connectivity settings
    grouped: FloatArray,
    grouped_hist: FloatArray | None,
    *,
    fs: float,
    window_s: float,
    hop_s: float,
    n_segment: int,
    n_welch_hop: int,
    band_hz: tuple[float, float],
) -> tuple[float, float, float]:
    """Compute native spread, normalized spread, and valid fraction for one candidate ``window_s``."""
    n_states, rep_count, _, _ = grouped.shape
    state_rep_frames: list[FloatArray] = []
    state_rep_masks: list[BoolArray] = []
    for s_idx in range(n_states):
        rep_vals: list[FloatArray] = []
        rep_masks: list[BoolArray] = []
        for r_idx in range(rep_count):
            hist_sr = None if grouped_hist is None else grouped_hist[s_idx, r_idx]
            ext = extract_connectivity_frames(
                grouped[s_idx, r_idx],
                fs,
                history=hist_sr,
                window_s=window_s,
                hop_s=hop_s,
                n_segment=n_segment,
                n_welch_hop=n_welch_hop,
                band_hz=band_hz,
            )
            rep_vals.append(ext.values)
            rep_masks.append(ext.valid_mask)
        state_rep_frames.append(np.stack(rep_vals, axis=0))
        state_rep_masks.append(np.stack(rep_masks, axis=0))

    block = np.stack(state_rep_frames, axis=0)
    mask_block = np.stack(state_rep_masks, axis=0)
    valid_frac = float(np.mean(mask_block)) if mask_block.size > 0 else 0.0
    ddof = 1 if rep_count > 1 else 0
    comp_valid = np.sum(mask_block, axis=1) >= (2 if rep_count > 1 else 1)
    if block.shape[2] == 0 or not np.any(comp_valid):
        return float("nan"), float("nan"), valid_frac

    with np.errstate(invalid="ignore", divide="ignore"):
        rep_var = np.nanvar(block, axis=1, ddof=ddof)
        comp_var = np.nanvar(block.reshape(-1, block.shape[-1]), axis=0, ddof=ddof)
    native_spread = float(np.sqrt(np.nanmean(rep_var[comp_valid])))
    s_dev = float(np.sqrt(np.nanmean(comp_var)))
    norm_spread = float(native_spread / s_dev) if s_dev > 0.0 and np.isfinite(s_dev) else float("nan")
    return native_spread, norm_spread, valid_frac


def select_connectivity_support_window(  # noqa: PLR0913 -- dev dataset, candidates, target, and spectral parameters
    dev_waveforms: FloatArray,
    fs: float = DEFAULT_WAVEFORM_FS,
    *,
    dev_histories: FloatArray | None = None,
    n_replicates: int | None = None,
    identities: Sequence[RolloutIdentity] | None = None,
    split: str = "dev",
    candidate_windows_s: Sequence[float] = CONNECTIVITY_CANDIDATE_WINDOWS_S,
    repeatability_target: float = 0.25,
    horizons_s: Sequence[float] = CONTROL_HORIZONS_S,
    hop_s: float = DEFAULT_FRAME_HOP_S,
    n_segment: int = 25,
    n_welch_hop: int = 3,
    band_hz: tuple[float, float] = SEIZURE_BAND_HZ,
) -> ConnectivitySupportSelection:
    """Select the shortest causal connectivity support window meeting development repeatability.

    Evaluates candidate support windows ``W_conn`` exclusively on ``split='dev'`` runs, computes
    the component-preserving conditional replicate spread across valid channel pairs, picks the
    shortest candidate meeting ``repeatability_target``, and reports ``W_conn / H`` across all
    declared Control Horizons.

    Parameters
    ----------
    dev_waveforms
        Development-split EEG waveforms of shape ``(n_states, n_replicates, steps, n_channels)``
        or ``(n_rollouts, steps, n_channels)``.
    fs
        Sampling frequency in Hz.
    dev_histories
        Optional pre-branch EEG histories matching the leading dimensions of ``dev_waveforms``.
    n_replicates
        Number of stochastic replicates per parent snapshot when ``dev_waveforms`` is 3D.
    identities
        Optional rollout identities; all entries must have ``split='dev'``.
    split
        Dataset split label; must be ``'dev'``.
    candidate_windows_s
        Candidate support durations in seconds, evaluated in ascending order.
    repeatability_target
        Maximum acceptable native conditional replicate spread ``sigma_rep`` on ``split='dev'``.
    horizons_s
        Declared Control Horizons ``H`` in seconds for ``W_conn / H`` reporting.
    hop_s
        Frame hop interval in seconds.
    n_segment
        STFT Segment length in 50 Hz samples.
    n_welch_hop
        Internal Welch hop in 50 Hz samples.
    band_hz
        Frequency band ``(low_hz, high_hz)`` in Hz.

    Returns
    -------
    ConnectivitySupportSelection
        Selected window, per-candidate spreads, validity fractions, and horizon ratios.
    """
    grouped, grouped_hist = _group_dev_trajectories(
        dev_waveforms,
        n_replicates=n_replicates,
        identities=identities,
        split=split,
        dev_histories=dev_histories,
    )
    candidates = tuple(sorted(float(w) for w in candidate_windows_s))
    if not candidates:
        msg = "candidate_windows_s cannot be empty."
        raise ValueError(msg)

    spreads: dict[float, float] = {}
    norm_spreads: dict[float, float] = {}
    valid_fractions: dict[float, float] = {}

    for w_s in candidates:
        spreads[w_s], norm_spreads[w_s], valid_fractions[w_s] = _score_connectivity_support_candidate(
            grouped,
            grouped_hist,
            fs=fs,
            window_s=w_s,
            hop_s=hop_s,
            n_segment=n_segment,
            n_welch_hop=n_welch_hop,
            band_hz=band_hz,
        )

    qualifying = [
        w_s
        for w_s in candidates
        if np.isfinite(spreads[w_s])
        and valid_fractions[w_s] >= CONNECTIVITY_VALID_PAIR_THRESHOLD
        and spreads[w_s] <= repeatability_target
    ]
    if qualifying:
        selected = qualifying[0]
        qualified = True
    else:
        finite_candidates = [w_s for w_s in candidates if np.isfinite(spreads[w_s])]
        selected = min(finite_candidates, key=lambda w_s: spreads[w_s]) if finite_candidates else candidates[-1]
        qualified = False

    ratios = {float(h): float(selected / float(h)) for h in horizons_s}
    return ConnectivitySupportSelection(
        selected_window_s=float(selected),
        candidate_windows_s=candidates,
        candidate_spreads=spreads,
        candidate_normalized_spreads=norm_spreads,
        candidate_valid_fractions=valid_fractions,
        support_to_horizon_ratio=ratios,
        repeatability_target=float(repeatability_target),
        qualified=qualified,
    )


@dataclass(frozen=True)
class HealthyConnectivityReference:
    """Healthy per-pair 5th--95th percentile interval reference fitted on ``split='reference'`` runs."""

    q05: FloatArray
    q95: FloatArray
    pairs: tuple[tuple[int, int], ...]
    window_s: float
    pair_valid_fraction: FloatArray
    eligible: BoolArray | None = None

    @classmethod
    def fit(
        cls,
        reference_connectivity: ConnectivityExtraction | FloatArray | Sequence[ConnectivityExtraction | FloatArray],
        *,
        window_s: float = CONNECTIVITY_CANDIDATE_WINDOWS_S[0],
        identities: Sequence[RolloutIdentity] | None = None,
        split: str = "reference",
    ) -> HealthyConnectivityReference:
        """Fit per-pair healthy percentile intervals on reference runs.

        Parameters
        ----------
        reference_connectivity
            Extracted connectivity Frames of shape ``(..., n_frames, n_pairs)`` or
            :class:`ConnectivityExtraction` instances from healthy reference runs.
        window_s
            Causal support window duration ``W_conn`` in seconds.
        identities
            Optional rollout identities validated for ``plant='healthy'`` and ``split='reference'``.
        split
            Dataset split label; must be ``'reference'``.
        """
        _validate_reference_split(identities, split)
        pairs: tuple[tuple[int, int], ...] | None = None
        blocks: list[FloatArray] = []

        if isinstance(reference_connectivity, ConnectivityExtraction):
            pairs = reference_connectivity.pairs
            window_s = reference_connectivity.window_s
            blocks.append(reference_connectivity.values.reshape(-1, reference_connectivity.values.shape[-1]))
        elif isinstance(reference_connectivity, np.ndarray):
            arr = np.asarray(reference_connectivity, dtype=np.float64)
            if arr.ndim < 2:  # noqa: PLR2004 -- (..., n_frames, n_pairs)
                msg = f"reference_connectivity array must have at least 2 dimensions, got {arr.shape}."
                raise ValueError(msg)
            blocks.append(arr.reshape(-1, arr.shape[-1]))
        else:
            for item in reference_connectivity:
                if isinstance(item, ConnectivityExtraction):
                    pairs = item.pairs
                    window_s = item.window_s
                    blocks.append(item.values.reshape(-1, item.values.shape[-1]))
                else:
                    arr = np.asarray(item, dtype=np.float64)
                    blocks.append(arr.reshape(-1, arr.shape[-1]))

        if not blocks:
            msg = "reference_connectivity cannot be empty."
            raise ValueError(msg)
        flat = np.concatenate(blocks, axis=0)
        if flat.shape[0] == 0:
            msg = "reference_connectivity contains 0 Frames."
            raise ValueError(msg)

        n_pairs = flat.shape[1]
        if pairs is None:
            n_ch = round((1.0 + np.sqrt(1.0 + 8.0 * n_pairs)) / 2.0)
            pairs = (
                connectivity_pair_indices(n_ch)
                if n_ch * (n_ch - 1) // 2 == n_pairs
                else tuple((0, p + 1) for p in range(n_pairs))
            )

        finite = np.isfinite(flat)
        pair_valid_frac = np.asarray(np.mean(finite, axis=0), dtype=np.float64)
        has_reference = np.any(finite, axis=0)
        q05 = np.full(n_pairs, np.nan)
        q95 = np.full(n_pairs, np.nan)
        if np.any(has_reference):
            q05[has_reference], q95[has_reference] = np.nanquantile(flat[:, has_reference], (0.05, 0.95), axis=0)
        return cls(
            q05=q05,
            q95=q95,
            pairs=pairs,
            window_s=float(window_s),
            pair_valid_fraction=pair_valid_frac,
        )


class ConnectivityPenaltyResult(NamedTuple):
    """Frame-level connectivity interval excess penalty and validity coverage diagnostics.

    Attributes
    ----------
    penalty
        Frame-level connectivity penalty ``P_conn(t)``, ``NaN`` where an eligible pair is invalid,
        shape ``(..., n_frames)``.
    frame_valid
        Boolean mask indicating which Frames have all eligible pairs valid,
        shape ``(..., n_frames)``.
    pair_valid_fraction
        Overall fraction of valid ``(frame, pair)`` entries in ``[0, 1]``.
    frame_valid_fraction
        Overall fraction of valid Frames in ``[0, 1]``.
    pair_penalties
        Per-pair squared interval excess ``l_p(t)``, ``NaN`` where masked, shape ``(..., n_frames, n_pairs)``.
    """

    penalty: FloatArray
    frame_valid: BoolArray
    pair_valid_fraction: float
    frame_valid_fraction: float
    pair_penalties: FloatArray

    @property
    def frame_penalty(self) -> FloatArray:
        """Alias for ``penalty``."""
        return self.penalty


def compute_connectivity_penalty(
    connectivity: ConnectivityExtraction | FloatArray,
    reference: HealthyConnectivityReference,
    *,
    valid_mask: BoolArray | None = None,
) -> ConnectivityPenaltyResult:
    """Average squared interval excess over fixed eligible pairs, requiring all to be valid."""
    if isinstance(connectivity, ConnectivityExtraction):
        vals = np.asarray(connectivity.values, dtype=np.float64)
        mask = (
            np.asarray(connectivity.valid_mask, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
    else:
        vals = np.asarray(connectivity, dtype=np.float64)
        mask = np.isfinite(vals) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & np.isfinite(vals))

    mask = mask & np.isfinite(vals)
    n_pairs = vals.shape[-1]
    if reference.q05.shape != (n_pairs,):
        msg = f"reference pair count ({reference.q05.shape[0]}) does not match connectivity pairs ({n_pairs})."
        raise ValueError(msg)

    eligible = np.ones(n_pairs, dtype=bool) if reference.eligible is None else reference.eligible
    valid = mask[..., eligible]
    excess = np.maximum(0.0, np.maximum(reference.q05 - vals, vals - reference.q95))
    pair_penalties = np.where(mask & eligible, excess**2, np.nan)
    frame_valid = np.all(valid, axis=-1) & bool(np.any(eligible))
    penalty = np.full(vals.shape[:-1], np.nan)
    penalty[frame_valid] = pair_penalties[..., eligible][frame_valid].mean(axis=-1)
    pair_valid_fraction = float(valid.mean()) if valid.size else 0.0
    frame_valid_fraction = float(np.mean(frame_valid)) if frame_valid.size > 0 else 0.0
    return ConnectivityPenaltyResult(
        penalty=np.asarray(penalty, dtype=np.float64),
        frame_valid=frame_valid,
        pair_valid_fraction=pair_valid_fraction,
        frame_valid_fraction=frame_valid_fraction,
        pair_penalties=np.asarray(pair_penalties, dtype=np.float64),
    )
