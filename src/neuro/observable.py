from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal.windows import hann

from neuro.config import EegMsGeometry, ObservableGeometry, StftGeometry
from neuro.predictor.artifact import ACTIVATIONS, Layers, mlp_forward
from neuro.provenance import TrainingProvenance
from neuro.spectral import LOG_FLOOR, MsEnvelope, PsdEnvelope
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray


def _segments(x: FloatArray, size: int, hop: int) -> FloatArray:
    """Hopped windows along the last axis: ``(..., n)`` -> ``(..., n_windows, size)``."""
    return np.lib.stride_tricks.sliding_window_view(x, size, axis=-1)[..., ::hop, :]


def _spectrogram(x: FloatArray, geometry: StftGeometry, fs: float) -> FloatArray:
    """Hopped periodograms of ``(..., n_channels, n_samples)`` -> ``(..., n_channels, n_frames, n_bins)``.

    The NumPy twin of :func:`neuro.predictor.losses.spectrogram` and of
    :func:`neuro.spectral.compute_periodograms`: one-sided, density-scaled, periodic Hann, no
    per-segment detrend.
    """
    n_segment = geometry.n_segment
    window = hann(n_segment, sym=False)
    spectrum = np.fft.rfft(_segments(x, n_segment, geometry.n_hop) * window, n=n_segment, axis=-1)
    psd = np.abs(spectrum) ** 2 / (fs * np.sum(window**2))

    # One-sided: every bin carries its negative-frequency twin, except DC and (even n) Nyquist.
    fold = np.full(psd.shape[-1], 2.0)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    return psd * fold


def pool_bins(power: FloatArray, n_bin_pool: int) -> FloatArray:
    """Mean power over consecutive groups of ``n_bin_pool`` bins, dropping the trailing remainder."""
    if n_bin_pool == 1:
        return power
    n_groups = power.shape[-1] // n_bin_pool
    grouped = power[..., : n_groups * n_bin_pool].reshape(*power.shape[:-1], n_groups, n_bin_pool)
    return grouped.mean(axis=-1)


def frame_kernel(kernel: str, width: int) -> FloatArray:
    """Normalised non-negative smoothing weights along the frame axis.

    Endpoints are kept strictly positive, so a width-``n`` taper really pools ``n`` frames -- the
    NumPy twin of :func:`neuro.predictor.losses.frame_kernel`.
    """
    if width == 1:
        return np.ones(1)
    if kernel == "boxcar":
        weights = np.ones(width)
    elif kernel == "triangular":
        weights = np.bartlett(width + 2)[1:-1]
    else:
        weights = np.hanning(width + 2)[1:-1]
    return weights / weights.sum()


def smooth_frames(power: FloatArray, weights: FloatArray) -> FloatArray:
    """Convolve power ``(..., n_frames, n_bins)`` along the frame axis, valid support only."""
    if weights.size == 1:
        return power
    windows = _segments(np.moveaxis(power, -2, -1), weights.size, 1)
    return np.moveaxis(windows @ weights, -1, -2)


def log_observable(y: FloatArray, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Reduce raw EEG ``(..., n_samples, n_channels)`` onto the Frame grid, in log units.

    Returns ``(..., n_frames, n_channels, n_values)``: the quantity the observable Predictor
    forecasts, the training Loss scores and the MPC Cost hinges -- computed by the one geometry
    object all three share.
    """
    x = np.moveaxis(np.asarray(y, dtype=np.float64), -1, -2)
    if isinstance(geometry, StftGeometry):
        bin_lo, bin_hi = geometry.bin_range(fs)
        power = _spectrogram(x, geometry, fs)[..., bin_lo:bin_hi]
        power = pool_bins(power, geometry.n_bin_pool)
        power = smooth_frames(power, frame_kernel(geometry.kernel, geometry.kernel_width))
    elif isinstance(geometry, EegMsGeometry):
        windows = _segments(x, geometry.window_steps(fs), geometry.hop_steps(fs))
        power = (windows**2).mean(axis=-1)[..., None]
    else:  # pragma: no cover -- the two kinds above are the whole of ObservableGeometry
        msg = f"unsupported observable geometry {type(geometry).__name__}"
        raise TypeError(msg)
    return np.log(np.moveaxis(power, -3, -2) + LOG_FLOOR)


def control_means(geometry: ObservableGeometry, horizon: int, fs: float) -> FloatArray:
    """Build the fixed ``(n_frames, horizon)`` operator averaging Control Currents over a Frame's support.

    Unweighted: the currents act by shifting a regional operating point rather than by cancelling
    cycles, so the mean is the first-order summary of a Segment's Stimulation Drive. The Hann taper
    belongs to the spectral estimator, not to the plant's response.
    """
    supports = geometry.frame_supports(horizon, fs)
    operator = np.zeros((len(supports), horizon), dtype=np.float64)
    for m, (start, end) in enumerate(supports):
        operator[m, start:end] = 1.0 / (end - start)
    return operator


def envelope_log_reference(envelope: PsdEnvelope | MsEnvelope, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Reduce a healthy envelope onto the Frame's value grid: ``(n_channels, n_values)`` in log units.

    Pooling happens on the reference *power*, before the log, exactly as it does on the measured
    power in :func:`log_observable`, so the hinge compares two quantities built the same way.
    """
    if isinstance(geometry, StftGeometry) and isinstance(envelope, PsdEnvelope):
        bin_lo, bin_hi = geometry.bin_range(fs)
        return np.log(pool_bins(envelope.power[:, bin_lo:bin_hi], geometry.n_bin_pool))
    if isinstance(geometry, EegMsGeometry) and isinstance(envelope, MsEnvelope):
        return np.log(envelope.power[:, None])
    msg = f"{type(envelope).__name__} is not the reference envelope for a {geometry.kind} observable."
    raise TypeError(msg)


def geometry_from_meta(meta: dict[str, Any]) -> ObservableGeometry:
    """Rebuild the recorded Observable geometry from an artifact's metadata block."""
    fields = dict(meta)
    kind = fields.pop("kind")
    if kind == StftGeometry.kind:
        return StftGeometry.model_validate(fields)
    if kind == EegMsGeometry.kind:
        return EegMsGeometry.model_validate(fields)
    msg = f"unsupported observable kind {kind!r}"
    raise ValueError(msg)


def geometry_meta(geometry: ObservableGeometry) -> dict[str, Any]:
    """Serialize an Observable geometry, tagged with its kind, for an artifact's metadata block."""
    return {"kind": geometry.kind, **geometry.model_dump(mode="json")}


@dataclass(frozen=True)
class ObservableArtifact:
    """Framework-free observable-space predictor: lift, one shared Frame transition, log readout.

    The Rollout recurses on the Frame grid rather than the sample grid, so a Control Horizon of
    ``horizon`` samples costs ``geometry.n_frames(horizon, fs)`` nodes on a state of order 10**1.

    Attributes
    ----------
    lift, transition : Layers
        ``E`` maps the history state to ``z_0``; ``f_theta(z_m, u_bar_m)`` advances one Frame and is
        shared across every Frame.
    readout : tuple[FloatArray, FloatArray]
        The affine ``(C, c)`` emitting **standardized** log-Observable; no exponential, no
        positivity constraint, and no ``LOG_FLOOR`` is ever applied to a predicted quantity.
    activation : Activation
        Activation applied after every layer of ``lift`` and ``transition`` except their last.
    n_y, n_u : int
        Past EEG and past control steps in the history window ``E`` lifts from.
    horizon : int
        The Control Horizon in samples the model was fit at; the Frame count follows from it and
        ``geometry``, so the artifact records geometry rather than a frozen Frame count.
    n_channels, n_controls : int
        Physical EEG channel count and control electrode count.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    geometry : ObservableGeometry
        The Observable grid the model was trained against -- the field no other artifact carries,
        and the one :mod:`neuro.validation` checks against the reference envelope.
    y_std, u_std, l_std : Standardizer
        Channel, control and log-Observable standardizers.
    provenance : TrainingProvenance
        What the training data was made of.
    """

    lift: Layers
    transition: Layers
    readout: tuple[FloatArray, FloatArray]
    activation: Activation
    n_y: int
    n_u: int
    horizon: int
    n_channels: int
    n_controls: int
    dt: float
    downsample: int
    geometry: ObservableGeometry
    y_std: Standardizer
    u_std: Standardizer
    l_std: Standardizer
    provenance: TrainingProvenance = field(default_factory=TrainingProvenance)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('observable')."""
        return "observable"

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        return 1.0 / self.dt

    @property
    def priming_steps(self) -> int:
        """Minimum number of history steps required to initialize the model state."""
        return max(self.n_y, self.n_u)

    @property
    def z_dim(self) -> int:
        """Dimension of the lifted Frame state."""
        return self.readout[0].shape[1]

    def n_frames(self, horizon: int | None = None) -> int:
        """Frames the recursion emits over ``horizon`` samples (default: the trained horizon)."""
        return self.geometry.n_frames(self.horizon if horizon is None else horizon, self.fs)

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        return self.geometry.n_values(self.fs)

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_channels)`` into standardized space."""
        return self.y_std.transform(np.asarray(y, dtype=np.float64))

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into an initial state (model-space y, **raw** u tail).

        Byte-identical in layout to :meth:`neuro.predictor.artifact.MLPArtifact.prime`, so State
        Absorption, Priming and readiness carry over unchanged.
        """
        y_arr = np.asarray(y_hist, dtype=np.float64)
        u_arr = np.asarray(u_hist, dtype=np.float64)
        z_past = self.encode(y_arr)[-self.n_y :].reshape(-1)
        u_past = u_arr[-self.n_u :].reshape(-1)
        return np.concatenate([z_past, u_past])

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
        y_arr = np.asarray(y_hists, dtype=np.float64)
        u_arr = np.asarray(u_hists, dtype=np.float64)
        n_batch = y_arr.shape[0]
        z_past = self.encode(y_arr)[:, -self.n_y :].reshape(n_batch, -1)
        u_past = u_arr[:, -self.n_u :].reshape(n_batch, -1)
        return np.concatenate([z_past, u_past], axis=-1)

    def lift_state(self, state: FloatArray) -> FloatArray:
        """Map a primed state ``(..., n_y * C + n_u * m)`` to the lifted Frame state ``(..., z_dim)``."""
        arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        u_past = self.u_std.transform(arr[..., n_z:].reshape(*arr.shape[:-1], self.n_u, self.n_controls))
        lift_in = np.concatenate([arr[..., :n_z], u_past.reshape(*arr.shape[:-1], -1)], axis=-1)
        return mlp_forward(lift_in, self.lift, self.activation)

    def forecast(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Forecast the log-Observable from ``state`` under raw ``u_future`` ``(..., horizon, m)``.

        Returns ``(..., n_frames, n_channels, n_values)`` in raw log units. ``u_future[t]`` is the
        control applied at prediction step ``t``, the same alignment the sample-grid Rollout uses.
        """
        u_arr = np.asarray(u_future, dtype=np.float64)
        horizon = u_arr.shape[-2]
        u_bar = self.u_std.transform(
            np.einsum("mt,...tc->...mc", control_means(self.geometry, horizon, self.fs), u_arr)
        )

        z = self.lift_state(state)
        frames = []
        for m in range(u_bar.shape[-2]):
            z = mlp_forward(np.concatenate([z, u_bar[..., m, :]], axis=-1), self.transition, self.activation)
            frames.append(z @ self.readout[0].T + self.readout[1])
        stacked = self.l_std.inverse_transform(np.stack(frames, axis=-2))
        return stacked.reshape(*stacked.shape[:-1], self.n_channels, self.n_values)

    @property
    def meta(self) -> dict[str, Any]:
        """Serializable dictionary representation of artifact metadata."""
        return {
            "model_type": self.model_type,
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "dt": self.dt,
            "downsample": self.downsample,
            "n_lift_layers": len(self.lift),
            "n_transition_layers": len(self.transition),
            "geometry": geometry_meta(self.geometry),
            **self.provenance.meta,
        }

    @classmethod
    def load(cls, artifact: str | Path) -> ObservableArtifact:
        """Load the single-``.npz`` artifact from disk (``artifact`` is a suffix-less stem)."""
        path = Path(artifact).with_suffix(".npz")
        with np.load(path) as npz:
            meta: dict[str, Any] = json.loads(str(npz["meta"]))
            if meta["activation"] not in ACTIVATIONS:
                msg = f"Unsupported activation: {meta['activation']!r}"
                raise ValueError(msg)

            def stack(prefix: str, count: int) -> Layers:
                return tuple(
                    (
                        np.asarray(npz[f"{prefix}.{i}.weight"], dtype=np.float64),
                        np.asarray(npz[f"{prefix}.{i}.bias"], dtype=np.float64),
                    )
                    for i in range(count)
                )

            return cls(
                lift=stack("lift", int(meta["n_lift_layers"])),
                transition=stack("transition", int(meta["n_transition_layers"])),
                readout=(
                    np.asarray(npz["readout.weight"], dtype=np.float64),
                    np.asarray(npz["readout.bias"], dtype=np.float64),
                ),
                activation=meta["activation"],
                n_y=int(meta["n_y"]),
                n_u=int(meta["n_u"]),
                horizon=int(meta["horizon"]),
                n_channels=int(meta["n_channels"]),
                n_controls=int(meta["n_controls"]),
                dt=float(meta["dt"]),
                downsample=int(meta["downsample"]),
                geometry=geometry_from_meta(meta["geometry"]),
                y_std=Standardizer.from_arrays(npz, "y"),
                u_std=Standardizer.from_arrays(npz, "u"),
                l_std=Standardizer.from_arrays(npz, "l"),
                provenance=TrainingProvenance.from_meta(meta),
            )

    def save(self, artifact: str | Path) -> None:
        """Persist weights, transforms and metadata into one ``.npz`` (``artifact`` is a stem).

        ``meta`` is stored as a 0-d unicode array holding JSON, so loading needs no ``allow_pickle``.
        """
        path = Path(artifact).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray[Any, Any]] = {"meta": np.array(json.dumps(self.meta))}
        for prefix, layers in (("lift", self.lift), ("transition", self.transition)):
            for i, (w, b) in enumerate(layers):
                arrays[f"{prefix}.{i}.weight"] = w
                arrays[f"{prefix}.{i}.bias"] = b
        arrays["readout.weight"], arrays["readout.bias"] = self.readout
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        arrays.update(self.l_std.arrays("l"))

        np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]


def load_envelope(path: str | Path, geometry: ObservableGeometry) -> PsdEnvelope | MsEnvelope:
    """Load the healthy reference envelope of the kind ``geometry`` scores, from the shared npz."""
    if isinstance(geometry, StftGeometry):
        return PsdEnvelope.load(path)
    return MsEnvelope.load(path)


def load_log_reference(path: str | Path, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Load and reduce the healthy envelope onto the Frame's value grid, in log units."""
    return envelope_log_reference(load_envelope(path, geometry), geometry, fs)
