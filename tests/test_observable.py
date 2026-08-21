"""Pin the observable target builder against the training losses and the recursion's causality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.config import EegMsGeometry, EegMsSpec, StftGeometry, StftSpec
from neuro.observable import ObservableArtifact, control_means, envelope_log_reference, log_observable
from neuro.predictor.losses import EegMsLoss, LossContext, StftLoss
from neuro.spectral import LOG_FLOOR, MsEnvelope, PsdEnvelope, compute_periodograms, windowed_mean_square

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.types import FloatArray

_SEED = 23
_FS = 50.0


def _ctx(n_channels: int) -> LossContext:
    """An identity-standardizer loss context, so the loss reads raw units straight through."""
    return LossContext(
        y_center=torch.zeros(n_channels, dtype=torch.float64),
        y_scale=torch.ones(n_channels, dtype=torch.float64),
        fs=_FS,
        epoch=None,
    )


def _trajectory(span: int, n_channels: int) -> FloatArray:
    """A random EEG window in raw units, offset so the DC bin is not degenerate."""
    rng = np.random.default_rng(_SEED)
    return rng.standard_normal((span, n_channels)) * 2.0 + 1.0


@pytest.mark.parametrize(
    "geometry",
    [
        StftGeometry(n_segment=50, n_hop=25),
        StftGeometry(n_segment=25, n_hop=12),
        StftGeometry(n_segment=25, n_hop=12, n_bin_pool=2),
        StftGeometry(n_segment=20, n_hop=10, kernel="hann", kernel_width=3),
        StftGeometry(n_segment=50, n_hop=25, band_hz=(3.0, 12.0)),
    ],
)
def test_stft_target_matches_the_training_loss(geometry: StftGeometry) -> None:
    """The offline target and StftLoss.log_spectrogram are two implementations of one quantity.

    They must agree, or the Predictor is trained against one quantity and scored on another.
    """
    span, n_channels = 75, 3
    y = _trajectory(span, n_channels)

    spec = StftSpec(weight=1.0, n_span=span, **geometry.model_dump())
    loss_value = StftLoss.from_spec(spec, _FS).log_spectrogram(
        torch.as_tensor(y[None], dtype=torch.float64), _ctx(n_channels)
    )

    target = log_observable(y, geometry, _FS)
    np.testing.assert_allclose(target, np.moveaxis(loss_value.numpy()[0], 0, 1), rtol=1e-12, atol=1e-12)
    assert target.shape == (geometry.n_frames(span, _FS), n_channels, geometry.n_values(_FS))


@pytest.mark.parametrize(("window_s", "hop_s"), [(1.0, 0.5), (0.1, 0.05), (0.4, 0.2)])
def test_eeg_ms_target_matches_the_training_loss(window_s: float, hop_s: float) -> None:
    """The offline target equals ``log(EegMsLoss.windowed_power)`` on the same trajectory."""
    span, n_channels = 75, 3
    y = _trajectory(span, n_channels)
    geometry = EegMsGeometry(window_s=window_s, hop_s=hop_s)

    spec = EegMsSpec(weight=1.0, span_s=span / _FS, window_s=window_s, hop_s=hop_s)
    power = EegMsLoss.from_spec(spec, _FS).windowed_power(
        torch.as_tensor(y[None], dtype=torch.float64), _ctx(n_channels)
    )

    target = log_observable(y, geometry, _FS)
    np.testing.assert_allclose(target[..., 0], np.log(power.numpy()[0] + LOG_FLOOR).T, rtol=1e-12, atol=1e-12)
    assert target.shape == (geometry.n_frames(span, _FS), n_channels, 1)


@pytest.mark.parametrize("span", [50, 60, 75, 100, 125])
@pytest.mark.parametrize(("n_segment", "n_hop"), [(50, 25), (25, 12), (50, 50), (20, 5)])
def test_resolved_frame_count_matches_the_geometry(span: int, n_segment: int, n_hop: int) -> None:
    """The Frame count the builder produces is the one the geometry derives, across ``(H, L, R)``."""
    geometry = StftGeometry(n_segment=n_segment, n_hop=n_hop)
    target = log_observable(_trajectory(span, 2), geometry, _FS)
    assert target.shape[0] == geometry.n_frames(span, _FS)
    assert len(geometry.frame_supports(span, _FS)) == target.shape[0]


def test_control_aggregation_averages_over_each_frames_support() -> None:
    """Each row of the aggregation operator is the uniform mean over exactly that Frame's Segment."""
    geometry = StftGeometry(n_segment=50, n_hop=25)
    operator = control_means(geometry, 75, _FS)

    assert operator.shape == (2, 75)
    np.testing.assert_allclose(operator.sum(axis=1), 1.0)
    for row, (start, end) in zip(operator, geometry.frame_supports(75, _FS), strict=True):
        assert np.count_nonzero(row) == end - start
        np.testing.assert_allclose(row[start:end], 1.0 / (end - start))


def test_forecast_of_a_frame_ignores_controls_landing_after_its_segment(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """Frame ``m``'s forecast is invariant to Control Currents after its Segment ends.

    Structurally guaranteed by the recursion, so this catches an indexing error in the control
    aggregation rather than a modelling one.
    """
    geometry = StftGeometry(n_segment=8, n_hop=4)
    horizon = 20
    art = make_observable_artifact(geometry, horizon=horizon, n_y=3, n_u=2)
    rng = np.random.default_rng(_SEED + 1)

    state = art.prime(rng.standard_normal((3, art.n_channels)), rng.standard_normal((2, art.n_controls)))
    u = rng.standard_normal((horizon, art.n_controls))
    base = art.forecast(state, u)

    supports = geometry.frame_supports(horizon, art.fs)
    assert len(supports) > 1, "the property is vacuous with a single frame"
    for m, (_, end) in enumerate(supports[:-1]):
        perturbed = u.copy()
        perturbed[end:] += 10.0
        np.testing.assert_allclose(art.forecast(state, perturbed)[: m + 1], base[: m + 1], rtol=1e-12, atol=1e-12)
        assert not np.allclose(art.forecast(state, perturbed)[m + 1 :], base[m + 1 :])


def test_artifact_round_trip_preserves_weights_standardizers_and_geometry(
    tmp_path: object, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """Save/load returns byte-identical weights, standardizers and the recorded Observable geometry."""
    geometry = StftGeometry(n_segment=8, n_hop=4, n_bin_pool=2, kernel="hann", kernel_width=2, band_hz=(2.0, 20.0))
    art = make_observable_artifact(geometry, horizon=20)

    path = tmp_path / "observable"  # ty: ignore[unsupported-operator]
    art.save(path)
    loaded = ObservableArtifact.load(path)

    assert loaded.geometry == geometry
    assert loaded.meta == art.meta
    for saved_block, loaded_block in ((art.lift, loaded.lift), (art.transition, loaded.transition)):
        for (w, b), (w2, b2) in zip(saved_block, loaded_block, strict=True):
            np.testing.assert_array_equal(w, w2)
            np.testing.assert_array_equal(b, b2)
    np.testing.assert_array_equal(art.readout[0], loaded.readout[0])
    np.testing.assert_array_equal(art.readout[1], loaded.readout[1])
    for saved_std, loaded_std in (
        (art.y_std, loaded.y_std),
        (art.u_std, loaded.u_std),
        (art.l_std, loaded.l_std),
    ):
        np.testing.assert_array_equal(saved_std.center, loaded_std.center)
        np.testing.assert_array_equal(saved_std.scale, loaded_std.scale)


def test_envelope_reference_pools_power_before_the_log() -> None:
    """The reference is pooled as power and logged once, exactly as the measured target is."""
    rng = np.random.default_rng(_SEED + 2)
    geometry = StftGeometry(n_segment=50, n_hop=25, n_bin_pool=5)
    envelope = PsdEnvelope(power=rng.uniform(0.1, 10.0, (3, 26)), fs=_FS, window=50, hop=25)

    reference = envelope_log_reference(envelope, geometry, _FS)
    expected = np.log(envelope.power[:, 1:26].reshape(3, 5, 5).mean(axis=2))

    assert reference.shape == (3, geometry.n_values(_FS))
    np.testing.assert_allclose(reference, expected, rtol=1e-12)


def test_eeg_ms_envelope_is_the_time_domain_mean_square() -> None:
    """The ``eeg_ms`` reference is the windowed mean square of the healthy trajectories, DC included."""
    y = _trajectory(150, 3)
    geometry = EegMsGeometry(window_s=1.0, hop_s=0.5)
    envelope = MsEnvelope(power=windowed_mean_square(y, window=50, hop=25).mean(axis=0), fs=_FS, window=50, hop=25)

    reference = envelope_log_reference(envelope, geometry, _FS)
    assert reference.shape == (3, 1)
    # Parseval over the spectral envelope would drop DC; the time-domain mean square keeps it.
    spectral_power = compute_periodograms(y, fs=_FS, window=50, hop=25)[..., 1:].sum(axis=(0, 2))
    assert not np.allclose(np.exp(reference[:, 0]), spectral_power)
