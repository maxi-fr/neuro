import importlib.util
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal as sps
import torch
import yaml
from scipy.signal.windows import hann

from neuro.config import StftGeometry, StftSpec
from neuro.control.costs import jax_compute_log_power_frames
from neuro.predictor.losses import LossContext, StftLoss
from neuro.spectral import (
    LOG_FLOOR,
    MsEnvelope,
    ObservableEnvelope,
    PsdEnvelope,
    compute_log_power_frames,
    compute_periodograms,
    hinge_penalty,
    windowed_mean_square,
)

_SEED = 11


def test_periodograms_match_scipy_per_window() -> None:
    """Each returned window equals an undetrended scipy.signal.periodogram of that slice alone."""
    rng = np.random.default_rng(_SEED)
    y = rng.standard_normal((200, 3))
    fs, window, hop = 50.0, 50, 25

    power = compute_periodograms(y, fs=fs, window=window, hop=hop)

    assert power.shape == ((200 - window) // hop + 1, 3, window // 2 + 1)
    w_hann = hann(window, sym=False)
    for m in range(power.shape[0]):
        _, expected = sps.periodogram(
            y[m * hop : m * hop + window, :], fs=fs, window=w_hann, detrend=False, axis=0, scaling="density"
        )
        np.testing.assert_allclose(power[m], expected.T, rtol=1e-12, atol=1e-15)


def test_dc_offset_is_retained_rather_than_detrended() -> None:
    """No per-segment detrend: a constant offset lands in the DC bin at its full analytic power.

    A detrended periodogram would put ~0 there. The MPC cost drops bin 0 at the use site instead,
    so the high-pass corner never moves with the segment length.
    """
    fs, window = 50.0, 50
    offset = 3.0
    y = np.full((window, 1), offset)

    power = compute_periodograms(y, fs=fs, window=window, hop=window)

    w_hann = hann(window, sym=False)
    expected_dc = (offset * w_hann.sum()) ** 2 / (fs * np.sum(w_hann**2))
    np.testing.assert_allclose(power[0, 0, 0], expected_dc, rtol=1e-12)
    assert power[0, 0, 0] > 1.0


def test_periodograms_are_not_averaged_over_windows() -> None:
    """A burst confined to one window stays confined to it, rather than being diluted across windows."""
    fs, window, hop = 50.0, 50, 25
    y = np.zeros((100, 1))
    y[:window, 0] = 5.0 * np.sin(2 * np.pi * 5.0 * np.arange(window) / fs)

    power = compute_periodograms(y, fs=fs, window=window, hop=hop)

    burst_bin = power[:, 0, 5]
    assert burst_bin[0] > 100 * burst_bin[-1]


def test_periodogram_resolves_a_known_tone() -> None:
    """A 5 Hz tone at fs=50 with a 50-sample window lands in the 5 Hz bin (df = 1 Hz)."""
    fs, window = 50.0, 50
    t = np.arange(window) / fs
    y = np.sin(2 * np.pi * 5.0 * t)[:, None]

    power = compute_periodograms(y, fs=fs, window=window, hop=window)

    assert int(np.argmax(power[0, 0])) == 5


def test_short_signal_yields_no_windows() -> None:
    """A signal shorter than one window produces an empty, correctly shaped result."""
    power = compute_periodograms(np.zeros((10, 4)), fs=50.0, window=50, hop=25)
    assert power.shape == (0, 4, 26)


def test_hinge_penalty_is_zero_under_the_envelope() -> None:
    """The one-sided hinge is exactly 0 while power stays under the reference everywhere."""
    power = np.full((3, 2, 5), 1e-3)
    assert hinge_penalty(power, np.full((2, 5), 1e3)) == 0.0


def test_hinge_penalty_ignores_quiet_bins() -> None:
    """Power far below the envelope contributes nothing, so the cost never asks for more power."""
    reference = np.full((2, 5), 1.0)
    over = np.full((1, 2, 5), np.e)
    quiet = over.copy()
    quiet[0, 0, 0] = 1e-12

    assert hinge_penalty(quiet, reference) < hinge_penalty(over, reference)
    np.testing.assert_allclose(hinge_penalty(over, reference), 1.0, rtol=1e-6)


def test_envelope_load_round_trips(tmp_path: Path) -> None:
    """PsdEnvelope.load recovers the geometry written alongside the envelope."""
    path = tmp_path / "psd.npz"
    np.savez(path, Pref=np.ones((4, 26)), freqs=np.arange(26), fs=50.0, L=50, R=25)

    envelope = PsdEnvelope.load(path)

    assert (envelope.fs, envelope.window, envelope.hop) == (50.0, 50, 25)
    assert envelope.power.shape == (4, 26)


def test_envelope_load_rejects_subset_bins(tmp_path: Path) -> None:
    """A reference whose bin count contradicts its window is rejected, never silently subset."""
    path = tmp_path / "psd.npz"
    np.savez(path, Pref=np.ones((4, 10)), freqs=np.arange(10), fs=50.0, L=50, R=25)

    with pytest.raises(ValueError, match="must not subset bins"):
        PsdEnvelope.load(path)


def test_compute_log_power_frames_order_of_operations() -> None:
    """Canonical reduction applies Hann, density, DC exclusion, band, pooling, kernel, floor, log in order."""
    rng = np.random.default_rng(_SEED)
    n_samples, n_channels, fs = 150, 2, 50.0
    y = rng.standard_normal((n_samples, n_channels))

    geometry = StftGeometry(
        n_segment=40,
        n_hop=20,
        band_hz=(4.0, 16.0),
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )

    frames = compute_log_power_frames(y, geometry, fs=fs)
    assert frames.ndim == 3
    assert frames.shape[1] == n_channels
    assert frames.shape[2] == geometry.n_values(fs)

    # Step-by-step manual replica
    n_raw_frames = (n_samples - 40) // 20 + 1
    w_hann = hann(40, sym=False)
    segments = np.stack([y[m * 20 : m * 20 + 40, :] for m in range(n_raw_frames)], axis=0)
    _, raw_power = sps.periodogram(segments, fs=fs, window=w_hann, detrend=False, axis=1, scaling="density")
    raw_power = np.asarray(raw_power, dtype=np.float64).transpose(0, 2, 1)

    bin_lo, bin_hi = geometry.bin_range(fs)
    assert bin_lo >= 1  # DC excluded
    band_power = raw_power[:, :, bin_lo:bin_hi]

    n_groups = band_power.shape[-1] // 2
    pooled_power = band_power[:, :, : n_groups * 2].reshape(n_raw_frames, n_channels, n_groups, 2).mean(axis=-1)

    # Hann kernel weights of width 3
    k_weights = hann(5, sym=True)[1:-1]
    k_weights = k_weights / k_weights.sum()
    n_frames = n_raw_frames - 3 + 1
    smoothed_power = np.stack(
        [np.sum(pooled_power[i : i + 3] * k_weights[:, None, None], axis=0) for i in range(n_frames)],
        axis=0,
    )
    expected_frames = np.log(smoothed_power + LOG_FLOOR)

    assert frames.shape == expected_frames.shape
    np.testing.assert_allclose(frames, expected_frames, rtol=1e-12, atol=1e-15)


def test_frame_sample_support_and_counts() -> None:
    """A Frame's sample support is (kernel_width - 1) * hop + segment and reports exact counts."""
    rng = np.random.default_rng(_SEED + 1)
    fs, n_segment, n_hop, width = 50.0, 50, 25, 3
    geometry = StftGeometry(
        n_segment=n_segment,
        n_hop=n_hop,
        kernel="hann",
        kernel_width=width,
    )

    support = (width - 1) * n_hop + n_segment
    assert support == 100
    assert geometry.sample_support_steps(fs) == support

    # Under support: 0 frames
    y_short = rng.standard_normal((99, 2))
    assert compute_log_power_frames(y_short, geometry, fs=fs).shape == (0, 2, geometry.n_values(fs))
    assert geometry.n_frames(99, fs) == 0

    # Exact support: exactly 1 frame
    y_exact = rng.standard_normal((100, 2))
    assert compute_log_power_frames(y_exact, geometry, fs=fs).shape == (1, 2, geometry.n_values(fs))
    assert geometry.n_frames(100, fs) == 1

    # Exact support + 2 hops: exactly 3 frames
    y_3 = rng.standard_normal((150, 2))
    assert compute_log_power_frames(y_3, geometry, fs=fs).shape == (3, 2, geometry.n_values(fs))
    assert geometry.n_frames(150, fs) == 3


@pytest.mark.parametrize("band_hz", [None, (3.0, 12.0), (8.0, 20.0)])
@pytest.mark.parametrize("n_bin_pool", [1, 2, 3])
@pytest.mark.parametrize(("kernel", "width"), [("boxcar", 1), ("triangular", 2), ("hann", 4)])
@pytest.mark.parametrize("n_segment", [32, 33])
def test_torch_reduction_agrees_with_canonical_numpy(
    band_hz: tuple[float, float] | None,
    n_bin_pool: int,
    kernel: Literal["boxcar", "triangular", "hann"],
    width: int,
    n_segment: int,
) -> None:
    """The torch reduction used by spectral training Loss matches canonical NumPy to float tolerance."""
    rng = np.random.default_rng(_SEED + 2)
    fs, n_hop, n_span, n_channels = 50.0, 8, 80, 2
    y = rng.standard_normal((n_span, n_channels))

    geometry = StftGeometry(
        n_segment=n_segment,
        n_hop=n_hop,
        band_hz=band_hz,
        n_bin_pool=n_bin_pool,
        kernel=kernel,
        kernel_width=width,
    )

    numpy_frames = compute_log_power_frames(y, geometry, fs=fs)

    stft_loss = StftLoss(
        weight=1.0,
        span_steps=n_span,
        start_epoch=0,
        geometry=geometry,
    )
    ctx = LossContext(
        y_center=torch.zeros(n_channels, dtype=torch.float64),
        y_scale=torch.ones(n_channels, dtype=torch.float64),
        fs=fs,
        epoch=0,
    )
    # torch input is (batch=1, span, channels)
    x_tensor = torch.as_tensor(y[None, :, :], dtype=torch.float64)
    # log_spectrogram output: (batch, channel, frame, bin)
    torch_frames = stft_loss.log_spectrogram(x_tensor, ctx).squeeze(0).permute(1, 0, 2).detach().numpy()

    assert torch_frames.shape == numpy_frames.shape
    np.testing.assert_allclose(torch_frames, numpy_frames, rtol=1e-10, atol=1e-12)


def test_jax_reduction_agrees_with_canonical_numpy() -> None:
    """The JAX reduction inside the waveform spectral hinge matches canonical NumPy to float tolerance."""
    rng = np.random.default_rng(_SEED + 3)
    n_samples, n_channels, fs, window, hop = 120, 3, 50.0, 40, 20
    y = rng.standard_normal((n_samples, n_channels))

    geom = StftGeometry(n_segment=window, n_hop=hop)
    numpy_frames = compute_log_power_frames(y, geom, fs=fs)

    jax_frames = jax_compute_log_power_frames(jnp.asarray(y), fs=fs, window=window, hop=hop)

    assert jax_frames.shape == numpy_frames.shape
    np.testing.assert_allclose(np.asarray(jax_frames), numpy_frames, rtol=1e-10, atol=1e-12)


def test_observable_envelope_save_load_round_trip(tmp_path: Path) -> None:
    """ObservableEnvelope records its full geometry and loads back identically alongside legacy envelopes."""
    path = tmp_path / "healthy.npz"
    fs = 50.0
    geom = StftGeometry(
        n_segment=50,
        n_hop=25,
        band_hz=(3.0, 12.0),
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )
    n_values = geom.n_values(fs)
    pref_frames = np.ones((4, n_values)) * 2.5
    pref_psd = np.ones((4, 26))
    pref_ms = np.ones(4)

    np.savez_compressed(
        path,
        Pref=pref_psd,
        Pref_ms=pref_ms,
        Pref_frames=pref_frames,
        fs=fs,
        L=geom.n_segment,
        R=geom.n_hop,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )

    # ObservableEnvelope load
    obs_env = ObservableEnvelope.load(path)
    assert obs_env.fs == fs
    assert obs_env.geometry == geom
    np.testing.assert_allclose(obs_env.power, pref_frames)

    # Legacy loaders still work on same file
    psd_env = PsdEnvelope.load(path)
    assert (psd_env.fs, psd_env.window, psd_env.hop) == (fs, 50, 25)
    np.testing.assert_allclose(psd_env.power, pref_psd)

    ms_env = MsEnvelope.load(path)
    assert (ms_env.fs, ms_env.window, ms_env.hop) == (fs, 50, 25)
    np.testing.assert_allclose(ms_env.power, pref_ms)


def test_observable_envelope_rejects_corrupted_artifact(tmp_path: Path) -> None:
    """ObservableEnvelope raises when Pref_frames is missing or value count mismatches geometry."""
    path_no_frames = tmp_path / "no_frames.npz"
    np.savez(path_no_frames, Pref=np.ones((4, 26)), fs=50.0, L=50, R=25)
    with pytest.raises(ValueError, match="carries no Observable frames array"):
        ObservableEnvelope.load(path_no_frames)

    path_bad_dim = tmp_path / "bad_dim.npz"
    np.savez(
        path_bad_dim,
        Pref_frames=np.ones((4, 99)),
        fs=50.0,
        n_segment=50,
        n_hop=25,
        band_hz=np.asarray([-1.0, -1.0]),
        n_bin_pool=1,
        kernel="boxcar",
        kernel_width=1,
    )
    with pytest.raises(ValueError, match="values per channel but its geometry implies"):
        ObservableEnvelope.load(path_bad_dim)


def test_build_healthy_psd_computes_observable_quantile_and_legacy(tmp_path: Path) -> None:
    """build_healthy_psd quantiles over canonical Frames and writes all three envelope arrays."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "build_healthy_psd.py"
    spec = importlib.util.spec_from_file_location("build_healthy_psd_mod", script_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_healthy_psd = mod.build_healthy_psd

    rng = np.random.default_rng(_SEED + 5)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    n_samples, n_channels, n_controls = 200, 3, 2
    # Save two synthetic trajectories
    for i in range(2):
        y_raw = rng.standard_normal((n_samples, n_channels))
        u_raw = rng.standard_normal((n_samples, n_controls))
        np.savez(data_dir / f"traj_{i}.npz", allow_pickle=True, **{"sensor_0.y_mea": y_raw, "controller.u": u_raw})

    config_path = tmp_path / "cfg.yaml"
    config_dict = {
        "experiments": [
            {
                "dynamics": {"dt": 0.02},  # 50 Hz
                "estimator": {"downsample": 1},
            }
        ]
    }
    config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
    output_path = tmp_path / "healthy_psd.npz"

    geom = StftGeometry(
        n_segment=40,
        n_hop=20,
        band_hz=(3.0, 15.0),
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )

    build_healthy_psd(
        config_path=config_path,
        data_dir=data_dir,
        output_path=output_path,
        quantile=0.85,
        geometry=geom,
    )

    # Verify ObservableEnvelope loads
    obs_env = ObservableEnvelope.load(output_path)
    assert obs_env.geometry == geom
    assert obs_env.fs == 50.0
    assert obs_env.power.shape == (n_channels, geom.n_values(50.0))

    # Verify PsdEnvelope and MsEnvelope still load
    psd_env = PsdEnvelope.load(output_path)
    assert psd_env.power.shape == (n_channels, geom.n_segment // 2 + 1)

    ms_env = MsEnvelope.load(output_path)
    assert ms_env.power.shape == (n_channels,)
