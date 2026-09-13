import numpy as np

from neuro.prediction_spectra import compare_spectra
from neuro.run_view import PredictionSeries, ViewSettings


def test_waveform_spectra_compare_identical_future_support_and_scaling() -> None:
    t = np.arange(101, dtype=np.float64) / 100
    observed = np.sin(2 * np.pi * 10 * t)[:, None]
    predictions = np.full((101, 101, 1), np.nan)
    predictions[0] = observed * 2
    series = PredictionSeries(t, predictions, observed, 0.01)
    result = compare_spectra(series, ViewSettings(decision=0, spectral_steps=100))
    assert result is not None
    assert result.frequencies[np.argmax(result.observed[0])] == 10
    np.testing.assert_allclose(result.predicted, 4 * result.observed, atol=1e-12)
    assert compare_spectra(series, ViewSettings(decision=0.5, spectral_steps=100)) is None


def test_observable_spectra_select_future_frame_without_fourier_transform() -> None:
    t = np.arange(5, dtype=np.float64) * 0.1
    observed = np.arange(30, dtype=np.float64).reshape(5, 6)
    predictions = np.zeros((5, 3, 6))
    predictions[1, 2] = observed[3] + 1
    series = PredictionSeries(t, predictions, observed, 0.1, np.array([0.0, 5.0, 10.0]))
    result = compare_spectra(series, ViewSettings(decision=0.1, spectral_frame=2))
    assert result is not None
    assert result.units == "log power"
    np.testing.assert_array_equal(result.observed, observed[3].reshape(2, 3))
    np.testing.assert_array_equal(result.predicted, result.observed + 1)
