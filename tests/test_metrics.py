from __future__ import annotations

import numpy as np
import pytest
import torch

from neuro.predictor.data import split_data_files
from neuro.predictor.evaluation import (
    accumulate_rollout_errors,
    evaluate_log_energy,
    evaluate_rollouts,
    nmse,
    window_energy,
)
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

_RNG = np.random.default_rng(0)


def test_nmse_of_zero_predictor_is_one() -> None:
    y = _RNG.normal(size=(64, 3))
    assert nmse((y**2).sum(), (y**2).sum()) == pytest.approx(1.0)


def test_nmse_matches_variance_normalization_on_centered_data() -> None:
    y = _RNG.normal(size=(64, 3))
    y -= y.mean()
    y_pred = y + 0.1 * _RNG.normal(size=y.shape)
    sq_err = float(((y - y_pred) ** 2).sum())

    energy_normalized = float(nmse(sq_err, float((y**2).sum())))
    variance_normalized = float(((y - y_pred) ** 2).mean()) / float(np.var(y))

    assert energy_normalized == pytest.approx(variance_normalized)


def test_nmse_pooled_is_power_weighted_mean_of_per_step() -> None:
    sq_err = np.array([1.0, 4.0, 9.0])
    power = np.array([10.0, 20.0, 5.0])

    per_step = nmse(sq_err, power)
    pooled = float(nmse(sq_err.sum(), power.sum()))

    assert pooled == pytest.approx(float(np.average(per_step, weights=power)))


def test_nmse_is_infinite_where_the_true_signal_is_silent() -> None:
    assert np.isinf(nmse(np.array([1.0, 0.0]), np.array([0.0, 0.0]))).all()


def _tiny_mlp_model(*, n_y: int, n_u: int, horizon: int, n_eeg: int, n_controls: int) -> AutoregressiveMLP:
    """A randomly-initialised MLP module, enough to free-run a rollout."""
    unit = Standardizer(center=np.zeros(n_eeg), scale=np.ones(n_eeg))
    return AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_eeg,
        n_controls=n_controls,
        hidden_size=4,
        depth=1,
        activation="relu",
        dt=0.01,
        y_std=unit,
        u_std=Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls)),
    )


def _silent_mlp_model(*, n_y: int, n_u: int, horizon: int, n_eeg: int, n_controls: int) -> AutoregressiveMLP:
    """An MLP module whose every weight is zero, so it free-runs to exactly zero.

    The residual skip is off: with it, a zero-weight stack would repeat the primed history's last
    sample instead of staying silent, and these tests are about the metrics on a zero predictor.
    """
    model = _tiny_mlp_model(n_y=n_y, n_u=n_u, horizon=horizon, n_eeg=n_eeg, n_controls=n_controls)
    model.residual = False
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
    return model


def test_evaluate_rollouts_pools_its_own_per_step_curve() -> None:
    horizon, n_eeg, n_controls = 6, 3, 2
    model = WaveformMLPModel.from_checkpoint(
        *_tiny_mlp_model(n_y=4, n_u=2, horizon=horizon, n_eeg=n_eeg, n_controls=n_controls).to_checkpoint()
    )
    trajs = [(_RNG.normal(size=(200, n_controls)), _RNG.normal(size=(200, n_eeg))) for _ in range(2)]

    rollout = evaluate_rollouts(model, trajs, horizon)
    _, power, _ = accumulate_rollout_errors(model, trajs, horizon, stride=25)

    assert rollout.per_step.shape == (horizon,)
    assert rollout.pooled == pytest.approx(float(np.average(rollout.per_step, weights=power)))


def test_window_energy_is_the_cross_channel_mean_square_per_trailing_window() -> None:
    y = np.arange(2 * 6 * 2, dtype=np.float64).reshape(2, 6, 2)

    energy = window_energy(y, window_steps=4, hop_steps=2)

    assert energy.shape == (2, 2)
    assert energy[0, 0] == pytest.approx(float((y[0, 0:4] ** 2).mean()))
    assert energy[1, 1] == pytest.approx(float((y[1, 2:6] ** 2).mean()))


def test_log_energy_is_zero_when_the_prediction_matches() -> None:
    """A silent model on a silent plant: both energies are floored identically, so the error is 0."""
    horizon, n_eeg, n_controls = 6, 3, 2
    model = WaveformMLPModel.from_checkpoint(
        *_silent_mlp_model(n_y=4, n_u=2, horizon=horizon, n_eeg=n_eeg, n_controls=n_controls).to_checkpoint()
    )
    trajs = [(np.zeros((200, n_controls)), np.zeros((200, n_eeg))) for _ in range(2)]

    score = evaluate_log_energy(model, trajs, horizon, window_steps=4, hop_steps=2)

    assert score.pooled == pytest.approx(0.0)
    assert score.per_position.shape == (2,)


def test_log_energy_separates_a_silent_predictor_that_nmse_cannot() -> None:
    """The motivating case: NMSE reads exactly 1.0 -- its saturation value -- and log-energy does not."""
    horizon, n_eeg, n_controls = 6, 3, 2
    model = WaveformMLPModel.from_checkpoint(
        *_silent_mlp_model(n_y=4, n_u=2, horizon=horizon, n_eeg=n_eeg, n_controls=n_controls).to_checkpoint()
    )
    trajs = [(_RNG.normal(size=(200, n_controls)), _RNG.normal(size=(200, n_eeg))) for _ in range(2)]

    assert evaluate_rollouts(model, trajs, horizon).pooled == pytest.approx(1.0)

    score = evaluate_log_energy(model, trajs, horizon, window_steps=4, hop_steps=2)
    assert np.isfinite(score.pooled)
    assert score.pooled > 1.0


def test_log_energy_rejects_a_horizon_shorter_than_one_window() -> None:
    model = WaveformMLPModel.from_checkpoint(
        *_tiny_mlp_model(n_y=4, n_u=2, horizon=6, n_eeg=3, n_controls=2).to_checkpoint()
    )
    trajs = [(_RNG.normal(size=(200, 2)), _RNG.normal(size=(200, 3))) for _ in range(2)]

    with pytest.raises(ValueError, match="shorter than the energy window"):
        evaluate_log_energy(model, trajs, 6, window_steps=8, hop_steps=2)


def test_split_data_files_holds_out_the_tail() -> None:
    files = [f"traj_{i}.npz" for i in range(10)]
    train, val = split_data_files(files, 0.8)
    assert (train, val) == (files[:8], files[8:])


@pytest.mark.parametrize("train_split", [0.0, 1.0])
def test_split_data_files_keeps_both_sides_non_empty(train_split: float) -> None:
    train, val = split_data_files(["a.npz", "b.npz", "c.npz"], train_split)
    assert train
    assert val


def test_split_data_files_rejects_a_single_file() -> None:
    with pytest.raises(ValueError, match="at least 2 trajectory files"):
        split_data_files(["only.npz"], 0.8)
