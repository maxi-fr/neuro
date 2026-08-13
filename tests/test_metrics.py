from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.artifacts import accumulate_rollout_errors, evaluate_rollouts, nmse  # noqa: E402
from neuro.nn_training import split_data_files  # noqa: E402
from neuro.prediction import AutoregressivePredictor, MLPArtifact  # noqa: E402
from neuro.transforms import Pipeline, Standardizer  # noqa: E402

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


def _tiny_mlp_artifact(*, n_y: int, n_u: int, horizon: int, n_eeg: int, n_controls: int) -> MLPArtifact:
    """A randomly-initialised MLP artifact, enough to free-run a rollout."""
    mlp = eqx.nn.MLP(
        in_size=n_y * n_eeg + n_u * n_controls,
        out_size=n_eeg,
        width_size=4,
        depth=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )
    model = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_eeg, n_controls=n_controls, activation="relu"
    )
    unit = Pipeline((Standardizer(center=np.zeros(n_eeg), scale=np.ones(n_eeg)),))
    return MLPArtifact(
        model=model,
        dt=0.01,
        downsample=1,
        y_pipeline=unit,
        u_pipeline=Pipeline((Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls)),)),
    )


def test_evaluate_rollouts_pools_its_own_per_step_curve() -> None:
    horizon, n_eeg, n_controls = 6, 3, 2
    art = _tiny_mlp_artifact(n_y=4, n_u=2, horizon=horizon, n_eeg=n_eeg, n_controls=n_controls)
    trajs = [(_RNG.normal(size=(200, n_controls)), _RNG.normal(size=(200, n_eeg))) for _ in range(2)]

    rollout = evaluate_rollouts(art, trajs, horizon)
    _, power, _ = accumulate_rollout_errors(art, trajs, horizon, stride=25)

    assert rollout.per_step.shape == (horizon,)
    assert rollout.pooled == pytest.approx(float(np.average(rollout.per_step, weights=power)))


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
