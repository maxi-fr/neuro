"""Pin the torch AutoregressiveMLP against the CasADi bridge and guard the torch-free seam."""

from __future__ import annotations

import itertools
import subprocess
import sys
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
import torch

from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN, _LATENT = 5, 2, 6, 3


def _random_layers(
    rng: np.random.Generator, in_size: int, out_size: int, hidden_size: int, depth: int
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(W, b)`` pairs for a ``depth``-hidden-layer MLP, scaled so activations stay O(1)."""
    sizes = [in_size, *[hidden_size] * depth, out_size]
    return tuple(
        (rng.standard_normal((n_out, n_in)) / np.sqrt(n_in), rng.standard_normal(n_out) * 0.1)
        for n_in, n_out in itertools.pairwise(sizes)
    )


def _build_artifact(latent_dim: int | None, depth: int, activation: Activation) -> MLPArtifact:
    """A random artifact with nontrivial standardizers and, when ``latent_dim`` is set, a PCA step."""
    rng = np.random.default_rng(_SEED)
    k = _N_EEG if latent_dim is None else latent_dim

    steps: list[Standardizer | PCAProjection] = [
        Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    ]
    if latent_dim is not None:
        q, _ = np.linalg.qr(rng.standard_normal((_N_EEG, _N_EEG)))
        steps.append(PCAProjection(basis=np.ascontiguousarray(q[:, :latent_dim].T), mean=rng.standard_normal(_N_EEG)))

    u_pipeline = Pipeline(
        (Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),)
    )
    return MLPArtifact(
        layers=_random_layers(rng, _N_Y * k + _N_U * _N_CONTROLS, k, _HIDDEN, depth),
        activation=activation,
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=k,
        n_controls=_N_CONTROLS,
        dt=0.01,
        downsample=2,
        y_pipeline=Pipeline(tuple(steps)),
        u_pipeline=u_pipeline,
    )


def _casadi_rollout(art: MLPArtifact, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray) -> FloatArray:
    """Chain ``f_step``/``f_out`` over the horizon on RAW controls -> raw EEG ``(horizon, n_eeg)``."""
    sym = NNSymbolicModel(art)
    x = np.concatenate([art.encode(y_hist[-art.n_y :]).flatten(), u_hist[-art.n_u :].flatten()])
    preds = []
    for t in range(art.horizon):
        x = np.array(ca.DM(sym.f_step(x, u_future[t]))).flatten()
        preds.append(np.array(ca.DM(sym.f_out(x))).flatten())
    return np.stack(preds)


def _model_space_features(art: MLPArtifact, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray) -> FloatArray:
    """Assemble the torch input row, controls transformed as ``transform_features`` does."""
    return np.concatenate(
        [
            art.encode(y_hist[-art.n_y :]).flatten(),
            art.u_pipeline.transform(u_hist[-art.n_u :]).flatten(),
            art.u_pipeline.transform(u_future).flatten(),
        ]
    )


def _context(art: MLPArtifact, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls."""
    rng = np.random.default_rng(seed)
    hist = max(art.n_y, art.n_u)
    return (
        rng.standard_normal((hist, art.n_eeg_channels)),
        rng.standard_normal((hist, art.n_controls)),
        rng.standard_normal((art.horizon, art.n_controls)),
    )


@pytest.mark.parametrize("latent_dim", [None, _LATENT])
def test_prime_rollout_matches_the_training_window_at_the_same_index(latent_dim: int | None) -> None:
    """``prime`` + ``rollout`` reproduce the torch rollout of the training window over the same targets.

    Regression: a ``prime`` state ends its y- and u-windows at the same step, so ``rollout`` must
    predict once *before* shifting a future control in. Shifting first ran evaluation one control
    step ahead of the alignment ``build_dataset_for_trajectory`` trains on, which silently changed
    the rollout NMSE that the sweep optimises.
    """
    art = _build_artifact(latent_dim, 2, "softplus")
    n_y, n_u, horizon = art.n_y, art.n_u, art.horizon
    rng = np.random.default_rng(_SEED + 9)
    t0 = max(n_y, n_u) + 3
    y_raw = rng.standard_normal((t0 + horizon, art.n_eeg_channels))
    u_raw = rng.standard_normal((t0 + horizon, art.n_controls))

    # build_dataset_for_trajectory pairs the window at index k with targets y[k+1 : k+1+horizon],
    # so the window predicting y[t0 : t0+horizon] is the one at k = t0 - 1.
    k = t0 - 1
    x = np.concatenate(
        [
            art.encode(y_raw[k - n_y + 1 : k + 1]).flatten(),
            art.u_pipeline.transform(u_raw[k - n_u : k]).flatten(),
            art.u_pipeline.transform(u_raw[k : k + horizon]).flatten(),
        ]
    )
    model = AutoregressiveMLP.from_artifact(art)
    pred = model(torch.as_tensor(x)[None, :]).detach().numpy().reshape(horizon, art.n_channels)
    want = art.decode(pred)

    got = art.rollout(art.prime(y_raw[:t0], u_raw[:t0]), u_raw[t0 : t0 + horizon])

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("latent_dim", [None, _LATENT])
def test_torch_rollout_matches_casadi(latent_dim: int | None, depth: int, activation: Activation) -> None:
    """The torch rollout reproduces the CasADi f_step/f_out chain, closing torch == CasADi == JAX.

    The u convention is the trap: ``NNSymbolicModel`` standardizes raw controls internally, while
    the torch module consumes controls already in model space.
    """
    art = _build_artifact(latent_dim, depth, activation)
    y_hist, u_hist, u_future = _context(art, _SEED + 1)

    want = _casadi_rollout(art, y_hist, u_hist, u_future)

    model = AutoregressiveMLP.from_artifact(art)
    x = torch.as_tensor(_model_space_features(art, y_hist, u_hist, u_future))[None, :]
    pred = model(x).detach().numpy().reshape(art.horizon, art.n_channels)
    got = art.decode(pred)

    assert got.shape == (art.horizon, art.n_eeg_channels)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_forward_is_row_independent() -> None:
    """Stacking two contexts into one batch predicts the same as running them one at a time."""
    art = _build_artifact(_LATENT, 2, "tanh")
    model = AutoregressiveMLP.from_artifact(art)

    rows = [_model_space_features(art, *_context(art, _SEED + offset)) for offset in (2, 3)]
    batched = model(torch.as_tensor(np.stack(rows))).detach().numpy()
    singles = np.concatenate([model(torch.as_tensor(row)[None, :]).detach().numpy() for row in rows])

    assert batched.shape == (2, art.horizon * art.n_channels)
    np.testing.assert_array_equal(batched, singles)


@pytest.mark.parametrize("latent_dim", [None, _LATENT])
def test_decode_buffers_carry_the_projection(latent_dim: int | None) -> None:
    """The PCA decode arrays ride on the module as float64 buffers (``None`` without a projection)."""
    art = _build_artifact(latent_dim, 2, "relu")
    model = AutoregressiveMLP.from_artifact(art)

    if latent_dim is None:
        assert model.decode_basis is None
        assert model.decode_mean is None
        return

    pca = art.y_pipeline.pca
    assert pca is not None
    assert model.decode_basis.dtype == torch.float64
    assert model.decode_basis.shape == (latent_dim, _N_EEG)
    assert model.decode_mean.shape == (_N_EEG,)
    np.testing.assert_array_equal(np.asarray(model.decode_basis), pca.basis)
    np.testing.assert_array_equal(np.asarray(model.decode_mean), pca.mean)


@pytest.mark.parametrize("depth", [0, 2])
def test_artifact_round_trip_is_exact(depth: int) -> None:
    """``from_artifact`` then ``to_artifact`` returns bit-identical weights and metadata."""
    art = _build_artifact(_LATENT, depth, "softplus")
    got = AutoregressiveMLP.from_artifact(art).to_artifact(art.dt, art.downsample, art.y_pipeline, art.u_pipeline)

    assert got.meta == art.meta
    for (got_w, got_b), (want_w, want_b) in zip(got.layers, art.layers, strict=True):
        np.testing.assert_array_equal(got_w, want_w)
        np.testing.assert_array_equal(got_b, want_b)


@pytest.mark.parametrize(
    "module",
    [
        "neuro.predictor.artifact",
        "neuro.predictor.data",
        "neuro.nn_predictor_casadi",
        "neuro.control",
        "neuro.artifacts",
    ],
)
def test_control_path_never_imports_torch(module: str) -> None:
    """The control path stays torch-free, so the rewrite cannot reach it.

    Runs in a subprocess because pytest has already imported torch via other test modules.
    """
    code = f"import importlib, sys; importlib.import_module({module!r}); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
