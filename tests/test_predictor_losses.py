"""Pin the torch losses: the Welch replica against SciPy and the PCA-decode gradient property."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import welch

from neuro.predictor.losses import predictor_loss, welch_psd
from neuro.predictor.module import AutoregressiveMLP

_SEED = 3


def _orthonormal_basis(rng: np.random.Generator, k: int, c: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((c, c)))
    return np.ascontiguousarray(q[:, :k].T)  # (k, c), orthonormal rows


def _model(
    n_y: int,
    n_u: int,
    horizon: int,
    k: int,
    n_controls: int,
    basis: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> AutoregressiveMLP:
    torch.manual_seed(_SEED)
    return AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=k,
        n_controls=n_controls,
        hidden_size=8,
        depth=1,
        activation="tanh",
        decode_basis=basis,
        decode_mean=mean,
    )


def _flat_grad(model: AutoregressiveMLP) -> np.ndarray:
    return np.concatenate([p.grad.reshape(-1).numpy() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("fs", [1.0, 250.0])
@pytest.mark.parametrize("nperseg", [8, 9])
def test_welch_psd_matches_scipy(nperseg: int, fs: float) -> None:
    """The replica reproduces ``scipy.signal.welch`` under the call the trainer makes.

    Both parities matter: the Nyquist bin is only left undoubled when ``nperseg`` is even.
    """
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal((4, 5 * nperseg))

    _, want = welch(x, fs=fs, nperseg=nperseg, noverlap=0, axis=-1)
    got = welch_psd(torch.as_tensor(x), nperseg, fs=fs).numpy()

    assert got.shape == (4, nperseg // 2 + 1)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0.0)


def test_decoded_mse_gradient_is_scalar_multiple_of_latent_gradient() -> None:
    """With an orthonormal basis, ``grad(channel MSE) == (k/C) * grad(latent MSE)``.

    Channel-space and latent-space MSE differ only by the mean's element count (``B*N*C`` vs
    ``B*N*k``); orthonormality makes the summed squared errors identical, so the gradients are
    parallel with ratio ``k/C``. This is what justifies rolling out in latent space while
    scoring in channel space.
    """
    rng = np.random.default_rng(_SEED)
    n_y, n_u, horizon, k, c, n_controls, batch = 2, 2, 1, 3, 6, 2, 8
    basis = _orthonormal_basis(rng, k, c)
    mean = rng.standard_normal(c)

    x = torch.as_tensor(rng.standard_normal((batch, n_y * k + n_u * n_controls + horizon * n_controls)))
    z = rng.standard_normal((batch, horizon, k))  # latent target
    step_mask = torch.ones(horizon, dtype=torch.float64)

    channel = _model(n_y, n_u, horizon, k, n_controls, basis, mean)
    y_channel = torch.as_tensor(z @ basis + mean)
    loss_channel, _ = predictor_loss(channel, channel(x).reshape(batch, horizon, k), y_channel, step_mask, 0.0)
    loss_channel.backward()

    latent = _model(n_y, n_u, horizon, k, n_controls)
    loss_latent, _ = predictor_loss(latent, latent(x).reshape(batch, horizon, k), torch.as_tensor(z), step_mask, 0.0)
    loss_latent.backward()

    np.testing.assert_allclose(_flat_grad(channel), (k / c) * _flat_grad(latent), rtol=1e-8, atol=1e-10)


def test_psd_is_gated_off_until_the_curriculum_reaches_full_length() -> None:
    """``step_mask[-1]`` gates the PSD term: zero while ``L < horizon``, live once ``L == horizon``."""
    rng = np.random.default_rng(_SEED + 1)
    n_y, n_u, horizon, c, n_controls, batch, w_psd = 2, 2, 6, 5, 2, 32, 0.1
    model = _model(n_y, n_u, horizon, c, n_controls)

    x = torch.as_tensor(rng.standard_normal((batch, n_y * c + n_u * n_controls + horizon * n_controls)))
    true_traj = torch.as_tensor(rng.standard_normal((batch, horizon, c)))
    pred_traj = model(x).reshape(batch, horizon, c)

    partial = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    total_partial, comps_partial = predictor_loss(model, pred_traj, true_traj, partial, w_psd)
    total_full, comps_full = predictor_loss(model, pred_traj, true_traj, torch.ones_like(partial), w_psd)

    assert comps_partial["psd"] == 0.0
    assert float(total_partial.detach()) == pytest.approx(comps_partial["mse"])
    assert comps_full["psd"] > 0.0
    assert float(total_full.detach()) == pytest.approx(comps_full["mse"] + w_psd * comps_full["psd"])


def test_single_step_horizon_skips_the_psd_term() -> None:
    """With ``horizon == 1`` there is no spectrum to score, so the PSD term is identically zero."""
    rng = np.random.default_rng(_SEED + 2)
    n_y, n_u, c, n_controls, batch = 2, 2, 4, 2, 8
    model = _model(n_y, n_u, 1, c, n_controls)

    x = torch.as_tensor(rng.standard_normal((batch, n_y * c + n_u * n_controls + n_controls)))
    true_traj = torch.as_tensor(rng.standard_normal((batch, 1, c)))

    _, comps = predictor_loss(model, model(x).reshape(batch, 1, c), true_traj, torch.ones(1, dtype=torch.float64), 1.0)

    assert comps["psd"] == 0.0
