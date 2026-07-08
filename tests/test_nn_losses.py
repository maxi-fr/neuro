"""Tests for the EEG-space loss: PCA decode wired into :func:`neuro.nn_training.compute_loss`.

These pin the two properties that justify computing the losses in channel space while the model
rolls out in latent space: (1) with an orthonormal basis the decoded-MSE gradient is just a
scalar multiple of the latent-MSE gradient (so moving MSE to EEG space does not change what the
model learns), and (2) the PSD/FC auxiliary losses now produce finite values/gradients under a
projection -- the combination that used to be forbidden.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_training import compute_loss  # noqa: E402
from neuro.prediction import AutoregressivePredictor  # noqa: E402

_SEED = 3


def _orthonormal_basis(rng: np.random.Generator, k: int, c: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((c, c)))
    return np.ascontiguousarray(q[:, :k].T)  # (k, c), orthonormal rows


def _model(n_y: int, n_u: int, horizon: int, k: int, n_controls: int) -> AutoregressivePredictor:
    mlp = eqx.nn.MLP(
        in_size=n_y * k + n_u * n_controls,
        out_size=k,
        width_size=8,
        depth=1,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
    return AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=k, n_controls=n_controls, activation="tanh"
    )


def _flat_grad(grads: eqx.Module) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    return np.concatenate([np.asarray(leaf).ravel() for leaf in leaves])


def test_decoded_mse_gradient_is_scalar_multiple_of_latent_gradient() -> None:
    """With an orthonormal basis, ``grad(channel MSE) == (k/C) * grad(latent MSE)``.

    Channel-space and latent-space MSE differ only by the mean's element count (``B*N*C`` vs
    ``B*N*k``); orthonormality makes the summed squared errors identical, so the gradients are
    parallel with ratio ``k/C``. This is the property that keeps EEG-space training equivalent
    to latent training for point accuracy.
    """
    rng = np.random.default_rng(_SEED)
    n_y, n_u, horizon, k, c, n_controls, batch = 2, 2, 1, 3, 6, 2, 8
    model = _model(n_y, n_u, horizon, k, n_controls)
    basis = _orthonormal_basis(rng, k, c)
    mean = rng.standard_normal(c)

    x = jnp.asarray(rng.standard_normal((batch, n_y * k + n_u * n_controls + horizon * n_controls)))
    z = rng.standard_normal((batch, horizon * k))  # latent target
    y_channel = jnp.asarray((z.reshape(batch, horizon, k) @ basis + mean).reshape(batch, horizon * c))
    basis_j, mean_j = jnp.asarray(basis), jnp.asarray(mean)

    step_mask = jnp.ones(horizon)
    grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    _, grad_channel = grad_fn(model, x, y_channel, step_mask, c, jnp.array(0.0), jnp.array(0.0), basis_j, mean_j)
    _, grad_latent = grad_fn(model, x, jnp.asarray(z), step_mask, k, jnp.array(0.0), jnp.array(0.0), None, None)

    np.testing.assert_allclose(_flat_grad(grad_channel), (k / c) * _flat_grad(grad_latent), rtol=1e-8, atol=1e-10)


def test_identity_decode_matches_no_decode() -> None:
    """An identity basis decode reproduces the no-projection losses exactly (plumbing check)."""
    rng = np.random.default_rng(_SEED + 1)
    n_y, n_u, horizon, c, n_controls, batch = 2, 2, 4, 5, 2, 16
    model = _model(n_y, n_u, horizon, c, n_controls)  # model predicts channels directly (k == C)

    x = jnp.asarray(rng.standard_normal((batch, n_y * c + n_u * n_controls + horizon * n_controls)))
    y = jnp.asarray(rng.standard_normal((batch, horizon * c)))

    identity = jnp.eye(c)
    zero = jnp.zeros(c)
    step_mask = jnp.ones(horizon)
    loss_dec, aux_dec = compute_loss(model, x, y, step_mask, c, jnp.array(0.1), jnp.array(0.1), identity, zero)
    loss_none, aux_none = compute_loss(model, x, y, step_mask, c, jnp.array(0.1), jnp.array(0.1), None, None)

    np.testing.assert_allclose(float(loss_dec), float(loss_none), rtol=1e-10, atol=1e-12)
    for key in ("mse", "psd", "fc"):
        np.testing.assert_allclose(float(aux_dec[key]), float(aux_none[key]), rtol=1e-10, atol=1e-12)


def test_projection_with_aux_losses_is_finite() -> None:
    """A projection combined with w_psd/w_fc > 0 (previously forbidden) yields finite loss/grads."""
    rng = np.random.default_rng(_SEED + 2)
    n_y, n_u, horizon, k, c, n_controls, batch = 2, 2, 6, 3, 8, 2, 32
    model = _model(n_y, n_u, horizon, k, n_controls)
    basis = _orthonormal_basis(rng, k, c)
    mean = rng.standard_normal(c)

    x = jnp.asarray(rng.standard_normal((batch, n_y * k + n_u * n_controls + horizon * n_controls)))
    y = jnp.asarray(rng.standard_normal((batch, horizon * c)))
    basis_j, mean_j = jnp.asarray(basis), jnp.asarray(mean)

    step_mask = jnp.ones(horizon)
    grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(model, x, y, step_mask, c, jnp.array(0.1), jnp.array(0.1), basis_j, mean_j)

    assert np.isfinite(float(loss))
    assert float(aux["psd"]) > 0.0  # a genuine channel-space spectrum, not the degenerate latent one
    assert float(aux["fc"]) > 0.0
    assert np.all(np.isfinite(_flat_grad(grads)))


def test_partial_mask_gates_psd_but_not_fc() -> None:
    """During the curriculum ramp (L < horizon) the PSD is gated off while the FC still grows with L."""
    rng = np.random.default_rng(_SEED + 3)
    n_y, n_u, horizon, c, n_controls, batch = 2, 2, 6, 5, 2, 32
    model = _model(n_y, n_u, horizon, c, n_controls)

    x = jnp.asarray(rng.standard_normal((batch, n_y * c + n_u * n_controls + horizon * n_controls)))
    y = jnp.asarray(rng.standard_normal((batch, horizon * c)))

    step_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # L = 3 < horizon
    grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(model, x, y, step_mask, c, jnp.array(0.1), jnp.array(0.1), None, None)

    assert float(aux["psd"]) == 0.0  # gated off until L == horizon
    assert float(aux["fc"]) > 0.0  # grows with L, scored over the first-L steps
    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(_flat_grad(grads)))
