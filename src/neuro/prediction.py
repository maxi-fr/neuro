from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", val=True)

if TYPE_CHECKING:
    from collections.abc import Callable


def get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Return the JAX activation function by name."""
    if name == "relu":
        return jax.nn.relu
    if name == "tanh":
        return jax.nn.tanh
    if name == "softplus":
        return jax.nn.softplus
    msg = f"Unsupported activation: {name}"
    raise ValueError(msg)


class AutoregressivePredictor(eqx.Module):
    """Wrapper that unrolls a 1-step MLP model over a prediction horizon.

    Attributes
    ----------
    model : eqx.nn.MLP
        The underlying 1-step MLP model.
    n_y : int
        The number of past EEG (output) steps the model needs as history.
    n_u : int
        The number of past control inputs the model needs as history.
    horizon : int
        The number of future steps to predict.
    n_channels : int
        The number of EEG (output) channels.
    n_controls : int
        The number of control (input) channels.
    """

    model: eqx.nn.MLP
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    activation: str = eqx.field(static=True, default="relu")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the autoregressive rollout."""
        y_past_flat = x[: self.n_y * self.n_channels]
        u_past_flat = x[self.n_y * self.n_channels : self.n_y * self.n_channels + self.n_u * self.n_controls]
        u_future_flat = x[self.n_y * self.n_channels + self.n_u * self.n_controls :]

        y_window = y_past_flat.reshape((self.n_y, self.n_channels))
        u_window = u_past_flat.reshape((self.n_u, self.n_controls))
        u_future = u_future_flat.reshape((self.horizon, self.n_controls))

        def scan_fn(
            carry: tuple[jax.Array, jax.Array], u_curr: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            y_w, u_w = carry
            new_u_w = jnp.concatenate([u_w[1:], u_curr[None, :]], axis=0)
            mlp_input = jnp.concatenate([y_w.flatten(), new_u_w.flatten()])
            y_next = self.model(mlp_input)
            new_y_w = jnp.concatenate([y_w[1:], y_next[None, :]], axis=0)
            return (new_y_w, new_u_w), y_next

        _, y_preds = jax.lax.scan(scan_fn, (y_window, u_window), u_future)
        return y_preds.flatten()
