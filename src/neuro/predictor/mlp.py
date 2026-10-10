from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuro.config import StftGeometry
from neuro.predictor.base import AutoregressiveModel, _apply_activation, _standardizer_arrays
from neuro.predictor.checkpoint import layer_arrays, layers_from_arrays, require_activation, require_model_type
from neuro.predictor.data import build_dataset_for_trajectory
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray


def _mlp(
    activation: Activation,
    weights: tuple[jax.Array, ...],
    biases: tuple[jax.Array, ...],
    z: jax.Array,
) -> jax.Array:
    """One MLP block forward pass; the activation follows every layer except the last."""
    for i, weight in enumerate(weights[:-1]):
        z = _apply_activation(activation, z @ weight.T + biases[i])
    return z @ weights[-1].T + biases[-1]


def _init_mlp_layers(
    key: jax.Array,
    sizes: list[int],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Initialize linear layers with Kaiming uniform weights."""
    weights: list[jax.Array] = []
    biases: list[jax.Array] = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, itertools.pairwise(sizes), strict=True):
        kw, kb = jax.random.split(k)
        bound = 1.0 / math.sqrt(n_in) if n_in > 0 else 1.0
        w = jax.random.uniform(kw, (n_out, n_in), minval=-bound, maxval=bound, dtype=jnp.float32)
        b = jax.random.uniform(kb, (n_out,), minval=-bound, maxval=bound, dtype=jnp.float32)
        weights.append(w)
        biases.append(b)
    return tuple(weights), tuple(biases)


class WaveformMLPModel(AutoregressiveModel):
    """Unified Equinox MLP predictor for waveform time series."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __init__(  # noqa: PLR0913 -- architecture record, standardizers, and weights define the model
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        n_outputs: int,
        hidden_size: int,
        depth: int,
        activation: Activation = "relu",
        residual: bool = True,
        dt: float = 0.0,
        downsample: int = 1,
        y_center: FloatArray | jax.Array | None = None,
        y_scale: FloatArray | jax.Array | None = None,
        u_center: FloatArray | jax.Array | None = None,
        u_scale: FloatArray | jax.Array | None = None,
        weights: tuple[FloatArray | jax.Array, ...] | None = None,
        biases: tuple[FloatArray | jax.Array, ...] | None = None,
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
        key: jax.Array | None = None,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
    ) -> None:
        """Initialize MLP weights and buffers."""
        n_out = int(n_outputs)
        if y_std is not None:
            y_center = y_std.center
            y_scale = y_std.scale
        if u_std is not None:
            u_center = u_std.center
            u_scale = u_std.scale

        if y_center is None:
            y_center = np.zeros(n_out, dtype=np.float64)
        if y_scale is None:
            y_scale = np.ones(n_out, dtype=np.float64)
        if u_center is None:
            u_center = np.zeros(n_controls, dtype=np.float64)
        if u_scale is None:
            u_scale = np.ones(n_controls, dtype=np.float64)

        super().__init__(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_out,
            hidden_size=hidden_size,
            depth=depth,
            activation=activation,
            residual=residual,
            dt=dt,
            downsample=downsample,
            y_center=y_center,
            y_scale=y_scale,
            u_center=u_center,
            u_scale=u_scale,
            provenance=provenance,
            n_history=n_history,
        )

        if weights is not None and biases is not None:
            self.weights = tuple(jnp.asarray(w, dtype=jnp.float32) for w in weights)
            self.biases = tuple(jnp.asarray(b, dtype=jnp.float32) for b in biases)
        else:
            prng = key if key is not None else jax.random.PRNGKey(0)
            sizes = [int(n_y) * n_out + int(n_u) * int(n_controls), *[int(hidden_size)] * int(depth), n_out]
            w_init, b_init = _init_mlp_layers(prng, sizes)
            self.weights = w_init
            self.biases = b_init

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Apply the MLP to standardized output and control windows."""
        features = jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        delta = _mlp(self.activation, self.weights, self.biases, features)
        if y_window.ndim > 2:  # noqa: PLR2004 -- matrix vs vector output shape
            delta = delta.reshape(y_window.shape[1:])
        return y_window[-1] + delta if self.residual else delta

    def with_readout(self, weight: FloatArray | jax.Array, bias: FloatArray | jax.Array) -> Self:
        """Return an updated model with new readout layer weights."""
        new_w = (*self.weights[:-1], jnp.asarray(weight, dtype=jnp.float32))
        new_b = (*self.biases[:-1], jnp.asarray(bias, dtype=jnp.float32))
        return eqx.tree_at(lambda m: (m.weights, m.biases), self, (new_w, new_b))

    def install_readout(self, A: FloatArray) -> None:
        """Write the closed-form-fitted readout A (c, f), bias column last, into the module."""
        w = np.asarray(A[:, :-1], dtype=np.float32)
        b = np.asarray(A[:, -1], dtype=np.float32)
        new_w = (*self.weights[:-1], jnp.asarray(w))
        new_b = (*self.biases[:-1], jnp.asarray(b))
        object.__setattr__(self, "weights", new_w)
        object.__setattr__(self, "biases", new_b)

    def design_normal_equations(
        self,
        trajectories: list[tuple[FloatArray, FloatArray]],
    ) -> tuple[FloatArray, FloatArray]:
        """Fold one-step input features and next-step targets into ``(G, P)``, bias column last."""
        c = self.n_outputs
        m = self.n_controls
        y_len = self.n_y * c
        f = y_len + self.n_u * m + 1
        G = np.zeros((f, f), dtype=np.float64)
        P = np.zeros((f, c), dtype=np.float64)

        for u_raw, y_raw in trajectories:
            y_arr = np.asarray(y_raw, dtype=np.float64)
            if y_arr.ndim > 2:  # noqa: PLR2004 -- 2D check for observable frame flattening
                y_arr = y_arr.reshape(len(y_arr), -1)
            u_std = self.u_std.transform(np.asarray(u_raw, dtype=np.float64))
            y_std = self.y_std.transform(y_arr)
            X, Y = build_dataset_for_trajectory(u_std, y_std, self.n_y, self.n_u, self.horizon)
            X_1step = np.hstack([X[:, :y_len], X[:, y_len + m : y_len + (self.n_u + 1) * m]])
            X_design = np.hstack([X_1step, np.ones((X_1step.shape[0], 1))])
            G += X_design.T @ X_design
            targets = Y[:, :c]
            if self.residual:
                targets = targets - X[:, y_len - c : y_len]
            P += X_design.T @ targets

        return G, P

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the portable MLP checkpoint from JAX arrays."""
        meta = {
            "model_type": "mlp",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_outputs": self.n_outputs,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "residual": int(self.residual),
            "dt": self.dt,
            "downsample": self.downsample,
            "n_layers": len(self.weights),
            **self.provenance.meta,
        }
        arrays = layer_arrays("layer", self.weights, self.biases)
        arrays.update(_standardizer_arrays("y", self.y_center, self.y_scale))
        arrays.update(_standardizer_arrays("u", self.u_center, self.u_scale))
        return meta, arrays

    @classmethod
    def _checkpoint_kwargs(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> dict[str, Any]:
        """Unpack MLP checkpoint metadata and arrays into constructor arguments."""
        require_model_type(meta, "mlp")
        require_activation(meta)
        weights, biases = layers_from_arrays(arrays, "layer", int(meta["n_layers"]))
        y_std, u_std = Standardizer.from_arrays(arrays, "y"), Standardizer.from_arrays(arrays, "u")
        return {
            "n_y": int(meta["n_y"]),
            "n_u": int(meta["n_u"]),
            "horizon": int(meta["horizon"]),
            "n_channels": int(meta["n_channels"]),
            "n_controls": int(meta["n_controls"]),
            "n_outputs": int(meta["n_outputs"]),
            "hidden_size": int(meta["hidden_size"]),
            "depth": int(meta["depth"]),
            "activation": meta["activation"],
            "residual": bool(meta["residual"]),
            "dt": float(meta["dt"]),
            "downsample": int(meta["downsample"]),
            "y_center": y_std.center,
            "y_scale": y_std.scale,
            "u_center": u_std.center,
            "u_scale": u_std.scale,
            "weights": weights,
            "biases": biases,
            "provenance": TrainingProvenance.from_meta(meta),
        }

    @classmethod
    def from_checkpoint(
        cls,
        meta: dict[str, Any],
        arrays: dict[str, FloatArray],
        *,
        n_history: int | None = None,
    ) -> Self:
        """Rebuild an MLP model from a portable checkpoint."""
        kwargs = cls._checkpoint_kwargs(meta, arrays)
        if n_history is not None:
            kwargs["n_history"] = n_history
        return cls(**kwargs)

    def with_history(self, n_history: int) -> Self:
        """Return an identical runtime with extended output history."""
        meta, arrays = self.to_checkpoint()
        return self.from_checkpoint(meta, arrays, n_history=n_history)


class ObservableMLPModel(WaveformMLPModel):
    """JAX runtime for an Observable MLP checkpoint."""

    geometry: StftGeometry = eqx.field(static=True)

    def __init__(self, *, geometry: StftGeometry, **core: Any) -> None:  # noqa: ANN401 -- forwarded kwargs to super
        """Attach the recorded Observable geometry to the MLP model."""
        super().__init__(**core)
        self.geometry = geometry

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the (meta, arrays) pair with geometry included."""
        meta, arrays = super().to_checkpoint()
        meta["geometry"] = self.geometry.model_dump()
        for name in ("y_center", "y_scale"):
            arrays[name] = arrays[name].reshape(self.n_channels, -1)
        return meta, arrays

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run Observable histories and preserve their channel-frequency axes."""
        flat = super().free_run(y_hists, u_hists, u_futures)
        return flat.reshape(flat.shape[0], flat.shape[1], self.n_channels, -1)

    @classmethod
    def _checkpoint_kwargs(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> dict[str, Any]:
        """Add the recorded Observable geometry to constructor arguments."""
        if "geometry" not in meta:
            msg = "ObservableMLPModel requires a checkpoint with recorded 'geometry'."
            raise ValueError(msg)
        return {
            **super()._checkpoint_kwargs(meta, arrays),
            "geometry": StftGeometry.model_validate(meta["geometry"]),
        }
