from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuro.transforms import Pipeline

jax.config.update("jax_enable_x64", val=True)

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.types import FloatArray


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


@dataclass(frozen=True)
class MLPArtifact:
    """Loaded ``run_nn_predictor`` artifact: the live predictor, native dt, and transforms.

    Attributes
    ----------
    model : AutoregressivePredictor
        The trained (or freshly built) predictor.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    y_pipeline : Pipeline
        Raw EEG -> model space map: a channel-space :class:`~neuro.transforms.Standardizer`
        followed by an optional :class:`~neuro.transforms.PCAProjection`. Encodes measured EEG
        the model consumes and decodes its predictions back to raw EEG channels.
    u_pipeline : Pipeline
        Raw control -> model space map (a single standardizer).
    """

    model: AutoregressivePredictor
    dt: float
    downsample: int
    y_pipeline: Pipeline
    u_pipeline: Pipeline

    @property
    def n_y(self) -> int:
        """Number of past EEG (output) steps in the model's history window."""
        return self.model.n_y

    @property
    def n_u(self) -> int:
        """Number of past control (input) steps in the model's history window."""
        return self.model.n_u

    @property
    def horizon(self) -> int:
        """Direct-prediction horizon of the underlying model."""
        return self.model.horizon

    @property
    def n_channels(self) -> int:
        """Number of EEG output channels."""
        return self.model.n_channels

    @property
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.model.n_controls

    @property
    def n_eeg_channels(self) -> int:
        """Number of raw EEG channels (the PCA basis' input dimension, else ``n_channels``)."""
        pca = self.y_pipeline.pca
        return pca.basis.shape[1] if pca is not None else self.n_channels

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_eeg_channels)`` into model space (standardize, then project)."""
        return self.y_pipeline.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_eeg_channels)`` from model-space values."""
        return self.y_pipeline.inverse_transform(np.asarray(z, dtype=np.float64))

    @classmethod
    def load(cls, artifact: str | Path) -> MLPArtifact:
        """Load the 3-file artifact (``.eqx``/``.json``/``.scalers.npz``) from disk."""
        artifact = Path(artifact)
        meta: dict[str, Any] = json.loads(artifact.with_suffix(".json").read_text())
        with np.load(artifact.with_suffix(".scalers.npz")) as sc:
            arrays = {k: np.asarray(sc[k], dtype=np.float64) for k in sc.files}

        y_pipeline = Pipeline.from_serialized("y", meta["y_pipeline"], arrays)
        u_pipeline = Pipeline.from_serialized("u", meta["u_pipeline"], arrays)

        activation_name = meta.get("activation", "relu")
        mlp = eqx.nn.MLP(
            in_size=meta["in_size"],
            out_size=meta["out_size"],
            width_size=meta["hidden_size"],
            depth=meta["depth"],
            activation=get_activation(activation_name),
            key=jax.random.PRNGKey(0),
        )
        skeleton = AutoregressivePredictor(
            model=mlp,
            n_y=meta["n_y"],
            n_u=meta["n_u"],
            horizon=meta["horizon"],
            n_channels=meta["n_channels"],
            n_controls=meta["n_controls"],
            activation=activation_name,
        )
        model = eqx.tree_deserialise_leaves(str(artifact), skeleton)

        return cls(
            model=model,
            dt=float(meta["dt"]),
            downsample=int(meta["downsample"]),
            y_pipeline=y_pipeline,
            u_pipeline=u_pipeline,
        )

    def save(self, artifact: str | Path) -> None:
        """Persist the predictor (eqx leaves) plus a JSON sidecar and the transform arrays."""
        artifact = Path(artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(artifact), self.model)
        mlp = self.model.model
        meta = {
            "in_size": int(mlp.in_size),
            "out_size": int(mlp.out_size),
            "hidden_size": int(mlp.width_size),
            "depth": int(mlp.depth),
            "activation": self.model.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "downsample": self.downsample,
            "dt": self.dt,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_eeg_channels": self.n_eeg_channels,
            "y_pipeline": self.y_pipeline.step_tags(),
            "u_pipeline": self.u_pipeline.step_tags(),
        }
        artifact.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        transform_arrays = {**self.y_pipeline.array_dict("y"), **self.u_pipeline.array_dict("u")}
        np.savez(artifact.with_suffix(".scalers.npz"), **transform_arrays)  # ty: ignore[invalid-argument-type]
