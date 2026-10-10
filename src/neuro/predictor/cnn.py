from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuro.config import StftGeometry
from neuro.predictor.base import AutoregressiveModel, _apply_activation, _standardizer_arrays
from neuro.predictor.checkpoint import layer_arrays, layers_from_arrays, require_activation, require_model_type
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray


def _init_cnn_waveform_layers(  # noqa: PLR0913 -- architecture initializer needs all layer hyperparams
    key: jax.Array,
    *,
    n_channels: int,
    hidden_size: int,
    depth: int,
    kernel_size: int,
    n_u: int,
    n_controls: int,
    n_outputs: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Initialize causal 1D temporal convolution weights and dense head weights."""
    conv_w: list[jax.Array] = []
    conv_b: list[jax.Array] = []
    keys = jax.random.split(key, depth + 2)

    for i in range(depth):
        in_c = n_channels if i == 0 else hidden_size
        out_c = hidden_size
        kw, kb = jax.random.split(keys[i])
        bound = 1.0 / math.sqrt(in_c * kernel_size)
        w = jax.random.uniform(kw, (out_c, in_c, kernel_size), minval=-bound, maxval=bound, dtype=jnp.float32)
        b = jax.random.uniform(kb, (out_c,), minval=-bound, maxval=bound, dtype=jnp.float32)
        conv_w.append(w)
        conv_b.append(b)

    head_in = hidden_size + n_u * n_controls
    kw0, kb0 = jax.random.split(keys[depth])
    bound0 = 1.0 / math.sqrt(head_in)
    w0 = jax.random.uniform(kw0, (hidden_size, head_in), minval=-bound0, maxval=bound0, dtype=jnp.float32)
    b0 = jax.random.uniform(kb0, (hidden_size,), minval=-bound0, maxval=bound0, dtype=jnp.float32)

    kw1, kb1 = jax.random.split(keys[depth + 1])
    bound1 = 1.0 / math.sqrt(hidden_size)
    w1 = jax.random.uniform(kw1, (n_outputs, hidden_size), minval=-bound1, maxval=bound1, dtype=jnp.float32)
    b1 = jax.random.uniform(kb1, (n_outputs,), minval=-bound1, maxval=bound1, dtype=jnp.float32)

    return tuple(conv_w), tuple(conv_b), (w0, w1), (b0, b1)


def _init_cnn_observable_layers(  # noqa: PLR0913 -- architecture initializer needs all layer hyperparams
    key: jax.Array,
    *,
    n_channels: int,
    hidden_size: int,
    depth: int,
    kernel_size: int,
    frequency_kernel_size: int,
    n_values: int,
    n_u: int,
    n_controls: int,
    n_outputs: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Initialize causal 2D time-frequency convolution weights and dense head weights."""
    conv_w: list[jax.Array] = []
    conv_b: list[jax.Array] = []
    keys = jax.random.split(key, depth + 2)

    for i in range(depth):
        in_c = n_channels if i == 0 else hidden_size
        out_c = hidden_size
        kw, kb = jax.random.split(keys[i])
        bound = 1.0 / math.sqrt(in_c * kernel_size * frequency_kernel_size)
        w = jax.random.uniform(
            kw, (out_c, in_c, kernel_size, frequency_kernel_size), minval=-bound, maxval=bound, dtype=jnp.float32
        )
        b = jax.random.uniform(kb, (out_c,), minval=-bound, maxval=bound, dtype=jnp.float32)
        conv_w.append(w)
        conv_b.append(b)

    head_in = hidden_size * n_values + n_u * n_controls
    kw0, kb0 = jax.random.split(keys[depth])
    bound0 = 1.0 / math.sqrt(head_in)
    w0 = jax.random.uniform(kw0, (hidden_size, head_in), minval=-bound0, maxval=bound0, dtype=jnp.float32)
    b0 = jax.random.uniform(kb0, (hidden_size,), minval=-bound0, maxval=bound0, dtype=jnp.float32)

    kw1, kb1 = jax.random.split(keys[depth + 1])
    bound1 = 1.0 / math.sqrt(hidden_size)
    w1 = jax.random.uniform(kw1, (n_outputs, hidden_size), minval=-bound1, maxval=bound1, dtype=jnp.float32)
    b1 = jax.random.uniform(kb1, (n_outputs,), minval=-bound1, maxval=bound1, dtype=jnp.float32)

    return tuple(conv_w), tuple(conv_b), (w0, w1), (b0, b1)


class _CNNBase(AutoregressiveModel):
    """Shared CNN runtime state, dense head, and portable checkpoint handling."""

    conv_weights: tuple[jax.Array, ...]
    conv_biases: tuple[jax.Array, ...]
    head_weights: tuple[jax.Array, ...]
    head_biases: tuple[jax.Array, ...]
    n_values: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    frequency_kernel_size: int | None = eqx.field(static=True)
    geometry: StftGeometry | None = eqx.field(static=True)

    def __init__(  # noqa: PLR0913 -- explicit portable CNN architecture record
        self,
        *,
        conv_weights: tuple[FloatArray | jax.Array, ...],
        conv_biases: tuple[FloatArray | jax.Array, ...],
        head_weights: tuple[FloatArray | jax.Array, ...],
        head_biases: tuple[FloatArray | jax.Array, ...],
        kernel_size: int,
        frequency_kernel_size: int | None,
        geometry: StftGeometry | None,
        **core: Any,  # noqa: ANN401 -- forwarded to AutoregressiveModel
    ) -> None:
        """Initialize CNN checkpoint arrays and shared runtime state."""
        super().__init__(**core)
        if geometry is None:
            if self.n_outputs != self.n_channels:
                msg = "waveform CNN n_outputs must equal n_channels."
                raise ValueError(msg)
            self.n_values = 1
        else:
            if self.n_outputs < 1 or self.n_outputs % self.n_channels != 0:
                msg = "Observable CNN n_outputs must be a positive multiple of n_channels."
                raise ValueError(msg)
            self.n_values = self.n_outputs // self.n_channels
        self.conv_weights = tuple(jnp.asarray(x) for x in conv_weights)
        self.conv_biases = tuple(jnp.asarray(x) for x in conv_biases)
        self.head_weights = tuple(jnp.asarray(x) for x in head_weights)
        self.head_biases = tuple(jnp.asarray(x) for x in head_biases)
        self.kernel_size = int(kernel_size)
        self.frequency_kernel_size = None if frequency_kernel_size is None else int(frequency_kernel_size)
        self.geometry = geometry

    @classmethod
    def _checkpoint_kwargs(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> dict[str, Any]:
        """Unpack shared CNN metadata, layers, standardizers, and Observable geometry."""
        require_model_type(meta, "cnn")
        require_activation(meta)
        geometry = StftGeometry.model_validate(meta["geometry"]) if "geometry" in meta else None
        conv_w, conv_b = layers_from_arrays(arrays, "conv", int(meta["n_convs"]))
        head_w, head_b = layers_from_arrays(arrays, "head", int(meta["n_head_layers"]))
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
            "conv_weights": conv_w,
            "conv_biases": conv_b,
            "head_weights": head_w,
            "head_biases": head_b,
            "n_history": None,
            "kernel_size": int(meta["kernel_size"]),
            "frequency_kernel_size": (int(meta["frequency_kernel_size"]) if geometry is not None else None),
            "geometry": geometry,
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
        """Rebuild a CNN model from its portable checkpoint."""
        kwargs = cls._checkpoint_kwargs(meta, arrays)
        kwargs["n_history"] = n_history
        return cls(**kwargs)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the portable CNN checkpoint from runtime arrays."""
        meta = {
            "model_type": "cnn",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_outputs": self.n_outputs,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "residual": int(self.residual),
            "dt": self.dt,
            "downsample": self.downsample,
            "n_convs": len(self.conv_weights),
            "n_head_layers": len(self.head_weights),
            **self.provenance.meta,
        }
        if self.geometry is not None:
            meta["frequency_kernel_size"] = self.frequency_kernel_size
            meta["geometry"] = self.geometry.model_dump()
        arrays = layer_arrays("conv", self.conv_weights, self.conv_biases)
        arrays.update(layer_arrays("head", self.head_weights, self.head_biases))
        y_center = self.y_center.reshape(self.n_channels, self.n_values) if self.geometry is not None else self.y_center
        y_scale = self.y_scale.reshape(self.n_channels, self.n_values) if self.geometry is not None else self.y_scale
        arrays.update(_standardizer_arrays("y", y_center, y_scale))
        arrays.update(_standardizer_arrays("u", self.u_center, self.u_scale))
        return meta, arrays

    def with_history(self, n_history: int) -> Self:
        """Return an identical runtime with extended output history."""
        meta, arrays = self.to_checkpoint()
        return self.from_checkpoint(meta, arrays, n_history=n_history)


class WaveformCNNModel(_CNNBase):
    """Unified Equinox causal waveform CNN predictor."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        hidden_size: int,
        depth: int,
        kernel_size: int = 3,
        activation: Activation = "relu",
        residual: bool = True,
        dt: float = 0.0,
        downsample: int = 1,
        conv_weights: tuple[FloatArray | jax.Array, ...] | None = None,
        conv_biases: tuple[FloatArray | jax.Array, ...] | None = None,
        head_weights: tuple[FloatArray | jax.Array, ...] | None = None,
        head_biases: tuple[FloatArray | jax.Array, ...] | None = None,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
        y_center: FloatArray | jax.Array | None = None,
        y_scale: FloatArray | jax.Array | None = None,
        u_center: FloatArray | jax.Array | None = None,
        u_scale: FloatArray | jax.Array | None = None,
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
        key: jax.Array | None = None,
        n_outputs: int | None = None,
        frequency_kernel_size: int | None = None,
        geometry: StftGeometry | None = None,
    ) -> None:
        """Initialize waveform CNN weights and buffers."""
        n_out = n_channels if n_outputs is None else int(n_outputs)
        if y_std is not None:
            y_center, y_scale = y_std.center, y_std.scale
        if u_std is not None:
            u_center, u_scale = u_std.center, u_std.scale

        if conv_weights is None or conv_biases is None or head_weights is None or head_biases is None:
            prng = key if key is not None else jax.random.PRNGKey(0)
            conv_w, conv_b, head_w, head_b = _init_cnn_waveform_layers(
                prng,
                n_channels=n_channels,
                hidden_size=hidden_size,
                depth=depth,
                kernel_size=kernel_size,
                n_u=n_u,
                n_controls=n_controls,
                n_outputs=n_out,
            )
        else:
            conv_w, conv_b, head_w, head_b = conv_weights, conv_biases, head_weights, head_biases

        super().__init__(
            conv_weights=conv_w,
            conv_biases=conv_b,
            head_weights=head_w,
            head_biases=head_b,
            kernel_size=kernel_size,
            frequency_kernel_size=frequency_kernel_size,
            geometry=geometry,
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
            y_center=np.zeros(n_out) if y_center is None else y_center,
            y_scale=np.ones(n_out) if y_scale is None else y_scale,
            u_center=np.zeros(n_controls) if u_center is None else u_center,
            u_scale=np.ones(n_controls) if u_scale is None else u_scale,
            provenance=provenance,
            n_history=n_history,
        )

    @classmethod
    def from_checkpoint(
        cls, meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
    ) -> Self:
        """Rebuild a waveform CNN from a checkpoint."""
        if "geometry" in meta:
            msg = "waveform CNN cannot load a structured Observable checkpoint."
            raise ValueError(msg)
        return super().from_checkpoint(meta, arrays, n_history=n_history)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Apply causal temporal convolutions and dense head."""
        z = y_window
        for i, (weight, bias) in enumerate(zip(self.conv_weights, self.conv_biases, strict=True)):
            z = jnp.asarray(z, dtype=weight.dtype)
            z = (
                jax.lax.conv_general_dilated(
                    z.T[None, ...],
                    weight,
                    window_strides=(1,),
                    padding=((weight.shape[2] - 1, 0),),
                    dimension_numbers=("NCH", "OIH", "NCH"),
                )[0].T
                + bias[None, :]
            )
            if i < len(self.conv_weights) - 1:
                z = _apply_activation(self.activation, z)
        features = jnp.concatenate([z[-1], u_window.reshape(-1)])
        for i, (weight, bias) in enumerate(zip(self.head_weights, self.head_biases, strict=True)):
            features = features @ weight.T + bias
            if i < len(self.head_weights) - 1:
                features = _apply_activation(self.activation, features)
        return features + y_window[-1] if self.residual else features


class ObservableCNNModel(_CNNBase):
    """Unified Equinox causal time-frequency Observable CNN predictor."""

    geometry: StftGeometry = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        *,
        geometry: StftGeometry,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        hidden_size: int,
        depth: int,
        kernel_size: int = 3,
        frequency_kernel_size: int = 3,
        activation: Activation = "relu",
        residual: bool = True,
        dt: float = 0.0,
        downsample: int = 1,
        conv_weights: tuple[FloatArray | jax.Array, ...] | None = None,
        conv_biases: tuple[FloatArray | jax.Array, ...] | None = None,
        head_weights: tuple[FloatArray | jax.Array, ...] | None = None,
        head_biases: tuple[FloatArray | jax.Array, ...] | None = None,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
        y_center: FloatArray | jax.Array | None = None,
        y_scale: FloatArray | jax.Array | None = None,
        u_center: FloatArray | jax.Array | None = None,
        u_scale: FloatArray | jax.Array | None = None,
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
        key: jax.Array | None = None,
        n_outputs: int | None = None,
    ) -> None:
        """Initialize Observable CNN weights and buffers."""
        if n_outputs is not None:
            n_out = int(n_outputs)
            n_vals = n_out // n_channels
        else:
            fs = 1.0 / dt if dt > 0.0 else 50.0
            n_vals = geometry.n_values(fs)
            n_out = n_channels * n_vals
        if y_std is not None:
            y_center, y_scale = y_std.center, y_std.scale
        if u_std is not None:
            u_center, u_scale = u_std.center, u_std.scale

        if conv_weights is None or conv_biases is None or head_weights is None or head_biases is None:
            prng = key if key is not None else jax.random.PRNGKey(0)
            conv_w, conv_b, head_w, head_b = _init_cnn_observable_layers(
                prng,
                n_channels=n_channels,
                hidden_size=hidden_size,
                depth=depth,
                kernel_size=kernel_size,
                frequency_kernel_size=frequency_kernel_size,
                n_values=n_vals,
                n_u=n_u,
                n_controls=n_controls,
                n_outputs=n_out,
            )
        else:
            conv_w, conv_b, head_w, head_b = conv_weights, conv_biases, head_weights, head_biases

        super().__init__(
            conv_weights=conv_w,
            conv_biases=conv_b,
            head_weights=head_w,
            head_biases=head_b,
            kernel_size=kernel_size,
            frequency_kernel_size=frequency_kernel_size,
            geometry=geometry,
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
            y_center=np.zeros((n_channels, n_vals)) if y_center is None else y_center,
            y_scale=np.ones((n_channels, n_vals)) if y_scale is None else y_scale,
            u_center=np.zeros(n_controls) if u_center is None else u_center,
            u_scale=np.ones(n_controls) if u_scale is None else u_scale,
            provenance=provenance,
            n_history=n_history,
        )

    @classmethod
    def from_checkpoint(
        cls, meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
    ) -> Self:
        """Rebuild an Observable CNN runtime from a structured checkpoint."""
        if "geometry" not in meta:
            msg = "Observable CNN checkpoint requires recorded 'geometry'."
            raise ValueError(msg)
        return super().from_checkpoint(meta, arrays, n_history=n_history)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Apply causal time-frequency convolutions and preserve frequency positions."""
        z = y_window.reshape(self.n_y, self.n_channels, self.n_values).transpose(1, 0, 2)[None, ...]
        for i, (weight, bias) in enumerate(zip(self.conv_weights, self.conv_biases, strict=True)):
            z = (
                jax.lax.conv_general_dilated(
                    jnp.asarray(z, dtype=weight.dtype),
                    weight,
                    window_strides=(1, 1),
                    padding=((weight.shape[2] - 1, 0), ((weight.shape[3] - 1) // 2, weight.shape[3] // 2)),
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                + bias[None, :, None, None]
            )
            if i < len(self.conv_weights) - 1:
                z = _apply_activation(self.activation, z)
        features = jnp.concatenate([z[0, :, -1, :].reshape(-1), u_window.reshape(-1)])
        for i, (weight, bias) in enumerate(zip(self.head_weights, self.head_biases, strict=True)):
            features = features @ weight.T + bias
            if i < len(self.head_weights) - 1:
                features = _apply_activation(self.activation, features)
        prediction = features.reshape(self.n_channels, self.n_values)
        if self.residual:
            prediction = prediction + y_window[-1].reshape(self.n_channels, self.n_values)
        if y_window.ndim > 2:  # noqa: PLR2004 -- 2D check for observable frame representation
            return prediction
        return prediction.reshape(-1)

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run Observable histories and preserve their channel-frequency axes."""
        flat = super().free_run(y_hists, u_hists, u_futures)
        return flat.reshape(flat.shape[0], flat.shape[1], self.n_channels, self.n_values)
