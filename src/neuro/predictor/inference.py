from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.dynamics.base import DiscreteDynamics

from neuro.config import StftGeometry
from neuro.predictor.checkpoint import (
    layer_arrays,
    layers_from_arrays,
    load_checkpoint,
    require_activation,
    require_model_type,
    save_checkpoint,
)
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


class InferencePredictor(ABC):
    """Runtime-only interface every deployed predictor implements, on the jax side.

    The controller absorbs measurements into the model's opaque state (``absorb``), holds off
    until the state is primed (``is_ready``), seeds its MPC state from the unprimed state
    (``initial_state``) and recurses one position per call (``discrete_dynamics``) -- the
    priming-seam protocol the incumbent MPC used. ``free_run`` is the stateless free-run entry
    evaluation uses, and ``save``/``load``/``to_checkpoint``/``from_checkpoint`` round-trip the
    exchange checkpoint. Attributes such as channel/control/output counts, ``dt``, ``m``, ``n``
    and ``ne`` are part of the contract by documentation, not abstract enforcement.

    The free-run entry is ``free_run``, not the spec's ``rollout``: trajopt's ``AbstractModel``
    already defines ``rollout(trajectory, x0)`` and its iLQR solver calls it on these models, so
    the spec's name is taken by an incompatible contract.
    """

    n_y: int
    n_u: int
    n_channels: int
    n_controls: int
    n_outputs: int
    dt: float
    m: int
    n: int
    ne: int
    p: int

    @abstractmethod
    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output y = g(x, u, t) of shape ``(p,)``."""
        ...

    def has_control_feedthrough(self) -> bool:
        """Report whether the physical output depends on Control Current feedthrough ``u``."""
        return False

    @property
    def priming_steps(self) -> int:
        """Samples of raw history Priming needs before the state is ready: the wider of both windows."""
        return max(self.n_y, self.n_u)

    @abstractmethod
    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one position -- a sample or a Frame -- under control ``u`` -> ``x'``."""
        ...

    @abstractmethod
    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` into the model's opaque state."""
        ...

    @abstractmethod
    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the state has absorbed enough history to begin predicting."""
        ...

    @abstractmethod
    def initial_state(self) -> FloatArray:
        """Return the unprimed state."""
        ...

    @abstractmethod
    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run raw-in -> raw-out ``(B, positions, outputs)``, stateless in jax."""
        ...

    @abstractmethod
    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the exchange checkpoint is written from."""
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the model from a ``(meta, arrays)`` pair, in memory."""
        ...

    n_history: int = 1

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs of shape ``(count, n_outputs)`` preceding the newest sample."""
        raise NotImplementedError

    def with_history(self, n_history: int) -> Self:
        """Return a copy of the model with an extended history buffer."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Persist the exchange checkpoint to ``path`` (a suffix-less stem)."""
        meta, arrays = self.to_checkpoint()
        save_checkpoint(path, meta=meta, arrays=arrays)

    @classmethod
    def load(cls, path: str | Path, *, n_history: int | None = None) -> Self:
        """Rebuild the model from an exchange checkpoint on disk."""
        meta, arrays = load_checkpoint(path)
        if cls is InferencePredictor:
            return cast("Self", inference_from_checkpoint(meta, arrays, n_history=n_history))
        if issubclass(cls, ShiftRegisterModel):
            return cast("Self", cls.from_checkpoint(meta, arrays, n_history=n_history))
        return cls.from_checkpoint(meta, arrays)


def inference_from_checkpoint(
    meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
) -> InferencePredictor:
    """Construct the runtime adapter selected by checkpoint architecture metadata."""
    if meta.get("model_type") == "cnn":
        return WaveformCNNModel.from_checkpoint(meta, arrays, n_history=n_history)
    if "geometry" in meta:
        return ObservableMLPModel.from_checkpoint(meta, arrays, n_history=n_history)
    if meta.get("model_type") != "mlp":
        msg = f"unsupported neural checkpoint model_type: {meta.get('model_type')!r}"
        raise ValueError(msg)
    return WaveformMLPModel.from_checkpoint(meta, arrays, n_history=n_history)


def _apply_activation(activation: Activation, z: jax.Array) -> jax.Array:
    """Apply the model's activation elementwise."""
    if activation == "relu":
        return jnp.maximum(z, 0.0)
    if activation == "tanh":
        return jnp.tanh(z)
    return jnp.logaddexp(z, 0.0)


def _standardizer_arrays(prefix: str, center: jax.Array, scale: jax.Array) -> dict[str, FloatArray]:
    """Key one jax-side standardizer pair under the ``Standardizer`` convention both sides share."""
    return Standardizer(center=np.asarray(center, dtype=np.float64), scale=np.asarray(scale, dtype=np.float64)).arrays(
        prefix
    )


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


class ShiftRegisterModel(DiscreteDynamics, InferencePredictor):
    """Shared shift-register and standardizer state for waveform runtime adapters.

    The state is the flattened ``(n_history, n_outputs)`` standardized output window followed by
    the raw control window. Both MLP and CNN runtimes therefore share Priming, State Absorption,
    and Rollout while each supplies its own one-step feature map and checkpoint format. The
    ``discrete_dynamics`` shifts the newest control into the control window before predicting, so
    the predicted position depends on the Control Current applied at that step. ``free_run`` uses
    the training-aligned recursion and shifts the future control after each prediction.
    """

    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_history: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    downsample: int = eqx.field(static=True)
    activation: Activation = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    provenance: TrainingProvenance = eqx.field(static=True)
    y_center: jax.Array
    y_scale: jax.Array
    u_center: jax.Array
    u_scale: jax.Array

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls, meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
    ) -> Self:
        """Rebuild a shift-register runtime, optionally extending its history buffer."""
        ...

    @abstractmethod
    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Predict one standardized output from the current output and control windows."""
        ...

    def __init__(  # noqa: PLR0913 -- architecture record, standardizers and weights are the model
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
        activation: Activation,
        residual: bool,
        dt: float,
        downsample: int,
        y_center: FloatArray | jax.Array,
        y_scale: FloatArray | jax.Array,
        u_center: FloatArray | jax.Array,
        u_scale: FloatArray | jax.Array,
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
    ) -> None:
        """Copy checkpoint buffers into flat JAX solver arrays.

        Observable checkpoint standardizers may carry ``(n_channels, n_values)`` axes; they are
        flattened here because the controller state and solver interface remain one-dimensional.
        """
        n_out = int(n_outputs)
        y_c = np.asarray(y_center)
        y_s = np.asarray(y_scale)
        if y_c.size != n_out or y_s.size != n_out:
            msg = f"y_center/scale size ({y_c.size}) must equal model n_outputs ({n_out})."
            raise ValueError(msg)
        n_hist = int(n_y) if n_history is None else max(int(n_y), int(n_history))
        super().__init__(
            n=n_hist * n_out + int(n_u) * int(n_controls),
            m=int(n_controls),
            ne=n_hist * n_out + int(n_u) * int(n_controls),
            p=n_out,
        )
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.n_history = n_hist
        self.n_channels = int(n_channels)
        self.n_controls = int(n_controls)
        self.n_outputs = n_out
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.dt = float(dt)
        self.downsample = int(downsample)
        self.activation = activation
        self.residual = bool(residual)
        self.provenance = provenance if provenance is not None else TrainingProvenance()
        self.y_center = jnp.asarray(y_c.reshape(-1))
        self.y_scale = jnp.asarray(y_s.reshape(-1))
        self.u_center = jnp.asarray(u_center)
        self.u_scale = jnp.asarray(u_scale)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output y = g(x, u, t) of shape ``(n_outputs,)``."""
        del u, t
        newest = x[..., (self.n_history - 1) * self.n_outputs : self.n_history * self.n_outputs]
        return newest * self.y_scale + self.y_center

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs of shape ``(count, n_outputs)`` preceding the newest sample."""
        start_idx = (self.n_history - 1 - count) * self.n_outputs
        end_idx = (self.n_history - 1) * self.n_outputs
        past_std = x[..., start_idx:end_idx].reshape(-1, count, self.n_outputs)
        out = past_std * self.y_scale + self.y_center
        return out[0] if x.ndim == 1 else out

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one position: shift ``u`` into the control window, predict, and shift both windows."""
        del t, dt
        n_z = self.n_history * self.n_outputs
        y_window = x[:n_z].reshape(self.n_history, self.n_outputs)
        u_window_raw = x[n_z:].reshape(self.n_u, self.n_controls)
        u_window = jnp.concatenate([u_window_raw[1:], u.reshape(1, -1)], axis=0)
        y_mlp_in = y_window[-self.n_y :]
        z_next = self._predict(y_mlp_in, (u_window - self.u_center) / self.u_scale)
        y_window = jnp.concatenate([y_window[1:], z_next[None, :]], axis=0)
        return jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def _rollout_one(self, y_hist: jax.Array, u_hist: jax.Array, u_future: jax.Array) -> jax.Array:
        """Free-run one raw history under raw future controls using flat output state arrays."""
        y_window = (y_hist[-self.n_y :].reshape(self.n_y, self.n_outputs) - self.y_center) / self.y_scale
        u_window = (u_hist[-self.n_u :] - self.u_center) / self.u_scale
        u_future = (u_future - self.u_center) / self.u_scale

        def step(
            carry: tuple[jax.Array, jax.Array], u_next: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            y_window, u_window = carry
            y_next = self._predict(y_window, u_window)
            return (
                jnp.concatenate([y_window[1:], y_next[None]]),
                jnp.concatenate([u_window[1:], u_next[None]]),
            ), y_next

        (_, _), preds = jax.lax.scan(step, (y_window, u_window), u_future)
        return preds * self.y_scale + self.y_center

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run raw-in -> raw-out ``(B, steps, n_outputs)``, batched over independently primed histories."""
        return jax.vmap(self._rollout_one)(jnp.asarray(y_hists), jnp.asarray(u_hists), jnp.asarray(u_futures))

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` into the shift-register state."""
        n_z = self.n_history * self.n_outputs
        state_arr = np.asarray(state, dtype=np.float64)
        y_window = state_arr[:n_z].reshape(self.n_history, self.n_outputs)
        u_window = state_arr[n_z:].reshape(self.n_u, self.n_controls)
        z = (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        y_window = np.concatenate([y_window[1:], z[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        return np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the output window holds no NaN, i.e. at least ``n_history`` positions were absorbed."""
        n_z = self.n_history * self.n_outputs
        return not bool(np.isnan(np.asarray(state, dtype=np.float64)[:n_z]).any())

    def initial_state(self) -> FloatArray:
        """NaN-padded output window and zero-padded control window: nothing absorbed yet."""
        y_buf = np.full(self.n_history * self.n_outputs, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def with_history(self, n_history: int) -> Self:
        """Return a copy of the model with an extended history buffer."""
        meta, arrays = self.to_checkpoint()
        return self.from_checkpoint(meta, arrays, n_history=n_history)

class _MLPModel(ShiftRegisterModel):
    """Architecture-specific MLP prediction and checkpoint core."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __init__(  # noqa: PLR0913 -- explicit portable MLP architecture record
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
        activation: Activation,
        residual: bool,
        dt: float,
        downsample: int,
        y_center: FloatArray | jax.Array,
        y_scale: FloatArray | jax.Array,
        u_center: FloatArray | jax.Array,
        u_scale: FloatArray | jax.Array,
        weights: tuple[FloatArray | jax.Array, ...],
        biases: tuple[FloatArray | jax.Array, ...],
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
    ) -> None:
        """Copy MLP checkpoint weights and shared state buffers into JAX arrays."""
        super().__init__(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_outputs,
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
        self.weights = tuple(jnp.asarray(weight) for weight in weights)
        self.biases = tuple(jnp.asarray(bias) for bias in biases)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Apply the MLP to standardized output and control windows."""
        z = _mlp(
            self.activation,
            self.weights,
            self.biases,
            jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)]),
        )
        return z + y_window[-1] if self.residual else z

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
        """Rebuild an MLP runtime from a portable checkpoint."""
        kwargs = cls._checkpoint_kwargs(meta, arrays)
        if n_history is not None:
            kwargs["n_history"] = n_history
        return cls(**kwargs)


class WaveformCNNModel(ShiftRegisterModel):
    """JAX runtime for a causal waveform CNN checkpoint.

    The shift-register protocol is shared with the MLP adapter; only the one-step feature map
    differs, which keeps Priming, State Absorption, and Rollout semantics identical.
    """

    conv_weights: tuple[jax.Array, ...]
    conv_biases: tuple[jax.Array, ...]
    head_weights: tuple[jax.Array, ...]
    head_biases: tuple[jax.Array, ...]
    kernel_size: int = eqx.field(static=True)

    @classmethod
    def from_checkpoint(
        cls, meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
    ) -> Self:
        """Rebuild the CNN runtime from a portable checkpoint."""
        require_model_type(meta, "cnn")
        require_activation(meta)
        conv_w, conv_b = layers_from_arrays(arrays, "conv", int(meta["n_convs"]))
        head_w, head_b = layers_from_arrays(arrays, "head", int(meta["n_head_layers"]))
        y_std, u_std = Standardizer.from_arrays(arrays, "y"), Standardizer.from_arrays(arrays, "u")
        return cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            n_outputs=int(meta["n_outputs"]),
            hidden_size=int(meta["hidden_size"]),
            depth=int(meta["depth"]),
            activation=meta["activation"],
            residual=bool(meta["residual"]),
            dt=float(meta["dt"]),
            downsample=int(meta["downsample"]),
            y_center=y_std.center,
            y_scale=y_std.scale,
            u_center=u_std.center,
            u_scale=u_std.scale,
            conv_weights=conv_w,
            conv_biases=conv_b,
            head_weights=head_w,
            head_biases=head_b,
            n_history=n_history,
            kernel_size=int(meta["kernel_size"]),
            provenance=TrainingProvenance.from_meta(meta),
        )

    def __init__(
        self,
        *,
        conv_weights: tuple[FloatArray, ...],
        conv_biases: tuple[FloatArray, ...],
        head_weights: tuple[FloatArray, ...],
        head_biases: tuple[FloatArray, ...],
        kernel_size: int,
        **core: Any,  # noqa: ANN401 -- subclass forwards shared checkpoint fields
    ) -> None:
        """Copy CNN checkpoint buffers into JAX arrays and initialise the shared state protocol."""
        super().__init__(**core)
        self.conv_weights = tuple(jnp.asarray(x) for x in conv_weights)
        self.conv_biases = tuple(jnp.asarray(x) for x in conv_biases)
        self.head_weights = tuple(jnp.asarray(x) for x in head_weights)
        self.head_biases = tuple(jnp.asarray(x) for x in head_biases)
        self.kernel_size = int(kernel_size)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Apply causal temporal convolutions, then the dense Control Current head."""
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
        z = jnp.concatenate([z[-1], u_window.reshape(-1)])
        for i, (weight, bias) in enumerate(zip(self.head_weights, self.head_biases, strict=True)):
            z = z @ weight.T + bias
            if i < len(self.head_weights) - 1:
                z = _apply_activation(self.activation, z)
        return z + y_window[-1] if self.residual else z

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the portable CNN checkpoint from JAX arrays."""
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
        arrays = layer_arrays("conv", self.conv_weights, self.conv_biases)
        arrays.update(layer_arrays("head", self.head_weights, self.head_biases))
        arrays.update(_standardizer_arrays("y", self.y_center, self.y_scale))
        arrays.update(_standardizer_arrays("u", self.u_center, self.u_scale))
        return meta, arrays


class WaveformMLPModel(_MLPModel):
    """The waveform MLP runtime on the sample grid."""

    def __init__(  # noqa: PLR0913 -- explicit portable MLP architecture record
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
        activation: Activation,
        residual: bool,
        dt: float,
        downsample: int,
        y_center: FloatArray | jax.Array,
        y_scale: FloatArray | jax.Array,
        u_center: FloatArray | jax.Array,
        u_scale: FloatArray | jax.Array,
        weights: tuple[FloatArray | jax.Array, ...],
        biases: tuple[FloatArray | jax.Array, ...],
        provenance: TrainingProvenance | None = None,
        n_history: int | None = None,
    ) -> None:
        """Build a waveform MLP runtime from checkpoint arrays and metadata."""
        super().__init__(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_outputs,
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
            weights=weights,
            biases=biases,
            provenance=provenance,
            n_history=n_history,
        )


class ObservableMLPModel(_MLPModel):
    """The core on the Frame grid: one position is one Frame under the Control Current held over that hop.

    The Observable geometry the model was trained at rides along in the checkpoint, so config
    cross-validation can compare it against the Estimator that feeds the model at runtime.
    """

    geometry: StftGeometry = eqx.field(static=True)

    def __init__(self, *, geometry: StftGeometry, **core: Any) -> None:  # noqa: ANN401 -- forwarded verbatim to the core's own typed signature
        """Copy the checkpoint's float64 buffers into jax arrays and record the Observable geometry."""
        super().__init__(**core)
        self.geometry = geometry

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the torch side also writes and reads, geometry included."""
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
        """Add the recorded Observable geometry to the core's constructor arguments."""
        if "geometry" not in meta:
            msg = "ObservableMLPModel requires a checkpoint with recorded 'geometry'."
            raise ValueError(msg)
        return {
            **super()._checkpoint_kwargs(meta, arrays),
            "geometry": StftGeometry.model_validate(meta["geometry"]),
        }
