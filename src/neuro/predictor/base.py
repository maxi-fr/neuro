from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.dynamics.base import DiscreteDynamics

from neuro.predictor.checkpoint import load_checkpoint, save_checkpoint
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


def _apply_activation(activation: Activation, z: jax.Array) -> jax.Array:
    """Apply the model activation elementwise."""
    if activation == "relu":
        return jnp.maximum(z, 0.0)
    if activation == "tanh":
        return jnp.tanh(z)
    return jnp.logaddexp(z, 0.0)


def _standardizer_arrays(prefix: str, center: jax.Array, scale: jax.Array) -> dict[str, FloatArray]:
    """Key one standardizer pair under the Standardizer convention both sides share."""
    return Standardizer(center=np.asarray(center, dtype=np.float64), scale=np.asarray(scale, dtype=np.float64)).arrays(
        prefix
    )


class InferencePredictor(ABC):
    """Runtime interface every deployed predictor implements on the JAX side."""

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
    n_history: int = 1

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
        """Free-run raw-in -> raw-out ``(B, steps, n_outputs)``, batched over independently primed histories."""
        ...

    @abstractmethod
    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the portable checkpoint pair ``(meta, arrays)``."""
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        meta: dict[str, Any],
        arrays: dict[str, FloatArray],
    ) -> Self:
        """Rebuild a model from its portable checkpoint."""
        ...

    @abstractmethod
    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs preceding the newest sample."""
        ...

    @abstractmethod
    def with_history(self, n_history: int) -> Self:
        """Return an identical runtime with extended output history."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizer buffers, and metadata into one ``.npz`` checkpoint."""
        meta, arrays = self.to_checkpoint()
        save_checkpoint(path, meta=meta, arrays=arrays)

    @classmethod
    def load(cls, path: str | Path, *, n_history: int | None = None) -> Self:
        """Rebuild a model from a saved checkpoint stem."""
        from neuro.predictor.inference import (  # noqa: PLC0415 -- deferred to prevent circular import with inference
            inference_from_checkpoint,
        )

        meta, arrays = load_checkpoint(path)
        if cls is InferencePredictor:
            return cast("Self", inference_from_checkpoint(meta, arrays, n_history=n_history))
        if issubclass(cls, AutoregressiveModel):
            return cast("Self", cls.from_checkpoint(meta, arrays, n_history=n_history))  # ty: ignore[unknown-argument]
        return cls.from_checkpoint(meta, arrays)


class AutoregressiveModel(DiscreteDynamics, InferencePredictor):
    """Unified Equinox autoregressive model supporting both training rollout and runtime MPC dynamics."""

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
        """Initialize shared autoregressive state and standardizer buffers."""
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
        self.y_center = jnp.asarray(y_c.reshape(-1), dtype=jnp.float64)
        self.y_scale = jnp.asarray(y_s.reshape(-1), dtype=jnp.float64)
        self.u_center = jnp.asarray(u_center, dtype=jnp.float64)
        self.u_scale = jnp.asarray(u_scale, dtype=jnp.float64)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401 -- dynamically assignable fields on frozen module
        """Allow assignment of recorded provenance and cached buffers."""
        if name in ("provenance", "downsample", "weights", "biases"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def with_provenance(self, provenance: TrainingProvenance, downsample: int) -> Self:
        """Return an updated model with recorded provenance and downsample factor."""
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "downsample", int(downsample))
        return self

    @property
    def y_std(self) -> Standardizer:
        """Output Standardizer reconstructed from buffers."""
        return Standardizer(
            center=np.asarray(self.y_center, dtype=np.float64), scale=np.asarray(self.y_scale, dtype=np.float64)
        )

    @property
    def u_std(self) -> Standardizer:
        """Control Standardizer reconstructed from buffers."""
        return Standardizer(
            center=np.asarray(self.u_center, dtype=np.float64), scale=np.asarray(self.u_scale, dtype=np.float64)
        )

    @abstractmethod
    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """Predict one standardized output from standardized history windows."""
        ...

    def rollout_standardized_one(
        self,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
    ) -> jax.Array:
        """Roll out one standardized sequence over ``horizon`` steps using scan."""
        weights = getattr(self, "weights", None) or getattr(self, "conv_weights", None)
        dtype = weights[0].dtype if weights is not None and len(weights) > 0 else y_hist.dtype
        y_window = jnp.asarray(y_hist[-self.n_y :], dtype=dtype)
        u_window = jnp.asarray(u_hist[-self.n_u :], dtype=dtype)
        u_future_arr = jnp.asarray(u_future, dtype=dtype)

        def step(
            carry: tuple[jax.Array, jax.Array], u_next: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            y_w, u_w = carry
            u_w = jnp.concatenate([u_w[1:], u_next[None]], axis=0)
            y_next = self._predict(y_w, u_w)
            y_w = jnp.concatenate([y_w[1:], y_next[None]], axis=0)
            return (y_w, u_w), y_next

        _, preds = jax.lax.scan(step, (y_window, u_window), u_future_arr)
        return preds

    def rollout_standardized(
        self,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
    ) -> jax.Array:
        """Batched standardized rollout over ``horizon`` steps: ``(B, ...) -> (B, horizon, ...)``."""
        y_h = jnp.asarray(y_hist)
        u_h = jnp.asarray(u_hist)
        u_f = jnp.asarray(u_future)
        return jax.vmap(self.rollout_standardized_one)(y_h, u_h, u_f)

    def rollout(
        self,
        *args: Any,  # noqa: ANN401 -- dynamic MPC dispatch accepts varying arguments
        **kwargs: Any,  # noqa: ANN401 -- dynamic MPC dispatch accepts varying arguments
    ) -> Any:  # noqa: ANN401 -- returns Trajectory for MPC or jax.Array for training
        """Forward simulate MPC trajectory or evaluate batched standardized training rollout."""
        if (len(args) >= 1 and hasattr(args[0], "X")) or "trajectory" in kwargs:
            return super().rollout(*args, **kwargs)
        return self.rollout_standardized(*args, **kwargs)

    def forward(
        self,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
    ) -> jax.Array:
        """Forward pass alias for batched standardized rollout."""
        return self.rollout_standardized(y_hist, u_hist, u_future)

    def __call__(
        self,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
    ) -> jax.Array:
        """Callable alias for batched standardized rollout."""
        return self.rollout_standardized(y_hist, u_hist, u_future)

    def _rollout_one(
        self,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
    ) -> jax.Array:
        """Standardize raw inputs, unroll over ``horizon`` steps, and inverse-transform to raw units."""
        y_shape = y_hist.shape
        y_flat = y_hist.reshape(y_shape[0], -1)
        y_std_flat = (y_flat - self.y_center) / self.y_scale
        y_std = y_std_flat.reshape(y_shape)
        u_std = (u_hist - self.u_center) / self.u_scale
        u_future_std = (u_future - self.u_center) / self.u_scale

        preds_std = self.rollout_standardized_one(y_std, u_std, u_future_std)
        pred_shape = preds_std.shape
        preds_flat = preds_std.reshape(pred_shape[0], -1)
        preds_raw_flat = preds_flat * self.y_scale + self.y_center
        return preds_raw_flat.reshape(pred_shape)

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run raw-in -> raw-out ``(B, steps, n_outputs)``, batched over independently primed histories."""
        return jax.vmap(self._rollout_one)(jnp.asarray(y_hists), jnp.asarray(u_hists), jnp.asarray(u_futures))

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
        """Advance one position: shift ``u`` into control window, predict, and shift both windows."""
        del t, dt
        n_z = self.n_history * self.n_outputs
        y_window = x[:n_z].reshape(self.n_history, self.n_outputs)
        u_window_raw = x[n_z:].reshape(self.n_u, self.n_controls)
        u_window = jnp.concatenate([u_window_raw[1:], u.reshape(1, -1)], axis=0)
        y_mlp_in = y_window[-self.n_y :]
        z_next = self._predict(y_mlp_in, (u_window - self.u_center) / self.u_scale)
        y_window = jnp.concatenate([y_window[1:], z_next[None, :]], axis=0)
        return jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

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
