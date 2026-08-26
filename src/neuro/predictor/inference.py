from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.dynamics.base import DiscreteDynamics

from neuro.observable import control_means, geometry_from_meta, geometry_meta
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

    from neuro.config import ObservableGeometry
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
    def output(self, x: jax.Array) -> jax.Array:
        """Decode the raw output one state carries."""
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

    def save(self, path: str | Path) -> None:
        """Persist the exchange checkpoint to ``path`` (a suffix-less stem)."""
        meta, arrays = self.to_checkpoint()
        save_checkpoint(path, meta=meta, arrays=arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Rebuild the model from an exchange checkpoint on disk."""
        meta, arrays = load_checkpoint(path)
        return cls.from_checkpoint(meta, arrays)


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


class WaveformMLPModel(DiscreteDynamics, InferencePredictor):
    """trajopt ``DiscreteDynamics`` adapter for the waveform MLP predictor, on the sample grid.

    Holds the checkpoint's float64 weights and standardizer buffers as jax arrays, so the model
    rolls one MLP ``step`` per call with no torch in the loop. ``discrete_dynamics`` reproduces
    the incumbent CasADi ``NNSymbolicModel.step`` state machine exactly: the newest control
    enters the control window *before* the prediction, so the predicted sample depends on the
    control applied at that step. ``free_run`` is the training-aligned free run used by
    evaluation -- prime on raw history, then one prediction per future control with the control
    shifted in *after* -- so it reproduces the decoded torch ``forward`` rather than the MPC
    recursion.
    """

    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
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
    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __init__(  # noqa: PLR0913 -- architecture record, standardizers and weights are the model
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        hidden_size: int,
        depth: int,
        activation: Activation,
        residual: bool,
        dt: float,
        downsample: int,
        y_center: FloatArray,
        y_scale: FloatArray,
        u_center: FloatArray,
        u_scale: FloatArray,
        weights: tuple[FloatArray, ...],
        biases: tuple[FloatArray, ...],
        provenance: TrainingProvenance | None = None,
    ) -> None:
        """Copy the checkpoint's float64 buffers into jax arrays."""
        super().__init__(
            n=n_y * n_channels + n_u * n_controls,
            m=n_controls,
            ne=n_y * n_channels + n_u * n_controls,
        )
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.n_channels = int(n_channels)
        self.n_controls = int(n_controls)
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.dt = float(dt)
        self.downsample = int(downsample)
        self.activation = activation
        self.residual = bool(residual)
        self.provenance = provenance if provenance is not None else TrainingProvenance()
        self.y_center = jnp.asarray(y_center)
        self.y_scale = jnp.asarray(y_scale)
        self.u_center = jnp.asarray(u_center)
        self.u_scale = jnp.asarray(u_scale)
        self.weights = tuple(jnp.asarray(weight) for weight in weights)
        self.biases = tuple(jnp.asarray(bias) for bias in biases)

    @property
    def n_outputs(self) -> int:
        """Output width per position: one EEG sample across all channels."""
        return self.n_channels

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """One MLP forward pass on standardized windows -> the next standardized sample ``(n_channels,)``.

        With the residual skip the MLP output adds the window's last sample, so the layers fit
        the one-step delta exactly as the torch module does.
        """
        z = _mlp(
            self.activation,
            self.weights,
            self.biases,
            jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)]),
        )
        return z + y_window[-1] if self.residual else z

    def output(self, x: jax.Array) -> jax.Array:
        """Decode the newest standardized y-window row into the raw predicted sample ``(n_channels,)``.

        The prediction the state carries is always its newest row, so the raw sample is a state
        component -- the decode the spectral hinge reads off any sample-grid model.
        """
        n_z = self.n_y * self.n_channels
        z_last = x[n_z - self.n_channels : n_z]
        return z_last * self.y_scale + self.y_center

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one sample: shift ``u`` into the control window, predict, and shift both windows.

        Mirrors the incumbent MPC's ``NNSymbolicModel.f_step``: the predicted ``y_{t+1}`` is the
        MLP output on the y-window ending at ``t`` and the u-window ending at ``t + 1`` after
        ``u`` is shifted in, and the returned state's control window ends with ``u``.
        """
        del t, dt
        n_z = self.n_y * self.n_channels
        y_window = x[:n_z].reshape(self.n_y, self.n_channels)
        u_window_raw = x[n_z:].reshape(self.n_u, self.n_controls)
        u_window = jnp.concatenate([u_window_raw[1:], u.reshape(1, -1)], axis=0)
        z_next = self._predict(y_window, (u_window - self.u_center) / self.u_scale)
        y_window = jnp.concatenate([y_window[1:], z_next[None, :]], axis=0)
        return jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def _rollout_one(self, y_hist: jax.Array, u_hist: jax.Array, u_future: jax.Array) -> jax.Array:
        """Free-run one raw history under raw future controls -> raw ``(steps, n_channels)``."""
        y_window = (y_hist[-self.n_y :] - self.y_center) / self.y_scale
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
        """Free-run raw-in -> raw-out ``(B, steps, n_channels)``, batched over independently primed histories."""
        return jax.vmap(self._rollout_one)(jnp.asarray(y_hists), jnp.asarray(u_hists), jnp.asarray(u_futures))

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` into the shift-register state."""
        n_z = self.n_y * self.n_channels
        state_arr = np.asarray(state, dtype=np.float64)
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z:].reshape(self.n_u, self.n_controls)
        z = (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        y_window = np.concatenate([y_window[1:], z[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        return np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        n_z = self.n_y * self.n_channels
        return not bool(np.isnan(np.asarray(state, dtype=np.float64)[:n_z]).any())

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window and zero-padded control window: nothing absorbed yet."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the torch side also writes and reads."""
        meta = {
            "model_type": "mlp",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
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
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the model from a ``(meta, arrays)`` pair, in memory."""
        require_model_type(meta, "mlp")
        require_activation(meta)
        weights, biases = layers_from_arrays(arrays, "layer", int(meta["n_layers"]))
        y_std, u_std = Standardizer.from_arrays(arrays, "y"), Standardizer.from_arrays(arrays, "u")
        return cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            hidden_size=int(meta["hidden_size"]),
            depth=int(meta["depth"]),
            activation=meta["activation"],
            residual=bool(meta.get("residual", False)),
            dt=float(meta["dt"]),
            downsample=int(meta["downsample"]),
            y_center=y_std.center,
            y_scale=y_std.scale,
            u_center=u_std.center,
            u_scale=u_std.scale,
            weights=weights,
            biases=biases,
            provenance=TrainingProvenance.from_meta(meta),
        )


class ObservableModel(DiscreteDynamics, InferencePredictor):
    """trajopt ``DiscreteDynamics`` adapter for the one-Frame-per-step Observable predictor.

    The opaque state is the history register -- ``[standardized EEG window | raw control
    window]`` -- followed by the lifted Frame state (and the residual carry block, when the
    checkpoint's skip is on). One step is one Frame: ``discrete_dynamics`` runs the shared
    transition under the Frame-mean control ``u`` and carries the register unchanged; the lift
    (register -> Frame state) happens once at ``absorb``/``initial_state``. ``free_run`` is the
    training-aligned free run: prime, lift once, then chain ``discrete_dynamics``/``output`` per
    Frame, so the free run and the recursion the MPC steps are the same code.
    """

    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    n_values: int = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    downsample: int = eqx.field(static=True)
    activation: Activation = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    geometry: ObservableGeometry = eqx.field(static=True)
    provenance: TrainingProvenance = eqx.field(static=True)
    lift_weights: tuple[jax.Array, ...]
    lift_biases: tuple[jax.Array, ...]
    transition_weights: tuple[jax.Array, ...]
    transition_biases: tuple[jax.Array, ...]
    readout_w: jax.Array
    readout_b: jax.Array
    y_center: jax.Array
    y_scale: jax.Array
    u_center: jax.Array
    u_scale: jax.Array
    l_center: jax.Array
    l_scale: jax.Array

    def __init__(  # noqa: PLR0913 -- architecture record, geometry, standardizers and weights are the model
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        z_dim: int,
        n_values: int,
        fs: float,
        downsample: int,
        activation: Activation,
        residual: bool,
        geometry: ObservableGeometry,
        lift_weights: tuple[FloatArray, ...],
        lift_biases: tuple[FloatArray, ...],
        transition_weights: tuple[FloatArray, ...],
        transition_biases: tuple[FloatArray, ...],
        readout_w: FloatArray,
        readout_b: FloatArray,
        y_center: FloatArray,
        y_scale: FloatArray,
        u_center: FloatArray,
        u_scale: FloatArray,
        l_center: FloatArray,
        l_scale: FloatArray,
        provenance: TrainingProvenance | None = None,
    ) -> None:
        """Copy the checkpoint's float64 buffers into jax arrays."""
        n_hist = n_y * n_channels + n_u * n_controls
        carry_size = n_channels * n_values if residual else 0
        super().__init__(n=n_hist + z_dim + carry_size, m=n_controls, ne=n_hist + z_dim + carry_size)
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.n_channels = int(n_channels)
        self.n_controls = int(n_controls)
        self.horizon = int(horizon)
        self.z_dim = int(z_dim)
        self.n_values = int(n_values)
        self.fs = float(fs)
        self.dt = 1.0 / float(fs)
        self.downsample = int(downsample)
        self.activation = activation
        self.residual = bool(residual)
        self.geometry = geometry
        self.provenance = provenance if provenance is not None else TrainingProvenance()
        self.lift_weights = tuple(jnp.asarray(weight) for weight in lift_weights)
        self.lift_biases = tuple(jnp.asarray(bias) for bias in lift_biases)
        self.transition_weights = tuple(jnp.asarray(weight) for weight in transition_weights)
        self.transition_biases = tuple(jnp.asarray(bias) for bias in transition_biases)
        self.readout_w = jnp.asarray(readout_w)
        self.readout_b = jnp.asarray(readout_b)
        self.y_center = jnp.asarray(y_center)
        self.y_scale = jnp.asarray(y_scale)
        self.u_center = jnp.asarray(u_center)
        self.u_scale = jnp.asarray(u_scale)
        self.l_center = jnp.asarray(l_center)
        self.l_scale = jnp.asarray(l_scale)

    @property
    def n_outputs(self) -> int:
        """Output width per position: one raw log-Observable Frame across channels and values."""
        return self.n_channels * self.n_values

    @property
    def _n_hist(self) -> int:
        """Register width: the standardized EEG window plus the raw control window."""
        return self.n_y * self.n_channels + self.n_u * self.n_controls

    def n_frames(self, horizon: int) -> int:
        """Frames the recursion emits over ``horizon`` samples at the checkpoint's geometry."""
        return self.geometry.n_frames(horizon, self.fs)

    def _lift(self, register: jax.Array) -> jax.Array:
        """Lift a history register to the Frame state via the checkpoint's lift block."""
        n_z = self.n_y * self.n_channels
        u_window = register[n_z:].reshape(self.n_u, self.n_controls)
        lift_in = jnp.concatenate([register[:n_z], ((u_window - self.u_center) / self.u_scale).reshape(-1)])
        return _mlp(self.activation, self.lift_weights, self.lift_biases, lift_in)

    def output(self, x: jax.Array) -> jax.Array:
        """Decode the Frame level into the raw log-Observable frame ``(n_channels * n_values,)``.

        With the residual skip the level is the accumulated carry block; otherwise it is the
        readout of the lifted Frame state.
        """
        if self.residual:
            l_std = x[self._n_hist + self.z_dim :]
        else:
            z = x[self._n_hist :]
            l_std = z @ self.readout_w.T + self.readout_b
        return l_std * self.l_scale + self.l_center

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one Frame: run the shared transition on the carried lifted state under Frame-mean ``u``.

        The register is carried unchanged -- only the lifted Frame state advances.
        """
        del t, dt
        z = x[self._n_hist : self._n_hist + self.z_dim]
        v = (u - self.u_center) / self.u_scale
        z_next = _mlp(self.activation, self.transition_weights, self.transition_biases, jnp.concatenate([z, v]))
        if self.residual:
            carry = x[self._n_hist + self.z_dim :] + (z_next @ self.readout_w.T + self.readout_b)
            return jnp.concatenate([x[: self._n_hist], z_next, carry])
        return jnp.concatenate([x[: self._n_hist], z_next])

    def _rollout_one(
        self, y_hist: jax.Array, u_hist: jax.Array, u_future: jax.Array, aggregate: jax.Array
    ) -> jax.Array:
        """Free-run one raw history under raw future controls -> raw ``(n_frames, n_outputs)``.

        Primes the register, lifts once, then chains :meth:`discrete_dynamics`/:meth:`output` --
        the recursion the MPC itself steps -- one call per Frame, so the Frame transition has a
        single owner. ``aggregate`` averages the raw Control Currents over each Frame's support,
        which is the ``u`` the transition takes.
        """
        register = jnp.concatenate(
            [
                ((y_hist[-self.n_y :] - self.y_center) / self.y_scale).reshape(-1),
                u_hist[-self.n_u :].reshape(-1),
            ]
        )
        state = jnp.concatenate([register, self._lift(register)])
        if self.residual:
            state = jnp.concatenate([state, jnp.zeros(self.n_outputs)])

        def step(state: jax.Array, u_bar_m: jax.Array) -> tuple[jax.Array, jax.Array]:
            state = self.discrete_dynamics(state, u_bar_m, 0.0, self.dt)
            return state, self.output(state)

        _, preds = jax.lax.scan(step, state, aggregate @ u_future)
        return preds

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run raw-in -> raw-out ``(B, n_frames, n_outputs)``, batched over independently primed histories."""
        u_futures_arr = jnp.asarray(u_futures)
        aggregate = jnp.asarray(control_means(self.geometry, u_futures_arr.shape[1], self.fs))
        return jax.vmap(self._rollout_one, in_axes=(0, 0, 0, None))(
            jnp.asarray(y_hists), jnp.asarray(u_hists), u_futures_arr, aggregate
        )

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` to the register, re-lifting."""
        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z : self._n_hist].reshape(self.n_u, self.n_controls)
        z_new = (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        y_window = np.concatenate([y_window[1:], z_new[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        register = np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        z = np.asarray(self._lift(jnp.asarray(register, dtype=jnp.float64)), dtype=np.float64)
        state = np.concatenate([register, z])
        if self.residual:
            state = np.concatenate([state, np.zeros(self.n_outputs, dtype=np.float64)])
        return state

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        return not bool(np.isnan(np.asarray(state, dtype=np.float64)[: self.n_y * self.n_channels]).any())

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window, zero-padded control window, their (NaN) lifted Frame state, zero carry."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        register = np.concatenate([y_buf, u_buf])
        state = np.concatenate([register, self._lift(jnp.asarray(register, dtype=jnp.float64))])
        if self.residual:
            state = np.concatenate([state, np.zeros(self.n_outputs, dtype=np.float64)])
        return state

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the torch side also writes and reads."""
        meta = {
            "model_type": "observable",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "dt": self.dt,
            "downsample": self.downsample,
            "z_dim": self.z_dim,
            "lift_hidden": self.lift_weights[0].shape[0] if len(self.lift_weights) > 1 else 0,
            "lift_depth": len(self.lift_weights) - 1,
            "transition_hidden": self.transition_weights[0].shape[0] if len(self.transition_weights) > 1 else 0,
            "transition_depth": len(self.transition_weights) - 1,
            "n_lift_layers": len(self.lift_weights),
            "n_transition_layers": len(self.transition_weights),
            "residual": int(self.residual),
            "geometry": geometry_meta(self.geometry),
            **self.provenance.meta,
        }
        arrays = layer_arrays("lift", self.lift_weights, self.lift_biases)
        arrays.update(layer_arrays("transition", self.transition_weights, self.transition_biases))
        arrays["readout.weight"] = np.asarray(self.readout_w, dtype=np.float64)
        arrays["readout.bias"] = np.asarray(self.readout_b, dtype=np.float64)
        for prefix, center, scale in (
            ("y", self.y_center, self.y_scale),
            ("u", self.u_center, self.u_scale),
            ("l", self.l_center, self.l_scale),
        ):
            arrays.update(_standardizer_arrays(prefix, center, scale))
        return meta, arrays

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the model from a ``(meta, arrays)`` pair, in memory."""
        require_model_type(meta, "observable")
        require_activation(meta)
        lift_weights, lift_biases = layers_from_arrays(arrays, "lift", int(meta["n_lift_layers"]))
        transition_weights, transition_biases = layers_from_arrays(
            arrays, "transition", int(meta["n_transition_layers"])
        )
        y_std = Standardizer.from_arrays(arrays, "y")
        u_std = Standardizer.from_arrays(arrays, "u")
        l_std = Standardizer.from_arrays(arrays, "l")
        fs = 1.0 / float(meta["dt"])
        geometry = geometry_from_meta(meta["geometry"])
        return cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            z_dim=int(meta["z_dim"]),
            n_values=geometry.n_values(fs),
            fs=fs,
            downsample=int(meta["downsample"]),
            activation=meta["activation"],
            residual=bool(meta.get("residual", False)),
            geometry=geometry,
            lift_weights=lift_weights,
            lift_biases=lift_biases,
            transition_weights=transition_weights,
            transition_biases=transition_biases,
            readout_w=np.asarray(arrays["readout.weight"], dtype=np.float64),
            readout_b=np.asarray(arrays["readout.bias"], dtype=np.float64),
            y_center=y_std.center,
            y_scale=y_std.scale,
            u_center=u_std.center,
            u_scale=u_std.scale,
            l_center=l_std.center,
            l_scale=l_std.scale,
            provenance=TrainingProvenance.from_meta(meta),
        )
