from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import numpy as np
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import _EpigraphModel

from neuro.predictor.inference import (
    InferencePredictor,
    inference_from_checkpoint,
)

if TYPE_CHECKING:
    import jax

    from neuro.types import FloatArray


class AugmentedActivationModel(DiscreteDynamics, InferencePredictor):
    """DiscreteDynamics adapter adding an activation variable as the last control coordinate."""

    base_model: InferencePredictor
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, base_model: InferencePredictor) -> None:
        """Wrap a base InferencePredictor adding an activation coordinate as the last control."""
        super().__init__(n=base_model.n, m=base_model.m + 1, ne=base_model.ne, p=base_model.p)
        self.base_model = base_model
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = int(base_model.n_controls)
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output, dropping the activation coordinate when feedthrough is present."""
        u_phys = u[: self.base_model.m] if u is not None and self.base_model.has_control_feedthrough() else None
        return self.base_model.output(x, u_phys, t)

    def has_control_feedthrough(self) -> bool:
        """Whether the base model's output function depends directly on Control Current."""
        return self.base_model.has_control_feedthrough()

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one step with physical controls u[:base_model.m]."""
        del t, dt
        return self.base_model.discrete_dynamics(x, u[: self.base_model.m], 0.0, self.dt)

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb measurement and physical control history."""
        return self.base_model.absorb(state, y, u[: self.base_model.m])

    def is_ready(self, state: FloatArray) -> bool:
        """Report readiness from base model."""
        return self.base_model.is_ready(state)

    @property
    def n_history(self) -> int:
        """History buffer length of the base model if defined, else n_y."""
        return getattr(self.base_model, "n_history", getattr(self.base_model, "n_y", 1))

    @property
    def y_scale(self) -> FloatArray | None:
        """Physical observation scale vector from base model."""
        return getattr(self.base_model, "y_scale", None)

    @property
    def y_center(self) -> FloatArray | None:
        """Physical observation centering vector from base model."""
        return getattr(self.base_model, "y_center", None)

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs through the base model."""
        return self.base_model.past_outputs(x, count)

    def with_history(self, n_history: int) -> AugmentedActivationModel:
        """Return a copy wrapping base_model with extended history."""
        return AugmentedActivationModel(self.base_model.with_history(n_history))

    def initial_state(self) -> FloatArray:
        """Return base model's initial state."""
        return self.base_model.initial_state()

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run with the physical controls u[..., :base_model.m]."""
        u_f = np.asarray(u_futures, dtype=np.float64)
        u_h = np.asarray(u_hists, dtype=np.float64)
        u_f_phys = u_f[..., : self.base_model.m]
        u_h_phys = u_h[..., : self.base_model.m] if u_h.shape[-1] == self.m else u_h
        return self.base_model.free_run(y_hists, u_h_phys, u_f_phys)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Return base model checkpoint."""
        return self.base_model.to_checkpoint()

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild base model then wrap."""
        base = inference_from_checkpoint(meta, arrays)
        return cls(base)


class EpigraphInferenceModel(DiscreteDynamics, InferencePredictor):
    """DiscreteDynamics and InferencePredictor adapter for _EpigraphModel, stripping trailing slack coordinates."""

    epigraph_model: _EpigraphModel
    base_model: InferencePredictor
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, epigraph_model: _EpigraphModel) -> None:
        """Wrap an _EpigraphModel, forwarding InferencePredictor methods to the underlying model."""
        super().__init__(n=epigraph_model.n, m=epigraph_model.m, ne=epigraph_model.ne, p=epigraph_model.p)
        self.epigraph_model = epigraph_model
        base_model = epigraph_model.original
        if not isinstance(base_model, InferencePredictor):
            msg = f"epigraph_model.original ({type(base_model).__name__}) must implement InferencePredictor"
            raise TypeError(msg)
        self.base_model = base_model
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = int(getattr(base_model, "n_controls", base_model.m))
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output, dropping epigraph slack coordinates."""
        return self.epigraph_model.output(x, u, t)

    def output_state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Return the output state Jacobian of shape (p, n)."""
        return self.epigraph_model.output_state_jacobian(x, u, t)

    def output_control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Append zero epigraph columns to the output control Jacobian."""
        return self.epigraph_model.output_control_jacobian(x, u, t)

    def has_control_feedthrough(self) -> bool:
        """Whether the model's output function depends directly on Control Current."""
        return self.epigraph_model.has_control_feedthrough()

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one step with physical controls u[:base_model.m]."""
        return self.epigraph_model.discrete_dynamics(x, u, t, dt)

    def state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Return the dynamics state Jacobian of shape (n, n)."""
        return self.epigraph_model.state_jacobian(x, u, t, *args)

    def control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Append zero epigraph columns to the control Jacobian."""
        return self.epigraph_model.control_jacobian(x, u, t, *args)

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Retain the original model's error-state coordinates."""
        return self.epigraph_model.state_diff(x, x0)

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Retain the original error-state Jacobian of shape (n, ne)."""
        return self.epigraph_model.errstate_jacobian(x)

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb measurement and physical control history, stripping epigraph slacks."""
        return self.base_model.absorb(state, y, u[: self.base_model.m])

    def is_ready(self, state: FloatArray) -> bool:
        """Report readiness from base model."""
        return self.base_model.is_ready(state)

    def initial_state(self) -> FloatArray:
        """Return base model's initial state."""
        return self.base_model.initial_state()

    def min_past_controls(self) -> int:
        """Minimum past control steps required by the underlying model or geometry."""
        fn = getattr(self.base_model, "min_past_controls", None)
        if callable(fn):
            return int(fn())
        geom = getattr(self.base_model, "geometry", None)
        if geom is not None:
            geom_fn = getattr(geom, "min_past_controls", None)
            if callable(geom_fn):
                return int(geom_fn())
        return getattr(self.base_model, "n_u", 1)

    @property
    def n_history(self) -> int:
        """History buffer length of the base model if defined, else n_y."""
        return getattr(self.base_model, "n_history", getattr(self.base_model, "n_y", 1))

    @property
    def y_scale(self) -> FloatArray | None:
        """Physical observation scale vector from base model."""
        return getattr(self.base_model, "y_scale", None)

    @property
    def y_center(self) -> FloatArray | None:
        """Physical observation centering vector from base model."""
        return getattr(self.base_model, "y_center", None)

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs through the base model."""
        return self.base_model.past_outputs(x, count)

    def with_history(self, n_history: int) -> EpigraphInferenceModel:
        """Return a copy wrapping base_model with extended history."""
        extended = cast("DiscreteDynamics", self.base_model.with_history(n_history))
        return EpigraphInferenceModel(_EpigraphModel(extended))

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run with the physical controls u[..., :base_model.m]."""
        u_f = np.asarray(u_futures, dtype=np.float64)
        u_h = np.asarray(u_hists, dtype=np.float64)
        u_f_phys = u_f[..., : self.base_model.m]
        u_h_phys = u_h[..., : self.base_model.m] if u_h.shape[-1] == self.m else u_h
        return self.base_model.free_run(y_hists, u_h_phys, u_f_phys)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Return base model checkpoint."""
        return self.base_model.to_checkpoint()

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild base model then wrap."""
        base = cast("DiscreteDynamics", inference_from_checkpoint(meta, arrays))
        return cls(_EpigraphModel(base))
