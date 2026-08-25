from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    import casadi as ca

    from neuro.config import ObservableGeometry

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
StrArray = npt.NDArray[np.str_]


class SymbolicModel(Protocol):
    """Protocol shared by CasADi symbolic models (NNSymbolicModel, ESNSymbolicModel)."""

    @property
    def state_shape(self) -> tuple[int, int]:
        """State dimension shape."""
        ...

    @property
    def n_controls(self) -> int:
        """Control input count."""
        ...

    @property
    def n_channels(self) -> int:
        """Output channel count."""
        ...

    @property
    def native_horizon(self) -> int:
        """Native prediction horizon of the underlying artifact."""
        ...

    @property
    def is_linear(self) -> bool:
        """Whether the model represents a linear dynamic system."""
        ...

    @property
    def f_step(self) -> ca.Function:
        """Symbolic state step function."""
        ...

    @property
    def f_out(self) -> ca.Function:
        """Symbolic output function."""
        ...

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance symbolic state by one step under control input u."""
        ...

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Decode symbolic state to raw EEG output."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed or zero initial state vector."""
        ...

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new measurement y and previously applied control u into state."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if state has absorbed sufficient history to begin control."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """Runtime-only, raw-units interface every surrogate predictor implements.

    Predicts the next output(s) from an opaque internal state. State is the module's own business
    -- a shift register for the waveform MLP, a reservoir vector for the ESN -- and callers never
    inspect it. Every boundary value is in raw units: ``prime`` takes raw history, ``rollout`` and
    ``step`` return raw predictions, ``absorb`` takes raw measurements; the standardizers live
    inside the implementation, invisible through this interface.

    The members are runtime-only: ``__init__``, persistence and geometry are implementation
    specific. ``step`` advances one position, which is one sample for the waveform predictors and
    one Frame for the observable predictor. ``horizon`` is the native/trained horizon and an
    identity, not a hard bound on the length of ``u_future`` accepted by ``rollout``.
    """

    @property
    def n_channels(self) -> int:
        """Physical EEG channel count."""
        ...

    @property
    def n_controls(self) -> int:
        """Control input channel count."""
        ...

    @property
    def n_outputs(self) -> int:
        """Output width per position."""
        ...

    @property
    def dt(self) -> float:
        """The model's native time step, seconds."""
        ...

    @property
    def priming_steps(self) -> int:
        """Minimum number of history samples the state must absorb before predicting."""
        ...

    @property
    def horizon(self) -> int:
        """The native/trained horizon, not a bound on the accepted ``u_future`` length."""
        ...

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history ending at the same step into an initial state (Priming)."""
        ...

    def step(self, state: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Advance one position under raw control ``u`` -> ``(state', output)``."""
        ...

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from ``state`` under raw ``u_future`` -> ``(n_positions, n_outputs)``."""
        ...

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime` -> ``(B, state)``."""
        ...

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout` -> ``(B, n_positions, n_outputs)``."""
        ...

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new measurement ``y`` and applied control ``u`` into state (State Absorption)."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the state has absorbed enough history to begin predicting."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed state."""
        ...


@runtime_checkable
class RidgeFittable(Protocol):
    """Capability of a Predictor whose readout is linear in a feature vector it produces itself.

    A capability protocol, not a base-protocol member: it names the closed-form fit around the
    readout, so it stays independent of how the features are produced -- a depth-0 MLP is linear
    end-to-end, the ESN is nonlinear end-to-end with only a linear readout, and a depth-0
    observable MLP is linear end-to-end. ``is_linear`` would not express that, and a bare boolean
    would still leave the Ridge Trainer needing per-kind knowledge of how to extract features and
    where to write the result. The Trainer checks this capability at build time.
    """

    def design_normal_equations(
        self, trajectories: list[tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """Accumulate the normal equations from raw ``(u, y)`` trajectories: ``(G (f, f), P (f, c))``.

        The last feature column is the constant-1 bias, so the Ridge Trainer can leave it
        unregularized without knowing which feature it is.
        """
        ...

    def install_readout(self, A: FloatArray) -> None:
        """Write the closed-form-fitted readout ``A (c, f)``, bias column last, into the module."""
        ...


class ObservableModel(Protocol):
    """Protocol for models that forecast an Observable over the Control Horizon in one shot.

    A sibling of :class:`SymbolicModel`, not a variant of it: the stepping members are absent, so a
    model lacking them fails at build time instead of degrading the autoregressive path.
    """

    @property
    def state_shape(self) -> tuple[int, int]:
        """History state dimension shape."""
        ...

    @property
    def n_controls(self) -> int:
        """Control input count."""
        ...

    @property
    def n_channels(self) -> int:
        """Output channel count."""
        ...

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        ...

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        ...

    @property
    def native_horizon(self) -> int:
        """Control Horizon in samples the underlying artifact was fit at."""
        ...

    @property
    def geometry(self) -> ObservableGeometry:
        """The resolved Observable geometry the model forecasts on."""
        ...

    def n_frames(self, horizon: int) -> int:
        """Count the Frames the forecast emits over ``horizon`` samples."""
        ...

    def forecast(self, x0: ca.SX | ca.MX, u_seq: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Forecast the log-Observable: ``(state, (horizon, m) controls) -> (channels * values, frames)``."""
        ...

    @property
    def f_forecast(self) -> ca.Function:
        """Symbolic forecast function at the native horizon."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed or zero initial state vector."""
        ...

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new measurement y and previously applied control u into state."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if state has absorbed sufficient history to begin control."""
        ...
