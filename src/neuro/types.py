from __future__ import annotations

from typing import Literal, Protocol, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
StrArray = npt.NDArray[np.str_]

Activation = Literal["relu", "tanh", "softplus"]
"""Names of the supported MLP activation functions."""

ACTIVATIONS: frozenset[str] = frozenset(get_args(Activation))
"""Registry validating an activation name at checkpoint-load time."""

Layers = tuple[tuple[FloatArray, FloatArray], ...]
"""Per-layer ``(weight (out, in), bias (out,))`` pairs in forward-pass order."""


@runtime_checkable
class Predictor(Protocol):
    """Runtime-only, raw-units interface every surrogate predictor implements.

    Predicts the next output(s) from an opaque internal state. State is the module's own business
    -- a shift register for the waveform MLP -- and callers never
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
    readout, so it stays independent of how the features are produced -- a depth-0 MLP and a
    depth-0 observable MLP are both linear end-to-end. ``is_linear`` would not express that, and
    a bare boolean would still leave the Ridge Trainer needing per-kind knowledge of how to
    extract features and where to write the result. The Trainer checks this capability at build
    time.
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
