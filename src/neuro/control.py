"""Controller components for the simulate framework.

:class:`ZeroController` is a no-op controller that always emits a zero control
vector -- the controller to use for *open-loop* runs through the
:class:`~simulate.simulation.Simulation` orchestrator (which always requires a
controller). :class:`StimWindowController` is an open-loop tES schedule: it holds
a fixed stimulation amplitude over a ``[onset, offset)`` time window and emits zero
otherwise, the orchestrated counterpart to the ``stim_window`` argument of
:func:`~neuro.jansen_rit.simulate_network`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from simulate.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Dataclass for ZeroController logging."""


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero vector.

    Emitting a fixed ``(n_u, 1)`` control sidesteps the dimensional mismatch the
    stock :class:`~simulate.controller.PIDController` hits on a vector-output
    plant: the orchestrator seeds the loop with a scalar ``y_k = 0.0``, so the
    first estimated state is a scalar while every later one is the full EEG
    measurement vector -- no single PID gain shape fits both steps.
    """

    def __init__(self, dt: float, n_u: int = 1) -> None:
        """Initialize the zero controller for an ``n_u``-dimensional control."""
        super().__init__(dt)
        self.n_u = n_u

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), n_u=int(config.get("n_u", 1)))

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, ZeroControllerLog]:
        """Return a zero control vector regardless of reference or state."""
        u_vec = np.zeros((self.n_u, 1), dtype=np.float64)
        return cast("np.ndarray", self.from_col_vec(u_vec)), ZeroControllerLog()


@dataclasses.dataclass(frozen=True)
class StimWindowControllerLog:
    """Dataclass for StimWindowController logging."""

    active: bool


class StimWindowController(Controller[StimWindowControllerLog]):
    """Open-loop tES schedule: a fixed amplitude held over a ``[onset, offset)`` window.

    The control is the per-electrode tES current the plant projects to nodes through
    ``connectome.gamma``; ``amplitude`` is a scalar shared by every electrode or a
    per-electrode vector of length ``n_u``. The window is half-open in seconds, so
    stimulation is active for ``onset <= t < offset`` and zero elsewhere. This is the
    orchestrated equivalent of the ``u_hat_tES`` / ``stim_window`` arguments of
    :func:`~neuro.jansen_rit.simulate_network`.
    """

    def __init__(
        self,
        dt: float,
        onset: float,
        offset: float,
        amplitude: ArrayLike,
        n_u: int = 1,
    ) -> None:
        """Initialize the windowed stimulation schedule.

        Parameters
        ----------
        dt
            Controller update step in seconds.
        onset, offset
            Half-open stimulation window ``[onset, offset)`` in seconds.
        amplitude
            tES current held during the window: a scalar shared by every electrode
            or a length-``n_u`` per-electrode vector.
        n_u
            Number of stimulation electrodes (control dimension).
        """
        super().__init__(dt)
        if offset < onset:
            msg = f"offset ({offset}) must be >= onset ({onset})"
            raise ValueError(msg)
        self.onset = onset
        self.offset = offset
        self.n_u = n_u

        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.broadcast_to(amp, (n_u,))
        elif amp.size != n_u:
            msg = f"amplitude has {amp.size} entries but n_u is {n_u}"
            raise ValueError(msg)
        self.amplitude = amp.reshape((n_u, 1))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            onset=float(config["onset"]),
            offset=float(config["offset"]),
            amplitude=config["amplitude"],
            n_u=int(config.get("n_u", 1)),
        )

    def update(
        self,
        t: float,
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, StimWindowControllerLog]:
        """Emit the stimulation amplitude inside the window, zero outside it."""
        active = self.onset <= t < self.offset
        u_vec = self.amplitude if active else np.zeros((self.n_u, 1), dtype=np.float64)
        return cast("np.ndarray", self.from_col_vec(u_vec)), StimWindowControllerLog(active=active)
