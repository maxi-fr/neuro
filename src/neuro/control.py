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
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from simulate.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Dataclass for ZeroController logging."""


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero ``(n_u,)`` vector.

    This is the controller for *open-loop* runs through the
    :class:`~simulate.simulation.Simulation` orchestrator, which always requires a
    controller; with all-zero control the plant's ``project_control`` is a no-op.
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
        return np.zeros(self.n_u, dtype=np.float64), ZeroControllerLog()


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
        u = self.amplitude.reshape(-1) if active else np.zeros(self.n_u, dtype=np.float64)
        return u, StimWindowControllerLog(active=active)


_MULTISINE_F_MIN_HZ = 1
_MULTISINE_F_MAX_HZ = 15
_EPS = 1e-12


def _multisine(n_samples: int, n_elec: int, amp: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Build a random-phase multisine of peak amplitude ``amp``, one column per electrode."""
    t = np.arange(n_samples) * dt
    freqs = np.arange(_MULTISINE_F_MIN_HZ, _MULTISINE_F_MAX_HZ + 1)
    out = np.zeros((n_samples, n_elec))
    for elec in range(n_elec):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
        sig = np.sin(2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
        out[:, elec] = amp * sig / max(np.abs(sig).max(), _EPS)
    return out


def build_input_schedule(  # noqa: PLR0913
    *,
    input_type: str,
    n_steps: int,
    transient_steps: int,
    n_elec: int,
    amp: float,
    hold_ms: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the per-step tES schedule ``(n_steps, n_elec)``; zero during the leading transient.

    ``ras`` holds a random uniform amplitude per block, ``prbs`` a random binary +/-amp, and
    ``multisine`` a random-phase sum of sinusoids; ``hold_ms`` sets the block length for the
    first two.
    """
    u = np.zeros((n_steps, n_elec))
    active = n_steps - transient_steps
    if active <= 0:
        return u

    if input_type in ("ras", "prbs"):
        hold = max(1, round(hold_ms / (dt * 1000.0)))
        n_blocks = (active + hold - 1) // hold
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_elec))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_elec))
        seq = np.repeat(block_vals, hold, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_elec, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    u[transient_steps:] = seq
    return u


@dataclasses.dataclass(frozen=True)
class WaveformControllerLog:
    """Dataclass for WaveformController logging (the emitted control is logged universally)."""


class WaveformController(Controller[WaveformControllerLog]):
    """Open-loop controller that plays back a precomputed per-electrode tES waveform.

    Ignores the reference and estimated state; at time ``t`` it emits the schedule sample for
    step ``k = round(t / dt)`` (clamped to the last sample). Used to inject persistently-exciting
    tES inputs (random-amplitude steps ``ras``, a random binary signal ``prbs``, or a
    ``multisine``) for plant identification -- configured by
    ``configs/simulation/jansen_rit_seizure_excited.yaml``.
    """

    def __init__(self, dt: float, schedule: ArrayLike) -> None:
        """Initialize from a precomputed ``(n_steps, n_u)`` per-electrode schedule."""
        super().__init__(dt)
        self.schedule = np.atleast_2d(np.asarray(schedule, dtype=np.float64))
        self.n_u = self.schedule.shape[1]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build the schedule from the excitation parameters in the config dict."""
        dt = float(config["dt"])
        schedule = build_input_schedule(
            input_type=str(config["input_type"]),
            n_steps=round(float(config["duration"]) / dt),
            transient_steps=round(float(config.get("transient_ms", 0.0)) / (dt * 1000.0)),
            n_elec=int(config["n_u"]),
            amp=float(config["amp"]),
            hold_ms=float(config.get("hold_ms", 50.0)),
            dt=dt,
            rng=np.random.default_rng(int(config["input_seed"])),
        )
        return cls(dt=dt, schedule=schedule)

    def update(
        self,
        t: float,
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, WaveformControllerLog]:
        """Emit the scheduled per-electrode current for the current step."""
        k = min(round(t / self.dt), self.schedule.shape[0] - 1)
        return self.schedule[k], WaveformControllerLog()
