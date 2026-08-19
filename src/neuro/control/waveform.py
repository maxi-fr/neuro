from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import Field, PositiveFloat
from simulate.controller import Controller

from neuro.config import StrictConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from neuro.types import FloatArray

_MULTISINE_F_MIN_HZ = 1
_MULTISINE_F_MAX_HZ = 15
_EPS = 1e-12


def _multisine(n_samples: int, n_controls: int, amp: float, dt: float, rng: np.random.Generator) -> FloatArray:
    """Build a random-phase multisine of peak amplitude ``amp``, one column per electrode."""
    t = np.arange(n_samples) * dt
    freqs = np.arange(_MULTISINE_F_MIN_HZ, _MULTISINE_F_MAX_HZ + 1)
    out = np.zeros((n_samples, n_controls))
    for elec in range(n_controls):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
        sig = np.sin(2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
        out[:, elec] = amp * sig / max(np.abs(sig).max(), _EPS)
    return out


def build_input_schedule(  # noqa: PLR0913
    *,
    input_type: Literal["ras", "prbs", "multisine"],
    n_steps: int,
    transient_steps: int,
    n_controls: int,
    amp: float,
    hold_ms: float | Sequence[float],
    dt: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Build the per-step tES schedule ``(n_steps, n_controls)``; zero during the leading transient.

    ``ras`` holds a random uniform amplitude per block, ``prbs`` a random binary +/-amp, and
    ``multisine`` a random-phase sum of sinusoids; ``hold_ms`` sets the block length for the
    first two.
    """
    u = np.zeros((n_steps, n_controls))
    active = n_steps - transient_steps
    if active <= 0:
        return u

    if input_type in ("ras", "prbs"):
        holds = np.atleast_1d(np.asarray(hold_ms, dtype=np.float64))
        hold_steps = np.maximum(1, np.round(holds / (dt * 1000.0)).astype(int))
        n_blocks = int(np.ceil(active / hold_steps.min()))
        if hold_steps.size > 1:
            # Draw p ~ 1 / length so each entry gets an equal share of *time*: drawn uniformly, a
            # value's share is proportional to its own length and the longest hold eats the run.
            p = (1.0 / hold_steps) / np.sum(1.0 / hold_steps)
            lengths = rng.choice(hold_steps, size=n_blocks, p=p)
        else:
            lengths = np.repeat(hold_steps, n_blocks)
        n_blocks = int(np.searchsorted(np.cumsum(lengths), active) + 1)
        lengths = lengths[:n_blocks]
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_controls))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_controls))
        seq = np.repeat(block_vals, lengths, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_controls, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    zero_sum = seq - seq.mean(axis=1, keepdims=True)
    peak = np.abs(zero_sum).max(axis=1, keepdims=True)
    zero_sum *= np.minimum(1.0, amp / np.where(peak > 0.0, peak, 1.0))
    u[transient_steps:] = zero_sum
    return u


class _WaveformControllerConfig(StrictConfig):
    """Config schema for :class:`WaveformController`."""

    dt: float = Field(gt=0)
    input_type: Literal["ras", "prbs", "multisine"]
    duration: float = Field(gt=0)
    n_u: int = Field(ge=1)
    amp: float = Field(ge=0)
    input_seed: int = Field(ge=0)
    transient_ms: float = Field(default=0.0, ge=0)
    hold_ms: PositiveFloat | list[PositiveFloat] = 50.0


@dataclasses.dataclass(frozen=True)
class WaveformControllerLog:
    """Log carrying the applied control; excitation datasets are identified against it."""

    u: FloatArray


class WaveformController(Controller[WaveformControllerLog]):
    """Open-loop controller that plays back a precomputed per-electrode tES waveform."""

    def __init__(self, dt: float, schedule: ArrayLike) -> None:
        """Initialize from a precomputed ``(n_steps, n_u)`` per-electrode schedule."""
        super().__init__(dt)
        self.schedule = np.atleast_2d(np.asarray(schedule, dtype=np.float64))
        self.n_u = self.schedule.shape[1]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build the schedule from the excitation parameters in the config dict."""
        cfg = _WaveformControllerConfig.model_validate(config)
        schedule = build_input_schedule(
            input_type=cfg.input_type,
            n_steps=round(cfg.duration / cfg.dt),
            transient_steps=round(cfg.transient_ms / (cfg.dt * 1000.0)),
            n_controls=cfg.n_u,
            amp=cfg.amp,
            hold_ms=cfg.hold_ms,
            dt=cfg.dt,
            rng=np.random.default_rng(cfg.input_seed),
        )
        return cls(dt=cfg.dt, schedule=schedule)

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,  # noqa: ARG002
    ) -> tuple[FloatArray, WaveformControllerLog]:
        """Emit the scheduled per-electrode current for the current step."""
        k = round(t / self.dt)
        u = np.zeros(self.n_u, dtype=np.float64) if k >= self.schedule.shape[0] else self.schedule[k]
        return u, WaveformControllerLog(u=u.copy())
