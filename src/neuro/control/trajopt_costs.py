from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal.windows import hann
from trajopt.costs.base import CostFunction

from neuro.spectral import LOG_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.control.trajopt_mpc import WaveformMLPModel
    from neuro.spectral import PsdEnvelope
    from neuro.types import FloatArray


class SumCost(CostFunction):
    """Stage cost summing several sub-costs, each evaluated by its own path.

    ``stage_costs`` adds the sub-costs' stacked values: a per-knot cost such as
    :class:`L1ControlCost` contributes one entry per stage, while a whole-horizon cost such as
    :class:`SpectralHingeCost` concentrates its value in a single entry, so
    ``Objective.cost`` reports the exact total either way. ``evaluate`` sums the per-knot
    evaluations, which is what per-knot Taylor expansions (native solvers, the multiple-shooting
    Hessian) consume.
    """

    costs: tuple[CostFunction, ...]

    def __init__(self, costs: Sequence[CostFunction]) -> None:
        """Initialize from the sub-costs, which must agree on ``n`` and ``m``."""
        super().__init__(n=costs[0].n, m=costs[0].m)
        self.costs = tuple(costs)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the summed per-knot cost at one knot."""
        value = jnp.zeros(())
        for cost in self.costs:
            value = value + cost.evaluate(x, u, t)
        return value

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Sum the sub-costs' stacked stage values of shape ``(N - 1,)``."""
        total = jnp.zeros(X.shape[0])
        for cost in self.costs:
            total = total + cost.stage_costs(X, U, t)
        return total


class L1ControlCost(CostFunction):
    """Smooth surrogate for the horizon-mean L1 control penalty ``(w_l1 / horizon) * sum_k ||u_k||_1``.

    The incumbent's epigraph reformulation needs slack variables, which trajopt's objective and
    decision-vector layout cannot express, so the L1 becomes a per-knot stage cost. The plain
    norm is kept smooth as ``sqrt(u^2 + eps^2)`` because this ticket's solver testing showed the
    raw ``|u|`` kink breaks Ipopt's limited-memory search-direction computation (status -3
    after hundreds of iterations, on a problem that converges in under 50 without it), while
    the smooth surrogate converges to the same minimizer as the incumbent's epigraph (status 1,
    acceptable level) and the native ALTRO backend solves it directly. Cost parity with the
    epigraph is therefore approximate up to ``eps``, exact in the limit; ``eps = 1e-3`` keeps
    the minimizer within the parity test's tolerance (measured at ``8e-4`` against the
    incumbent's exact-zero controls, versus ``2.4e-2`` at ``eps = 1e-2``).
    """

    w_l1: jax.Array
    eps: jax.Array
    horizon: int = eqx.field(static=True)

    def __init__(self, *, n: int, m: int, w_l1: float, horizon: int, eps: float = 1e-3) -> None:
        """Initialize with the state/control dimensions, the weight, the horizon and the smoothness.

        Parameters
        ----------
        n, m
            Model state and control dimensions.
        w_l1
            Weight on the sparse-stimulation penalty; ``0`` disables it.
        horizon
            Control Horizon in steps, for the horizon-mean reduction.
        eps
            Smoothness radius of the surrogate; ``sqrt(u^2 + eps^2)`` replaces ``|u|``. Smaller
            ``eps`` tightens the sparsity residual toward the epigraph's exact zeros at the
            price of a stiffer solve; ``1e-3`` sits inside the parity tolerance.
        """
        super().__init__(n=n, m=m)
        self.w_l1 = jnp.asarray(w_l1)
        self.eps = jnp.asarray(eps)
        self.horizon = int(horizon)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the per-knot smooth L1 penalty ``(w_l1 / horizon) * sum(sqrt(u^2 + eps^2))``."""
        del x, t
        u_arr = jnp.asarray(u)
        return (self.w_l1 / self.horizon) * jnp.sum(jnp.sqrt(u_arr**2 + self.eps**2))


class SpectralHingeCost(CostFunction):
    """Mean squared one-sided log excess of the predicted spectrum over a healthy envelope.

    A whole-horizon functional: window ``m`` covers predicted outputs ``y_{m*hop+1} .. y_{m*hop
    + window}``, so no single knot carries enough history to score it (``window`` typically
    exceeds the predictor's ``n_y``). ``stage_costs`` therefore decodes the predicted outputs
    from the stage states -- one extra model step recovers the terminal output the stage knots
    do not carry -- and computes the exact windowed hinge with ``jnp.fft``; ``evaluate``
    returns ``0`` at any single knot, so per-knot local expansions (native solvers) degrade to
    the quadratic/L1 objective rather than mis-score the hinge. The transcription path
    (single-shooting and multiple-shooting Ipopt) evaluates the exact hinge through
    ``stage_costs``.

    The periodogram convention is that of :func:`neuro.spectral.compute_periodograms` and of
    the training loss: periodic Hann, no per-segment detrend, one-sided and density-scaled.
    The DC bin is not scored. The reduction is a mean over ``(window, channel, bin)`` -- never
    over windows alone -- so ``w_psd`` stays independent of the window count and a hot
    sub-window cannot be cancelled by a cold one.
    """

    n_y: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    n_windows: int = eqx.field(static=True)
    model: WaveformMLPModel
    w: jax.Array
    power: jax.Array

    def __init__(
        self,
        model: WaveformMLPModel,
        envelope: PsdEnvelope,
        *,
        w_psd: float,
        horizon: int,
    ) -> None:
        """Initialize from the waveform model, the healthy envelope and the spectral weight.

        Parameters
        ----------
        model
            The trajopt waveform model adapter whose predicted outputs are windowed; its
            ``z_last`` decode matches ``NNSymbolicModel.output``.
        envelope
            The healthy reference envelope; its ``window``/``hop`` geometry drives the cost.
        w_psd
            Weight on the spectral cost; ``0`` disables it.
        horizon
            Control Horizon in steps; must be at least ``envelope.window``.
        """
        super().__init__(n=model.n, m=model.m)
        self.model = model
        self.n_y = int(model.n_y)
        self.n_channels = int(model.n_channels)
        self.window = int(envelope.window)
        self.hop = int(envelope.hop)
        self.fs = float(envelope.fs)
        self.n_bins = int(envelope.power.shape[1])
        if envelope.power.shape[0] != model.n_channels:
            msg = f"envelope has {envelope.power.shape[0]} channels but the model outputs {model.n_channels}"
            raise ValueError(msg)
        if horizon < self.window:
            msg = f"horizon ({horizon}) is shorter than the envelope window ({self.window})"
            raise ValueError(msg)
        self.n_windows = (horizon - self.window) // self.hop + 1
        self.w = jnp.asarray(w_psd)
        self.power = jnp.asarray(envelope.power)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Return ``0``: the hinge is whole-horizon and is scored by :meth:`stage_costs`."""
        del x, u, t
        return jnp.zeros(())

    def _predicted_outputs(self, X: jax.Array, U: jax.Array) -> jax.Array:
        """Decode the raw predicted outputs ``y_1 .. y_H`` from the stage states and controls.

        Stage knot ``k`` carries the prediction ``y_k`` as its newest y-window row; the
        terminal output ``y_H`` is recovered with one extra model step from the last stage
        knot, since the stage trajectory excludes the terminal state.
        """
        n_z = self.n_y * self.n_channels
        z_last = X[1:, n_z - self.n_channels : n_z]
        y = z_last * self.model.y_scale + self.model.y_center
        x_term = self.model.discrete_dynamics(X[-1], U[-1], 0.0, 0.0)
        z_term = x_term[n_z - self.n_channels : n_z]
        return jnp.concatenate([y, (z_term * self.model.y_scale + self.model.y_center)[None, :]], axis=0)

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the exact windowed hinge, concentrated in the first stage entry.

        The hinge is a mean over ``(window, channel, bin)`` of the squared one-sided log
        excess, so it contributes a single scalar to the stage-cost sum; placing it in entry 0
        leaves every other stage's local cost untouched and makes ``sum(stage_costs)`` the
        exact total the CasADi graph reports.
        """
        del t
        y = self._predicted_outputs(X, U)  # (H, n_channels)
        segments = jnp.stack([y[m * self.hop : m * self.hop + self.window] for m in range(self.n_windows)])
        w_hann = jnp.asarray(hann(self.window, sym=False))
        y_tapered = segments * w_hann[None, :, None]
        spectrum = jnp.fft.rfft(y_tapered, axis=1)
        power = jnp.abs(spectrum) ** 2
        fold = np.full(self.n_bins, 2.0)
        fold[0] = 1.0
        if self.window % 2 == 0:
            fold[-1] = 1.0
        power = power * jnp.asarray(fold)[None, :, None] / (self.fs * jnp.sum(w_hann**2))
        power = jnp.moveaxis(power, 2, 1)  # (n_windows, n_channels, n_bins)
        log_excess = jnp.log(power[..., 1:] + LOG_FLOOR) - jnp.log(self.power[None, :, 1:])
        hinge = jnp.maximum(0.0, log_excess) ** 2
        value = self.w * jnp.mean(hinge)
        return jnp.zeros(X.shape[0]).at[0].set(value)


class ObservableHingeCost(CostFunction):
    """Mean squared one-sided log excess of a forecast Observable over its healthy envelope.

    Per-knot local: ``evaluate(x, u, t)`` hinges one forecast Frame ``x`` of shape
    ``(n_channels * n_values,)`` in raw log units against the ``ObservableGeometry``-derived
    reference, so unlike the spectral hinge it needs no whole-trajectory ``stage_costs``
    override and works with every solver backend. The reduction is a mean over
    ``(frame, channel, value)`` -- never over frames alone -- matching the training-time Loss
    and keeping ``w_psd`` with exactly the meaning it has on the spectral path.

    ``x`` is the Frame's forecast value; the Observable model adapter (ticket 03) exposes its
    state as that value, or composes this cost with its readout.
    """

    n_values: int = eqx.field(static=True)
    n_frames: int = eqx.field(static=True)
    w: jax.Array
    log_reference: jax.Array

    def __init__(
        self,
        *,
        n: int,
        m: int,
        log_reference: FloatArray,
        w_psd: float,
        n_frames: int,
    ) -> None:
        """Initialize from the reference reduced onto the Frame's value grid, in log units.

        Parameters
        ----------
        n, m
            Model state and control dimensions.
        log_reference
            The healthy envelope reduced onto the Frame grid, ``(n_channels, n_values)`` in
            raw log units -- the ``ObservableGeometry``-derived reference shared with the
            training-time Loss.
        w_psd
            Weight on the hinge.
        n_frames
            Frame count the horizon's forecast holds, for the horizon-mean reduction.
        """
        super().__init__(n=n, m=m)
        self.log_reference = jnp.asarray(log_reference).reshape(-1)
        self.n_values = self.log_reference.shape[0]
        self.n_frames = int(n_frames)
        self.w = jnp.asarray(w_psd)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Hinge one forecast Frame against the reference: ``w * mean_v(hinge^2) / n_frames``."""
        del u, t
        hinge = jnp.maximum(0.0, x - self.log_reference) ** 2
        return self.w * jnp.mean(hinge) / self.n_frames

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the per-Frame hinge over the forecast trajectory, one entry per stage."""
        return jax.vmap(self.evaluate)(X, U, t)
