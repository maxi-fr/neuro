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
    exceeds the predictor's history window). ``stage_costs`` slices the predicted outputs
    directly from the stage states using the build-time center/scale arrays and computes the
    exact windowed hinge with ``jnp.fft``; ``evaluate`` returns ``0`` at any single knot, so
    per-knot local expansions (native solvers) degrade to the quadratic/L1 objective rather than
    mis-score the hinge. The transcription path (single-shooting and multiple-shooting Ipopt)
    evaluates the exact hinge through ``stage_costs``.

    The periodogram convention is that of :func:`neuro.spectral.compute_periodograms` and of
    the training loss: periodic Hann, no per-segment detrend, one-sided and density-scaled.
    The DC bin is not scored. The reduction is a mean over ``(window, channel, bin)`` -- never
    over windows alone -- so ``w_psd`` stays independent of the window count and a hot
    sub-window cannot be cancelled by a cold one.
    """

    n_channels: int = eqx.field(static=True)
    n_y: int = eqx.field(static=True)
    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    n_windows: int = eqx.field(static=True)
    w: jax.Array
    power: jax.Array
    y_center: jax.Array
    y_scale: jax.Array

    def __init__(  # noqa: PLR0913 -- geometry, standardizers, envelope, weight and horizon
        self,
        *,
        n: int,
        m: int,
        n_y: int,
        n_channels: int,
        y_center: FloatArray | jax.Array,
        y_scale: FloatArray | jax.Array,
        envelope: PsdEnvelope,
        w_psd: float,
        horizon: int,
    ) -> None:
        """Initialize from geometry, standardizers, the healthy envelope and the spectral weight.

        Parameters
        ----------
        n, m
            Model state and control dimensions.
        n_y
            Past output steps in the model's history window.
        n_channels
            Physical EEG channel count.
        y_center, y_scale
            Center and scale arrays mapping standardized state units to raw EEG units.
        envelope
            The healthy reference envelope; its ``window``/``hop`` geometry drives the cost.
        w_psd
            Weight on the spectral cost; ``0`` disables it.
        horizon
            Control Horizon in steps; must be at least ``envelope.window``.
        """
        super().__init__(n=n, m=m)
        self.n_channels = int(n_channels)
        self.n_y = int(n_y)
        self.window = int(envelope.window)
        self.hop = int(envelope.hop)
        self.fs = float(envelope.fs)
        self.n_bins = int(envelope.power.shape[1])
        if envelope.power.shape[0] != n_channels:
            msg = f"envelope has {envelope.power.shape[0]} channels but the model outputs {n_channels}"
            raise ValueError(msg)
        if horizon < self.window:
            msg = f"horizon ({horizon}) is shorter than the envelope window ({self.window})"
            raise ValueError(msg)
        self.n_windows = (horizon - self.window) // self.hop + 1
        self.w = jnp.asarray(w_psd)
        self.power = jnp.asarray(envelope.power)
        self.y_center = jnp.asarray(y_center)
        self.y_scale = jnp.asarray(y_scale)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Return ``0``: the hinge is whole-horizon and is scored by :meth:`stage_costs`."""
        del x, u, t
        return jnp.zeros(())

    def _predicted_outputs(self, X: jax.Array) -> jax.Array:
        """Decode the raw predicted outputs from the stage states without stepping the model."""
        z_last = slice((self.n_y - 1) * self.n_channels, self.n_y * self.n_channels)
        return X[1:, z_last] * self.y_scale + self.y_center

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the exact windowed hinge over the stage trajectory, concentrated in entry 0."""
        del U, t
        y = self._predicted_outputs(X)  # (H - 1, n_channels)
        if y.shape[0] < self.window:
            return jnp.zeros(X.shape[0])
        log_power = jax_compute_log_power_frames(y, fs=self.fs, window=self.window, hop=self.hop)
        log_excess = log_power - jnp.log(self.power[None, :, 1:])
        hinge = jnp.maximum(0.0, log_excess) ** 2
        value = self.w * jnp.mean(hinge)
        return jnp.zeros(X.shape[0]).at[0].set(value)


def jax_compute_log_power_frames(
    y: jax.Array,
    *,
    fs: float,
    window: int,
    hop: int,
) -> jax.Array:
    """Log-power Frames of ``y`` ``(H, n_channels)`` on the unpooled, unbanded waveform STFT grid.

    Returns ``(n_windows, n_channels, window // 2)`` excluding DC.
    """
    n_windows = (y.shape[0] - window) // hop + 1
    segments = jnp.stack([y[m * hop : m * hop + window] for m in range(n_windows)])
    w_hann = jnp.asarray(hann(window, sym=False))
    y_tapered = segments * w_hann[None, :, None]
    spectrum = jnp.fft.rfft(y_tapered, axis=1)
    power = jnp.abs(spectrum) ** 2
    n_bins = window // 2 + 1
    fold = np.full(n_bins, 2.0)
    fold[0] = 1.0
    if window % 2 == 0:
        fold[-1] = 1.0
    power = power * jnp.asarray(fold)[None, :, None] / (fs * jnp.sum(w_hann**2))
    power = jnp.moveaxis(power, 2, 1)  # (n_windows, n_channels, n_bins)
    return jnp.log(power[..., 1:] + LOG_FLOOR)


class ExcludeInitialKnotState(CostFunction):
    """Wrap a per-knot stage cost, dropping its knot-0 state-only term in ``stage_costs``.

    The incumbent rollout cost steps first and scores ``y_next``, never the absorbed state at
    knot 0, so the transcription/reported-cost path (``stage_costs``) subtracts the wrapped
    cost's knot-0 state term. ``evaluate`` stays the wrapped cost's single-knot value, so native
    per-knot expansions still work; the dropped term is constant in the controls and never moves
    the minimizer.
    """

    inner: CostFunction

    def __init__(self, inner: CostFunction) -> None:
        """Initialize from the single-knot cost whose knot-0 state term is dropped."""
        super().__init__(n=inner.n, m=inner.m, terminal=inner.terminal)
        self.inner = inner

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the wrapped cost unchanged at one knot."""
        return self.inner.evaluate(x, u, t)

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate per-knot, dropping the wrapped cost's knot-0 state-only term."""
        base = self.inner.stage_costs(X, U, t)
        state0 = self.inner.evaluate(X[0], None, t[0])
        return base.at[0].add(-state0)


def has_whole_horizon_cost(cost: CostFunction) -> bool:
    """Whether ``cost`` or any :class:`SumCost` sub-cost is scored only through ``stage_costs``.

    Whole-horizon costs return ``0`` from ``evaluate`` and concentrate their value in
    ``stage_costs``; a native expansion-only solver would silently drop them.
    """
    if isinstance(cost, SumCost):
        return any(has_whole_horizon_cost(sub) for sub in cost.costs)
    return isinstance(cost, SpectralHingeCost)
