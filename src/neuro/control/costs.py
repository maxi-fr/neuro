from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal.windows import hann
from trajopt.costs.base import CostFunction

from neuro.spectral import LOG_FLOOR, _frame_kernel_weights

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trajopt.dynamics.base import AbstractModel

    from neuro.config import StftGeometry
    from neuro.spectral import ObservableEnvelope, PsdEnvelope


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


class ReducedEffortCost(CostFunction):
    """Quadratic effort ``(w_u / horizon) * ||Z v||^2`` on the reduced controls of a Nullspace Frame.

    The reduced controls ``v`` are not the physical currents: the electrodes carry ``u = Z v``
    over the last-electrode elimination basis, so scoring ``||v||^2`` would price a different
    quantity than the unreduced problem does. For that basis ``Z^T Z = I + 1 1^T`` exactly, which
    is this cost's closed form. Writing it as a control-only cost rather than folding ``Z^T Z``
    into a quadratic cost's ``R`` keeps the state weight diagonal, which at deployment scale is a
    ``(n,)`` vector rather than the dense ``(n, n)`` matrix trajopt would promote it to.
    """

    w_u: jax.Array
    horizon: int = eqx.field(static=True)

    def __init__(self, *, n: int, m: int, w_u: float, horizon: int) -> None:
        """Initialize with the reduced state/control dimensions, effort weight, and horizon."""
        super().__init__(n=n, m=m)
        self.w_u = jnp.asarray(w_u)
        self.horizon = int(horizon)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate ``(w_u / horizon) * (sum(v^2) + (sum v)^2)``, the expanded currents' squared norm."""
        del x, t
        if u is None:
            return jnp.zeros(())
        v = jnp.asarray(u)
        return (self.w_u / self.horizon) * (jnp.sum(v**2) + jnp.sum(v) ** 2)


class SpectralHingeCost(CostFunction):
    """Mean squared one-sided log excess of the predicted spectrum over a healthy envelope.

    A whole-horizon functional: window ``m`` covers the Frames ``y_{m*hop} .. y_{m*hop + window
    - 1}`` the stage trajectory carries, so no single knot holds enough history to score it
    (``window`` typically exceeds the predictor's history window). ``stage_costs`` decodes those
    Frames straight out of the stage states via the model's output function and computes
    the exact windowed hinge with ``jnp.fft``; ``evaluate`` returns ``0`` at any single knot, so
    per-knot local expansions (native solvers) degrade to the quadratic/L1 objective rather than
    mis-score the hinge. The transcription path (single-shooting and multiple-shooting Ipopt)
    evaluates the exact hinge through ``stage_costs``.

    The stage trajectory carries exactly ``horizon`` Frames, so the window grid always spans a
    whole Control Horizon. It spans the Control Horizon's first ``horizon`` Frames rather than
    its last: the final Frame lives only in the terminal knot, which a Cost reads one state at a
    time, and an FFT window straddling that knot does not split into a stage term plus a terminal
    term. The grid is anchored one Frame earlier instead -- one sample of phase, not a shorter
    horizon.

    The periodogram convention is that of :func:`neuro.spectral.compute_periodograms` and of
    the training loss: periodic Hann, no per-segment detrend, one-sided and density-scaled.
    The DC bin is not scored. The reduction is a mean over ``(window, channel, bin)`` -- never
    over windows alone -- so ``w_psd`` stays independent of the window count and a hot
    sub-window cannot be cancelled by a cold one.
    """

    model: AbstractModel
    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    w: jax.Array
    power: jax.Array

    def __init__(
        self,
        model: AbstractModel,
        envelope: PsdEnvelope,
        *,
        w_psd: float,
        horizon: int,
    ) -> None:
        """Initialize from the dynamical model, the healthy envelope and the spectral weight.

        Parameters
        ----------
        model
            Dynamical model implementing output evaluation.
        envelope
            The healthy reference envelope; its ``window``/``hop`` geometry drives the Cost.
        w_psd
            Weight on the spectral Cost; ``0`` disables it.
        horizon
            Control Horizon in steps, which is the Frame count the stage trajectory carries;
            must be at least ``envelope.window``.
        """
        super().__init__(n=model.n, m=model.m)
        if model.p is None:
            msg = "model must define an output dimension p"
            raise ValueError(msg)
        self.model = model
        self.window = int(envelope.window)
        self.hop = int(envelope.hop)
        self.fs = float(envelope.fs)
        if envelope.power.shape[0] != model.p:
            msg = f"envelope has {envelope.power.shape[0]} channels but the model outputs {model.p}"
            raise ValueError(msg)
        if horizon < self.window:
            msg = f"horizon ({horizon}) is shorter than the envelope window ({self.window})"
            raise ValueError(msg)
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

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the exact windowed hinge over the stage Frames, concentrated in entry 0.

        Parameters
        ----------
        X
            Stage states ``(horizon, n)``, one Frame each.
        """
        del U, t
        y = jax.vmap(self.model.output)(X)  # (horizon, n_channels)
        log_power = jax_compute_log_power_frames(y, fs=self.fs, window=self.window, hop=self.hop)
        log_excess = log_power - jnp.log(self.power[None, :, 1:])
        hinge = jnp.maximum(0.0, log_excess) ** 2
        return jnp.zeros(X.shape[0]).at[0].set(self.w * jnp.mean(hinge))


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


def jax_compute_observable_frames(y: jax.Array, geometry: StftGeometry, *, fs: float) -> jax.Array:
    """Log-power Frames of ``y`` ``(H, n_channels)`` under an Observable geometry, differentiably.

    The jax twin of :func:`neuro.spectral.compute_log_power_frames`, stage for stage: periodic
    Hann periodogram, band slice, bin pooling, Frame Kernel, log. Returns
    ``(n_frames, n_channels, n_values)``.
    """
    n_segment, n_hop = geometry.n_segment, geometry.n_hop
    n_raw_frames = (y.shape[0] - n_segment) // n_hop + 1
    w_hann = jnp.asarray(hann(n_segment, sym=False))
    segments = jnp.stack([y[m * n_hop : m * n_hop + n_segment] for m in range(n_raw_frames)])
    spectrum = jnp.fft.rfft(segments * w_hann[None, :, None], axis=1)

    fold = np.full(n_segment // 2 + 1, 2.0)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    power = jnp.abs(spectrum) ** 2 * jnp.asarray(fold)[None, :, None] / (fs * jnp.sum(w_hann**2))
    power = jnp.moveaxis(power, 2, 1)  # (n_raw_frames, n_channels, n_bins)

    bin_lo, bin_hi = geometry.bin_range(fs)
    power = power[:, :, bin_lo:bin_hi]
    if geometry.n_bin_pool > 1:
        n_groups = power.shape[-1] // geometry.n_bin_pool
        power = power[:, :, : n_groups * geometry.n_bin_pool].reshape(
            power.shape[0], power.shape[1], n_groups, geometry.n_bin_pool
        )
        power = power.mean(axis=-1)

    if geometry.kernel_width > 1:
        weights = jnp.asarray(_frame_kernel_weights(geometry.kernel, geometry.kernel_width))
        n_frames = n_raw_frames - geometry.kernel_width + 1
        power = jnp.stack(
            [jnp.sum(power[i : i + geometry.kernel_width] * weights[:, None, None], axis=0) for i in range(n_frames)]
        )

    return jnp.log(power + LOG_FLOOR)


class ObservableFrameHingeCost(CostFunction):
    """Mean squared one-sided log excess of the predicted waveform's Observable Frames over a healthy envelope.

    The Observable adapter for a Predictor whose knot carries a waveform sample rather than a
    Frame: it runs the :class:`~neuro.config.StftGeometry` reduction over the stage trajectory
    itself, so a Frame-domain envelope scores a waveform-domain model and
    :class:`ObservableHingeCost`'s Observable joins a comparison of Costs on one Predictor.

    Whole-horizon, for the reason :class:`SpectralHingeCost` is: a Frame pools
    ``sample_support_steps`` knots, so no single knot holds one. ``evaluate`` returns ``0`` and
    ``stage_costs`` carries the exact value, which the transcription path scores and a native
    per-knot expansion would drop -- hence the :func:`has_whole_horizon_cost` registration.

    The Frame grid spans the stage trajectory's ``horizon`` samples, anchored one sample before
    the Control Horizon's last knot: a Frame straddling the terminal knot does not split into a
    stage term plus a terminal term.
    """

    model: AbstractModel
    geometry: StftGeometry = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    w: jax.Array
    power: jax.Array

    def __init__(
        self,
        model: AbstractModel,
        envelope: ObservableEnvelope,
        *,
        w_hinge: float,
        horizon: int,
    ) -> None:
        """Initialize from the dynamical model, the healthy Observable envelope and the weight.

        Parameters
        ----------
        model
            Dynamical model implementing output evaluation.
        envelope
            The healthy Observable envelope, log power of shape ``(n_channels, n_values)``; its
            geometry drives the reduction.
        w_hinge
            Weight on the hinge Cost; ``0`` disables it.
        horizon
            Control Horizon in knots, which is the sample count the stage trajectory carries;
            must cover one Frame's sample support.
        """
        super().__init__(n=model.n, m=model.m)
        if model.p is None:
            msg = "model must define an output dimension p"
            raise ValueError(msg)
        if envelope.power.shape[0] != model.p:
            msg = f"envelope has {envelope.power.shape[0]} channels but the model outputs {model.p}"
            raise ValueError(msg)
        expected_values = envelope.geometry.n_values(envelope.fs)
        if envelope.power.shape[1] != expected_values:
            msg = (
                f"envelope has {envelope.power.shape[1]} values per channel but its geometry implies {expected_values}"
            )
            raise ValueError(msg)
        support = envelope.geometry.sample_support_steps(envelope.fs)
        if horizon < support:
            msg = f"horizon ({horizon}) is shorter than the sample support of one Frame ({support})"
            raise ValueError(msg)
        self.model = model
        self.geometry = envelope.geometry
        self.fs = float(envelope.fs)
        self.w = jnp.asarray(w_hinge)
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

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the hinge over the Frames the stage waveform carries, concentrated in entry 0.

        Parameters
        ----------
        X
            Stage states ``(horizon, n)``, one waveform sample each.
        """
        del U, t
        y = jax.vmap(self.model.output)(X)
        frames = jax_compute_observable_frames(y, self.geometry, fs=self.fs)
        hinge = jnp.maximum(0.0, frames - self.power[None]) ** 2
        return jnp.zeros(X.shape[0]).at[0].set(self.w * jnp.mean(hinge))


class ObservableHingeCost(CostFunction):
    """Mean squared one-sided log excess of predicted Frames over a healthy Observable envelope.

    Scored on the output space R^p where one knot state represents one Observable Frame.
    Wrapped in :class:`~trajopt.costs.output.OutputCost` for receding-horizon optimal control problems.
    """

    envelope: ObservableEnvelope = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    w: jax.Array
    power: jax.Array

    def __init__(
        self,
        envelope: ObservableEnvelope,
        *,
        w_hinge: float,
        horizon: int,
        terminal: bool = False,
    ) -> None:
        """Initialize from the healthy Observable envelope, weight, and horizon.

        Parameters
        ----------
        envelope
            The healthy Observable reference envelope.
        w_hinge
            Weight on the hinge Cost; ``0`` disables it.
        horizon
            Control Horizon in Frames; must be at least 1.
        terminal
            Whether this instance is the terminal Cost scoring the Control Horizon's last Frame.
        """
        expected_outputs = int(envelope.power.shape[0] * envelope.power.shape[1])
        if horizon < 1:
            msg = f"horizon ({horizon}) must be at least 1"
            raise ValueError(msg)
        super().__init__(n=expected_outputs, m=0, terminal=terminal)
        self.envelope = envelope
        self.horizon = int(horizon)
        self.w = jnp.asarray(w_hinge)
        self.power = jnp.asarray(envelope.power).reshape(-1)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Score the single Frame x in R^p against the healthy envelope."""
        del u, t
        excess = jnp.maximum(0.0, x - self.power)
        return (self.w / (self.horizon * self.power.shape[0])) * jnp.sum(excess**2)

    def as_terminal(self) -> ObservableHingeCost:
        """Derive a terminal cost scoring the final Frame."""
        return ObservableHingeCost(
            self.envelope,
            w_hinge=float(self.w),
            horizon=self.horizon,
            terminal=True,
        )


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
    return isinstance(cost, SpectralHingeCost | ObservableFrameHingeCost)
