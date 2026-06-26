from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from sklearn.decomposition import PCA
from tqdm import tqdm

from neuro.jansen_rit import delays_to_steps
from neuro.jansen_rit_jax import (
    coupling_from_history_jax,
    eeg_jax,
    heun_step_det_jax,
    lfp_jax,
    sigmoid_jax,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuro.jansen_rit_jax import JRParamsJax

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
JaxArray = jax.Array

# Differentiable (low, high) bounds for the scalar/vector free parameters, mirroring the
# ``bounds`` metadata on :class:`neuro.jansen_rit.JansenRitParams`. The data-dependent
# ``w_weights`` cap is added at runtime by :func:`make_bounds`.
_BOUNDS: dict[str, tuple[float, float]] = {
    "A": (0.0, 10.0),
    "mean_input": (0.0, 500.0),
    "K": (0.0, 10.0),
    "eeg_gain": (0.0, 10.0),
}


@dataclass(frozen=True)
class Reduction:
    """PCA reduction of the full network into ``N`` virtual modes."""

    gain: FloatArray  # coarse leadfield, shape (n_channels, N)
    components: FloatArray  # PCA components (modes -> regions), shape (N, R)
    delay_steps: IntArray  # smart constant delays, shape (N, N)
    explained_variance: float  # fraction of node-output variance retained


def reduce_via_pca(
    node_output: FloatArray, gain: FloatArray, full_delays_ms: FloatArray, dt: float, n_components: int
) -> Reduction:
    """PCA-reduce the full network to ``N`` virtual modes (the coarse-graining step).

    Mirrors ``notebooks/coarse_graining.py``: fit PCA on the per-region LFP, project the
    leadfield through the loadings (``gain @ Vᵀ``), and build a constant smart delay
    matrix via :func:`loading_weighted_delays`. The PCA basis is regime-specific, so fit
    ``node_output`` on data representative of the dynamics to be identified (e.g. include
    the seizure transition) -- a basis that does not span the data is unfittable.

    Parameters
    ----------
    node_output
        Per-region LFP ``x2 - x3`` over time, shape ``(T, R)``.
    gain
        Full leadfield, shape ``(n_channels, R)``.
    full_delays_ms
        Anatomical region-to-region delays in milliseconds, shape ``(R, R)``.
    dt
        Integration step (seconds) of the *reduced* model.
    n_components
        Number of virtual modes ``N``.

    Returns
    -------
    Reduction
        Coarse leadfield, PCA components, smart delays, and retained variance fraction.
    """
    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(node_output, dtype=np.float64))
    components = np.asarray(pca.components_, dtype=np.float64)
    return Reduction(
        gain=np.asarray(gain, dtype=np.float64) @ components.T,
        components=components,
        delay_steps=loading_weighted_delays(components, full_delays_ms, dt),
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def loading_weighted_delays(components: FloatArray, full_delays_ms: FloatArray, dt: float) -> npt.NDArray[np.int64]:
    """Initialise a constant delay matrix for PCA virtual modes from anatomical delays.

    PCA modes have no tracts, yet the reduced coupling should stay delayed. We project
    the full ``R x R`` anatomical delay matrix through the (absolute) mode loadings, so
    ``delay[m, n]`` is the loading-weighted mean of the region-pair delays under modes
    ``m`` and ``n``: ``(|V_m| D |V_n|) / (|V_m| 1)(|V_n| 1)``. The result is symmetric
    with a zero diagonal and is *fixed* during identification (index-based history is
    non-differentiable; see paper sections 4.1.4 / 9.5.5), so this init matters.

    Parameters
    ----------
    components
        PCA components mapping modes to regions, shape ``(N, R)`` (``pca.components_``).
    full_delays_ms
        Anatomical region-to-region delays in milliseconds, shape ``(R, R)``.
    dt
        Integration step in seconds.

    Returns
    -------
    numpy.ndarray
        Reduced delay matrix in integer steps, shape ``(N, N)``.
    """
    V = np.abs(np.asarray(components, dtype=np.float64))
    D = np.asarray(full_delays_ms, dtype=np.float64)
    wsum = V.sum(axis=1)
    den = np.outer(wsum, wsum)
    eff_ms = (V @ D @ V.T) / np.where(den > 0.0, den, 1.0)
    np.fill_diagonal(eff_ms, 0.0)
    return delays_to_steps(eff_ms, dt)


# --------------------------------------------------------------------------------------
# Differentiable bound reparametrisation
# --------------------------------------------------------------------------------------
def _to_constrained(raw: JaxArray, low: float, high: float) -> JaxArray:
    """Map an unconstrained raw value into ``(low, high)`` via a sigmoid."""
    return low + (high - low) * jax.nn.sigmoid(raw)


def _to_raw(value: FloatArray | float, low: float, high: float) -> FloatArray:
    """Inverse of :func:`_to_constrained` (host-side, for initialising ``theta``)."""
    frac = (np.asarray(value, dtype=np.float64) - low) / (high - low)
    frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
    return np.log(frac / (1.0 - frac))


def _w_from_tri(tri: JaxArray, n: int) -> JaxArray:
    """Scatter a strict-upper-triangle vector into a symmetric zero-diagonal matrix."""
    iu = jnp.triu_indices(n, k=1)
    upper = jnp.zeros((n, n), dtype=tri.dtype).at[iu].set(tri)
    return upper + upper.T


def make_bounds(w_max: float) -> dict[str, tuple[float, float]]:
    """Differentiable ``(low, high)`` bounds for every free parameter.

    The fixed physical bounds (:data:`_BOUNDS`) plus the data-dependent connection-weight
    cap ``(0, w_max)``, so :func:`pack_theta` / :func:`build_params` need only one mapping.
    """
    return {**_BOUNDS, "w_weights": (0.0, w_max)}


def pack_theta(
    base: JRParamsJax, free_names: Sequence[str], bounds: dict[str, tuple[float, float]]
) -> dict[str, JaxArray]:
    """Build the unconstrained optimisation variables from a base parameter set.

    Parameters
    ----------
    base
        Base reduced-model parameters; free entries seed the initial guess.
    free_names
        Subset of ``{"A", "w_weights", "eeg_gain", "K", "mean_input"}`` to identify.
    bounds
        ``(low, high)`` per free parameter (see :func:`make_bounds`).

    Returns
    -------
    dict
        Raw (unconstrained) leaves keyed by parameter name; feed to :func:`build_params`.
    """
    n = base.n_nodes
    theta: dict[str, JaxArray] = {}
    for name in free_names:
        if name not in bounds:
            msg = f"Unknown or unsupported free parameter {name!r}"
            raise ValueError(msg)
        low, high = bounds[name]
        if name == "w_weights":
            tri = np.asarray(base.w_weights, dtype=np.float64)[np.triu_indices(n, k=1)]
            theta[name] = jnp.asarray(_to_raw(tri, low, high))
        else:
            theta[name] = jnp.asarray(_to_raw(np.asarray(getattr(base, name)), low, high))
    return theta


def build_params(
    theta: dict[str, JaxArray], base: JRParamsJax, free_names: Sequence[str], bounds: dict[str, tuple[float, float]]
) -> JRParamsJax:
    """Rebuild full :class:`JRParamsJax` from raw ``theta`` with bounds applied.

    Parameters
    ----------
    theta
        Raw leaves from :func:`pack_theta` (and gradient updates thereof).
    base
        Base parameters supplying every non-free field.
    free_names
        The identified parameter names.
    bounds
        ``(low, high)`` per free parameter (see :func:`make_bounds`).

    Returns
    -------
    JRParamsJax
        Parameters with the free entries mapped back into their valid ranges.
    """
    updates: dict[str, JaxArray] = {}
    for name in free_names:
        low, high = bounds[name]
        constrained = _to_constrained(theta[name], low, high)
        updates[name] = _w_from_tri(constrained, base.n_nodes) if name == "w_weights" else constrained
    return replace(base, **updates)


def rollout_tbptt(x0: JaxArray, controls_node: JaxArray, p: JRParamsJax, dt: float, window: int) -> JaxArray:
    """Deterministic Heun rollout with truncated backprop every ``window`` steps.

    Identical in value to :func:`neuro.jansen_rit_jax.rollout_jax`, but a
    ``stop_gradient`` is applied to the scan carry (state and history) at each window
    boundary, so reverse-mode gradients propagate at most ``window`` steps. This bounds
    the exponential gradient growth Jansen-Rit exhibits past its Lyapunov time.

    Parameters
    ----------
    x0
        Initial state, shape ``(6, N)``.
    controls_node
        Per-step node-projected tES, shape ``(T, N)`` (zeros for uncontrolled fits).
    p
        Network parameters.
    dt
        Integration step in seconds.
    window
        Truncation length in steps (``>= 1``).

    Returns
    -------
    JaxArray
        State trajectory of shape ``(6, N, T)`` (steps ``1 .. T``).
    """

    def body(
        carry: tuple[JaxArray, JaxArray, JaxArray], u_node: JaxArray
    ) -> tuple[tuple[JaxArray, JaxArray, JaxArray], JaxArray]:
        x, history, k = carry
        s_y = sigmoid_jax(lfp_jax(x), p.e0, p.v0, p.r)
        history = history.at[k % p.max_history_len].set(s_y)
        coupling = coupling_from_history_jax(history, k, p)
        x_next = heun_step_det_jax(x, u_node, coupling, p, dt)
        boundary = ((k + 1) % window) == 0
        # jnp.where routes the gradient through the stop_gradient branch at boundaries
        # (value is unchanged, gradient is cut) -- truncated backprop through time.
        x_next = jnp.where(boundary, jax.lax.stop_gradient(x_next), x_next)
        history = jnp.where(boundary, jax.lax.stop_gradient(history), history)
        return (x_next, history, k + 1), x_next

    s0 = sigmoid_jax(lfp_jax(x0), p.e0, p.v0, p.r)
    history0 = jnp.broadcast_to(s0, (p.max_history_len, p.n_nodes))
    init = (x0, history0, jnp.array(0, dtype=jnp.int64))
    _, x_seq = jax.lax.scan(body, init, controls_node)
    return jnp.transpose(x_seq, (1, 2, 0))


def model_eeg(p: JRParamsJax, x0: JaxArray, controls: JaxArray, dt: float, window: int) -> JaxArray:
    """Deterministic model EEG: a TBPTT rollout projected through ``eeg_gain``, shape ``(n_ch, T)``."""
    return eeg_jax(rollout_tbptt(x0, controls, p, dt, window), p.eeg_gain)


@dataclass(frozen=True)
class RefineConfig:
    """Optax refinement settings."""

    steps: int = 300
    lr: float | optax.Schedule = 1e-2
    clip: float = 1.0


@dataclass(frozen=True)
class PredConfig:
    """N-step-ahead MSE loss settings, scored over windows of one continuous rollout.

    The model state evolves continuously and naturally from a single ``x0`` across the
    whole recording (real driving controls, no resets); horizon-length windows are then
    sliced from the post-``burn_in`` trajectory purely to organise the MSE by lead time,
    not to reinitialise state.

    Attributes
    ----------
    horizon
        Window length (steps) the MSE is organised into.
    burn_in
        Leading post-``x0`` samples discarded once, globally, before windowing (skips the
        ``x0``-to-limit-cycle transient); applied identically to data and model.
    stride
        Step spacing between consecutive window starts. ``None`` defaults to ``horizon``
        (non-overlapping windows).
    """

    horizon: int
    burn_in: int = 0
    stride: int | None = None


def _window_starts(n_steps: int, horizon: int, stride: int) -> IntArray:
    """Compute valid window-start indices ``t0`` s.t. ``[t0, t0 + horizon)`` fits."""
    if n_steps < horizon:
        return np.zeros(0, dtype=np.int64)
    last_start = n_steps - horizon
    return np.arange(0, last_start + 1, stride, dtype=np.int64)


@dataclass(frozen=True)
class _RecordingWindows:
    """One recording's continuous-rollout inputs plus its precomputed target windows."""

    x0: JaxArray  # (6, N)
    controls: JaxArray  # (T, N)
    targets: JaxArray  # (n_windows, n_channels, horizon)
    starts: IntArray  # (n_windows,) -- window starts within the post-burn_in trajectory


def _build_recording_windows(  # noqa: PLR0913
    y_data: FloatArray, controls: FloatArray, x0: JaxArray, burn_in: int, horizon: int, stride: int
) -> _RecordingWindows:
    """Slice one recording's post-``burn_in`` EEG into windows for the loss (no resets).

    Parameters
    ----------
    y_data
        Measured EEG, shape ``(n_channels, T)``.
    controls
        Node-projected control schedule, shape ``(T, N)``, fed to one continuous rollout.
    x0
        Initial state for the (single, continuous) rollout, shape ``(6, N)``.
    burn_in, horizon, stride
        See :class:`PredConfig`.

    Returns
    -------
    _RecordingWindows
        Empty ``targets``/``starts`` if the recording is too short for a single window.
    """
    y_post = y_data[:, burn_in:]
    starts = _window_starts(y_post.shape[1], horizon, stride)
    targets = np.stack([y_post[:, t0 : t0 + horizon] for t0 in starts], axis=0) if starts.size else np.zeros((0,))
    return _RecordingWindows(
        x0=x0,
        controls=jnp.asarray(controls, dtype=jnp.float64),
        targets=jnp.asarray(targets, dtype=jnp.float64),
        starts=starts,
    )


def _channel_scale(targets: JaxArray) -> tuple[JaxArray, JaxArray]:
    """Per-channel mean/std over all windows and horizon steps, shape ``(1, n_channels, 1)``.

    EEG channel units are arbitrary/uncalibrated (see :mod:`neuro.prediction`'s
    scale-invariant metrics note), so the loss z-scores both prediction and target with
    these *fixed* (data-only, non-``p``-dependent) constants rather than comparing raw
    amplitudes directly.
    """
    flat = jnp.moveaxis(targets, 1, 0).reshape(targets.shape[1], -1)  # (n_channels, n_windows*horizon)
    mean = jnp.mean(flat, axis=1)[None, :, None]
    std = jnp.std(flat, axis=1)[None, :, None] + 1e-12
    return mean, std


def make_pred_loss(
    recordings: Sequence[_RecordingWindows], dt: float, window: int, burn_in: int
) -> Callable[[JRParamsJax], JaxArray]:
    """Build the windowed N-step-ahead MSE loss over one continuous rollout per recording.

    Each recording's state evolves continuously and naturally from its own ``x0`` (no
    per-window resets); the post-``burn_in`` trajectory is sliced into ``horizon``-length
    windows purely to organise the (per-channel-normalised) MSE.

    Parameters
    ----------
    recordings
        Per-recording rollout inputs and precomputed target windows (see
        :func:`_build_recording_windows`).
    dt
        Integration step in seconds.
    window
        Truncated-backprop window length passed to :func:`rollout_tbptt`.
    burn_in
        Leading samples discarded from the model rollout before windowing; must match
        the value used to build ``recordings``.

    Returns
    -------
    Callable
        ``loss(p: JRParamsJax) -> scalar`` -- jit/grad-friendly.
    """
    chan_mean, chan_std = _channel_scale(jnp.concatenate([r.targets for r in recordings], axis=0))

    def _recording_loss(p: JRParamsJax, rec: _RecordingWindows) -> JaxArray:
        eeg = model_eeg(p, rec.x0, rec.controls, dt, window)[:, burn_in:]
        horizon = rec.targets.shape[-1]
        windows = jnp.stack([eeg[:, t0 : t0 + horizon] for t0 in rec.starts], axis=0)
        eeg_scaled = (windows - chan_mean) / chan_std
        target_scaled = (rec.targets - chan_mean) / chan_std
        return jnp.mean((eeg_scaled - target_scaled) ** 2)

    def loss(p: JRParamsJax) -> JaxArray:
        per_recording = jnp.stack([_recording_loss(p, rec) for rec in recordings])
        return jnp.mean(per_recording)

    return loss


# --------------------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------------------
def refine(
    loss_fn: Callable[[dict[str, JaxArray]], JaxArray], theta0: dict[str, JaxArray], cfg: RefineConfig
) -> tuple[dict[str, JaxArray], list[float]]:
    """Refine ``theta`` with gradient-clipped Adam.

    Parameters
    ----------
    loss_fn
        ``loss(theta) -> scalar`` over the raw optimisation variables.
    theta0
        Initial raw variables (from :func:`pack_theta`).
    cfg
        Optimiser settings (``lr`` may be a constant or an :class:`optax.Schedule`).

    Returns
    -------
    theta
        Optimised raw variables.
    history
        Loss value after each step.
    """
    opt = optax.chain(optax.clip_by_global_norm(cfg.clip), optax.adam(cfg.lr))
    state = opt.init(theta0)

    @jax.jit
    def step(theta: dict[str, JaxArray], state: optax.OptState) -> tuple[dict[str, JaxArray], optax.OptState, JaxArray]:
        value, grad = jax.value_and_grad(loss_fn)(theta)
        updates, state = opt.update(grad, state, theta)
        theta_next = cast("dict[str, JaxArray]", optax.apply_updates(theta, updates))
        return theta_next, state, value

    theta = theta0
    history: list[float] = []
    pbar = tqdm(range(cfg.steps), desc="Refining")
    for _ in pbar:
        theta, state, value = step(theta, state)
        val = float(value)
        history.append(val)
        pbar.set_postfix(loss=f"{val:.4f}")
    return theta, history


@dataclass(frozen=True)
class IdentifyResult:
    """Result of an :func:`identify` run."""

    params: JRParamsJax
    history: list[float]


def identify(  # noqa: PLR0913
    base: JRParamsJax,
    free_names: Sequence[str],
    y_data: FloatArray | list[FloatArray],
    dt: float,
    *,
    pred_cfg: PredConfig,
    window: int = 200,
    controls: FloatArray | list[FloatArray] | None = None,
    x0: FloatArray | None = None,
    w_max: float | None = None,
    refine_cfg: RefineConfig | None = None,
) -> IdentifyResult:
    """Gradient-refine reduced-model parameters against a windowed N-step-ahead MSE.

    Each recording's state evolves continuously and naturally from ``x0`` across the
    whole rollout (real driving ``controls``, no resets, see :func:`make_pred_loss`); the
    post-``pred_cfg.burn_in`` trajectory is sliced into ``pred_cfg.horizon``-length
    windows purely to organise the MSE. Multiple recordings each contribute their own
    rollout and windows into one combined loss (no statistical averaging).

    Parameters
    ----------
    base
        Base reduced-model parameters (fixed ``delay_steps``, ``sigma``...).
    free_names
        Subset of ``{"A", "w_weights", "eeg_gain", "K", "mean_input"}`` to identify.
    y_data
        Measured EEG, shape ``(n_channels, T)``, or a list of such recordings.
    dt
        Integration step in seconds.
    pred_cfg
        Horizon/burn-in/stride settings (see :class:`PredConfig`).
    window
        Truncated-backprop window length passed to :func:`rollout_tbptt`.
    controls
        Node-projected control schedule(s), shape ``(T, N)`` (or a list aligned with a
        list ``y_data``); defaults to zeros per recording. Pointwise MSE is only
        meaningful if this is a real, persistently-exciting driving input -- with zero
        controls the model's free-running phase will not track ``y_data``.
    x0
        Initial state shared by every recording's rollout, shape ``(6, N)``; defaults to
        zeros.
    w_max
        Upper bound per connection weight; defaults to ``10 x`` the max base weight (or 1).
    refine_cfg
        Optimiser settings (sensible defaults used).

    Returns
    -------
    IdentifyResult
        Identified parameters and the loss history.
    """
    refine_cfg = refine_cfg or RefineConfig()
    n = base.n_nodes
    y_list = cast("list[FloatArray]", y_data if isinstance(y_data, list) else [y_data])
    n_recordings = len(y_list)

    x0_arr = jnp.zeros((6, n), dtype=jnp.float64) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)

    if controls is None:
        ctrl_list = [np.zeros((y.shape[1], n), dtype=np.float64) for y in y_list]
    elif isinstance(controls, list):
        ctrl_list = [np.asarray(c, dtype=np.float64) for c in controls]
    else:
        if n_recordings != 1:
            msg = "controls must be a list aligned with y_data when y_data is a list"
            raise TypeError(msg)
        ctrl_list = [np.asarray(controls, dtype=np.float64)]

    if len(ctrl_list) != n_recordings:
        msg = f"controls has {len(ctrl_list)} recording(s) but y_data has {n_recordings}"
        raise ValueError(msg)

    stride = pred_cfg.stride if pred_cfg.stride is not None else pred_cfg.horizon
    recordings = [
        _build_recording_windows(y, c, x0_arr, pred_cfg.burn_in, pred_cfg.horizon, stride)
        for y, c in zip(y_list, ctrl_list, strict=True)
    ]
    recordings = [r for r in recordings if r.targets.shape[0] > 0]
    if not recordings:
        msg = "no recording is long enough to produce a single window; reduce burn_in/horizon"
        raise ValueError(msg)

    if w_max is None:
        w_max = 10.0 * (float(jnp.max(jnp.abs(base.w_weights))) or 1.0)
    bounds = make_bounds(w_max)

    pred_loss = make_pred_loss(recordings, dt, window, pred_cfg.burn_in)

    theta0 = pack_theta(base, free_names, bounds)

    def loss_theta(theta: dict[str, JaxArray]) -> JaxArray:
        return pred_loss(build_params(theta, base, free_names, bounds))

    theta, history = refine(loss_theta, theta0, refine_cfg)
    params = build_params(theta, base, free_names, bounds)
    return IdentifyResult(params=params, history=history)
