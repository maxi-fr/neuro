"""Pure-JAX system identification for a reduced Jansen-Rit network.

Exploration + gradient refinement, an autodiff alternative to the CasADi PCMS
solver in :mod:`neuro.sysid`. This is the workflow of Pille et al. 2025 (TVB-Optim):
a coarse parameter-space *exploration* over a few global knobs to pick a good
regime, followed by gradient *refinement* with Optax.

Scope (this phase): offline, **uncontrolled** (``u = 0``) identification of a
reduced ``N``-node model whose LFP ``x2 - x3`` is projected to the 62 EEG channels
by a *fixed* coarse leadfield ``eeg_gain`` (built by ``notebooks/coarse_graining.py``).

Identified parameters
---------------------
``A`` (length ``N``), ``w_weights`` (``N x N``, kept **non-negative and symmetric**
with zero diagonal, like :mod:`neuro.sysid`), and optionally the global scalars
``K`` and ``mean_input``. Bounds are enforced *differentiably* via a bounded-sigmoid
/ softplus reparametrisation rather than hard clamps, mirroring the paper's ``|w|``
trick.

Robustness choices (see plan / paper section 9.5)
-------------------------------------------------
Jansen-Rit turns chaotic at the oscillatory bifurcation -- exactly where the best
EEG match sits -- so reverse-mode AD through a long rollout explodes. We therefore:

* **truncate backprop** to short windows (``window`` steps) via ``stop_gradient`` on
  the scan carry at each window boundary (the JAX analogue of a small PCMS ``D``);
* make the loss **statistics-dominant** -- a phase-blind spectral *shape* term
  (paper Eq. 12) plus a spatial channel-correlation (FC-like) term, both invariant
  to the constant per-channel leadfield scale -- with the time-domain term a light,
  optional add-on (used mainly for self-consistency checks where the initial state
  is known).

64-bit precision is mandatory; call :func:`neuro.jansen_rit_jax.enable_x64` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from sklearn.decomposition import PCA

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

# Differentiable bounds for the free parameters, mirroring the ``bounds`` metadata on
# :class:`neuro.jansen_rit.JansenRitParams`. ``w_weights`` is bounded separately by a
# caller-provided ``w_max`` (cf. ``neuro.sysid.SysIDSolver._W_UB_FACTOR``).
_BOUNDS: dict[str, tuple[float, float]] = {
    "A": (0.0, 10.0),
    "mean_input": (0.0, 500.0),
    "K": (0.0, 10.0),
    "eeg_gain": (0.0, 10.0),
}
_EPS = 1e-30


# --------------------------------------------------------------------------------------
# Reduced-model construction helpers
# --------------------------------------------------------------------------------------
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


def pack_theta(base: JRParamsJax, free_names: Sequence[str], *, w_max: float) -> dict[str, JaxArray]:
    """Build the unconstrained optimisation variables from a base parameter set.

    Parameters
    ----------
    base
        Base reduced-model parameters; free entries seed the initial guess.
    free_names
        Subset of ``{"A", "w_weights", "K", "mean_input"}`` to identify.
    w_max
        Upper bound for each (non-negative) connection weight.

    Returns
    -------
    dict
        Raw (unconstrained) leaves keyed by parameter name; feed to :func:`build_params`.
    """
    n = base.n_nodes
    theta: dict[str, JaxArray] = {}
    for name in free_names:
        if name == "w_weights":
            iu = np.triu_indices(n, k=1)
            tri = np.asarray(base.w_weights, dtype=np.float64)[iu]
            theta[name] = jnp.asarray(_to_raw(tri, 0.0, w_max))
        elif name in _BOUNDS:
            low, high = _BOUNDS[name]
            theta[name] = jnp.asarray(_to_raw(np.asarray(getattr(base, name)), low, high))
        else:
            msg = f"Unknown or unsupported free parameter {name!r}"
            raise ValueError(msg)
    return theta


def build_params(
    theta: dict[str, JaxArray], base: JRParamsJax, free_names: Sequence[str], *, w_max: float
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
    w_max
        Upper bound for each connection weight.

    Returns
    -------
    JRParamsJax
        Parameters with the free entries mapped back into their valid ranges.
    """
    updates: dict[str, JaxArray] = {}
    for name in free_names:
        if name == "w_weights":
            tri = _to_constrained(theta[name], 0.0, w_max)
            updates[name] = _w_from_tri(tri, base.n_nodes)
        else:
            low, high = _BOUNDS[name]
            updates[name] = _to_constrained(theta[name], low, high)
    return replace(base, **updates)


# --------------------------------------------------------------------------------------
# Truncated-backprop forward rollout (deterministic, differentiable)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Loss configuration and statistics targets
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StatConfig:
    """Welch PSD settings and spectral comparison mode.

    Attributes
    ----------
    nperseg
        Welch segment length. ``None`` sizes it for ~1 Hz resolution
        (``min(T, round(fs))``), like :func:`neuro.utils.processing.compute_psd`: at the
        sim's high ``fs`` a fixed small ``nperseg`` widens the bins past the EEG rhythm
        and collapses it into DC (the psd-nperseg-resolution gotcha).
    noverlap
        Segment overlap; ``None`` uses ``nperseg // 2``.
    band
        Inclusive frequency band (Hz) the loss is computed over.
    spec_mode
        ``"magnitude"`` compares log-PSD directly (amplitude-sensitive -- keeps the
        amplitude ``A`` controls; best when the reduced ``eeg_gain`` scale is trusted).
        ``"shape"`` compares the per-channel spectral *shape* via correlation
        (paper Eq. 12), invariant to a constant per-channel scale -- use it when the
        reduced gain's scale differs from the plant's, at the cost of ``A``-identifiability.
    """

    nperseg: int | None = None
    noverlap: int | None = None
    band: tuple[float, float] = (1.0, 45.0)
    spec_mode: str = "magnitude"


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the loss terms (statistics-dominant by default)."""

    spec: float = 1.0
    spatial: float = 0.5
    time: float = 0.0


@dataclass(frozen=True)
class RefineConfig:
    """Optax refinement settings."""

    steps: int = 300
    lr: float = 1e-2
    clip: float = 1.0


@dataclass(frozen=True)
class _Targets:
    """Precomputed data-side statistics the model is fit against."""

    data_logpsd: JaxArray
    corr_data: JaxArray
    data_scaled: JaxArray
    chan_mean: JaxArray
    chan_std: JaxArray
    nperseg: int
    noverlap: int
    fs: float
    spec_mode: str = "magnitude"
    bin_idx: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _welch_settings(n_steps: int, dt: float, cfg: StatConfig) -> tuple[int, int, float, IntArray]:
    fs_full = 1.0 / dt
    requested = cfg.nperseg if cfg.nperseg is not None else round(fs_full)  # ~1 Hz resolution
    nperseg = min(requested, n_steps)
    noverlap = cfg.noverlap if cfg.noverlap is not None else nperseg // 2
    noverlap = min(noverlap, nperseg - 1)
    freqs = np.fft.rfftfreq(nperseg, d=dt)
    bin_idx = np.where((freqs >= cfg.band[0]) & (freqs <= cfg.band[1]))[0].astype(np.int64)
    if bin_idx.size == 0:
        bin_idx = np.arange(freqs.size, dtype=np.int64)
    return nperseg, noverlap, fs_full, bin_idx


def _log_psd(eeg: JaxArray, nperseg: int, noverlap: int, fs: float, bin_idx: IntArray) -> JaxArray:
    """One-sided log Welch PSD over the retained band bins, shape ``(n_ch, n_bins)``."""
    _, pxx = jax.scipy.signal.welch(eeg, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
    return jnp.log(pxx[:, bin_idx] + _EPS)


def _spec_mag_loss(model_logpsd: JaxArray, data_logpsd: JaxArray) -> JaxArray:
    """Phase-blind, amplitude-sensitive log-PSD MSE (keeps the amplitude ``A`` controls)."""
    return jnp.mean((model_logpsd - data_logpsd) ** 2)


def _spec_shape_loss(model_logpsd: JaxArray, data_logpsd: JaxArray) -> JaxArray:
    """Phase-blind, amplitude-invariant spectral-shape loss (paper Eq. 12).

    Mean over channels of ``1 - Pearson(log S_model, log S_data)``, in ``[0, 2]``.
    """
    a = model_logpsd - model_logpsd.mean(axis=1, keepdims=True)
    b = data_logpsd - data_logpsd.mean(axis=1, keepdims=True)
    num = jnp.sum(a * b, axis=1)
    den = jnp.sqrt(jnp.sum(a * a, axis=1) * jnp.sum(b * b, axis=1)) + 1e-12
    return jnp.mean(1.0 - num / den)


def _chan_corr(eeg: JaxArray) -> JaxArray:
    """Channel-by-channel correlation matrix (FC-like), shape ``(n_ch, n_ch)``."""
    e = eeg - eeg.mean(axis=1, keepdims=True)
    cov = e @ e.T
    d = jnp.sqrt(jnp.diag(cov)) + 1e-12
    return cov / jnp.outer(d, d)


def _spatial_loss(eeg: JaxArray, corr_data: JaxArray) -> JaxArray:
    """Mean squared error of the off-diagonal channel-correlation entries."""
    cm = _chan_corr(eeg)
    iu = jnp.triu_indices(cm.shape[0], k=1)
    return jnp.mean((cm[iu] - corr_data[iu]) ** 2)


def build_targets(y_data: FloatArray, dt: float, cfg: StatConfig, burn_in: int = 0) -> _Targets:
    """Precompute the data-side statistics (log-PSD shape, channel FC, scaled signal).

    Parameters
    ----------
    y_data
        Measured EEG, shape ``(n_channels, T)``.
    dt
        Integration step in seconds.
    cfg
        Welch settings.
    burn_in
        Number of leading samples to discard before computing statistics, so a startup
        transient does not contaminate the (low-frequency) PSD bins. Must match the value
        passed to :func:`make_stat_loss`.

    Returns
    -------
    _Targets
        Constants closed over by the loss.
    """
    y = jnp.asarray(y_data, dtype=jnp.float64)[:, burn_in:]
    nperseg, noverlap, fs, bin_idx = _welch_settings(y.shape[1], dt, cfg)
    chan_mean = jnp.mean(y, axis=1, keepdims=True)
    chan_std = jnp.std(y, axis=1, keepdims=True) + 1e-12
    return _Targets(
        data_logpsd=_log_psd(y, nperseg, noverlap, fs, bin_idx),
        corr_data=_chan_corr(y),
        data_scaled=(y - chan_mean) / chan_std,
        chan_mean=chan_mean,
        chan_std=chan_std,
        nperseg=nperseg,
        noverlap=noverlap,
        fs=fs,
        spec_mode=cfg.spec_mode,
        bin_idx=bin_idx,
    )


def average_targets(targets_list: Sequence[_Targets]) -> _Targets:
    """Average the data-side statistics over several recordings of the same regime.

    For a deterministic, uncontrolled fit the rollout is identical across noise seeds, so
    minimising the summed per-seed loss equals (up to a parameter-independent constant)
    fitting the *mean* PSD / FC target -- one rollout, one averaged target. All recordings
    must share the Welch sizing, i.e. have equal length (and thus identical ``bin_idx``).

    Parameters
    ----------
    targets_list
        Per-recording outputs of :func:`build_targets` (all same length).

    Returns
    -------
    _Targets
        A target whose ``data_logpsd``, ``corr_data`` and ``data_scaled`` are the
        across-recording means; Welch settings are taken from the first recording.
    """
    first = targets_list[0]
    if any(t.data_logpsd.shape != first.data_logpsd.shape for t in targets_list):
        msg = "average_targets requires equal-length recordings (matching Welch bins)."
        raise ValueError(msg)

    def _avg(get: Callable[[_Targets], JaxArray]) -> JaxArray:
        return jnp.mean(jnp.stack([get(t) for t in targets_list], axis=0), axis=0)

    return replace(
        first,
        data_logpsd=_avg(lambda t: t.data_logpsd),
        corr_data=_avg(lambda t: t.corr_data),
        data_scaled=_avg(lambda t: t.data_scaled),
    )


def make_stat_loss(  # noqa: PLR0913
    targets: _Targets, weights: LossWeights, x0: JaxArray, controls: JaxArray, dt: float, window: int, burn_in: int = 0
) -> Callable[[JRParamsJax], JaxArray]:
    """Build ``loss(p)`` comparing a model rollout's statistics to ``targets``.

    Parameters
    ----------
    targets
        Output of :func:`build_targets`.
    weights
        Term weights (spectral shape, spatial FC, optional time MSE).
    x0
        Initial state, shape ``(6, N)``.
    controls
        Node-projected control schedule, shape ``(T, N)`` (zeros = uncontrolled).
    dt
        Integration step in seconds.
    window
        Truncated-backprop window length in steps.
    burn_in
        Number of leading rollout samples to discard before computing statistics (the
        model ramps from ``x0`` into its limit cycle); must match :func:`build_targets`.

    Returns
    -------
    Callable
        ``loss(p: JRParamsJax) -> scalar`` -- jit/grad-friendly.
    """

    def loss(p: JRParamsJax) -> JaxArray:
        eeg = eeg_jax(rollout_tbptt(x0, controls, p, dt, window), p.eeg_gain)[:, burn_in:]
        total = jnp.asarray(0.0, dtype=jnp.float64)
        if weights.spec:
            model_logpsd = _log_psd(eeg, targets.nperseg, targets.noverlap, targets.fs, targets.bin_idx)
            spec_term = (
                _spec_shape_loss(model_logpsd, targets.data_logpsd)
                if targets.spec_mode == "shape"
                else _spec_mag_loss(model_logpsd, targets.data_logpsd)
            )
            total = total + weights.spec * spec_term
        if weights.spatial:
            total = total + weights.spatial * _spatial_loss(eeg, targets.corr_data)
        if weights.time:
            scaled = (eeg - targets.chan_mean) / targets.chan_std
            total = total + weights.time * jnp.mean((scaled - targets.data_scaled) ** 2)
        return total

    return loss


# --------------------------------------------------------------------------------------
# Exploration and refinement
# --------------------------------------------------------------------------------------
def explore_globals(
    base: JRParamsJax,
    stat_loss: Callable[[JRParamsJax], JaxArray],
    *,
    a_grid: Sequence[float] | None = None,
    k_grid: Sequence[float] | None = None,
    input_grid: Sequence[float] | None = None,
) -> tuple[dict[str, float], FloatArray]:
    """Grid-search the global knobs ``A``, ``K`` and ``mean_input`` for a good init.

    Evaluates ``stat_loss`` on the Cartesian product of the grids and returns the argmin
    as override values plus the full loss grid. ``A`` is searched as a single global scalar
    (broadcast to all nodes): the limit-cycle PSD-vs-``A`` landscape is flat/multimodal, so
    in-basin gradient on ``A`` crawls -- exploration is what localises it (Pille et al.).
    The grid is walked sequentially with :func:`jax.lax.map` (not ``vmap``) so a large grid
    over a long trajectory does not materialise every rollout at once.

    Parameters
    ----------
    base
        Base parameters (its ``A`` / ``K`` / ``mean_input`` are the defaults when a grid
        is ``None``).
    stat_loss
        A statistics loss ``loss(p)`` from :func:`make_stat_loss`.
    a_grid, k_grid, input_grid
        Candidate values for the global excitatory gain ``A``, coupling ``K`` and the
        (uniform) mean input.

    Returns
    -------
    overrides
        ``{"A": ..., "K": ..., "mean_input": ...}`` minimising the loss.
    losses
        Loss for every ``(A, K, mean_input)`` combination, shape ``(len(a)*len(k)*len(i),)``.
    """
    ag = list(a_grid) if a_grid is not None else [float(jnp.mean(base.A))]
    kg = list(k_grid) if k_grid is not None else [float(base.K)]
    ig = list(input_grid) if input_grid is not None else [float(jnp.mean(base.mean_input))]
    combos = jnp.asarray([[a, k, i] for a in ag for k in kg for i in ig], dtype=jnp.float64)

    def one(aki: JaxArray) -> JaxArray:
        p = replace(
            base,
            A=jnp.broadcast_to(aki[0], (base.n_nodes,)),
            K=aki[1],
            mean_input=jnp.broadcast_to(aki[2], (base.n_nodes,)),
        )
        return stat_loss(p)

    losses = jax.lax.map(one, combos)
    best = combos[int(jnp.argmin(losses))]
    return {"A": float(best[0]), "K": float(best[1]), "mean_input": float(best[2])}, np.asarray(losses)


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
        Optimiser settings.

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
    for _ in range(cfg.steps):
        theta, state, value = step(theta, state)
        history.append(float(value))
    return theta, history


@dataclass(frozen=True)
class IdentifyResult:
    """Result of an :func:`identify` run."""

    params: JRParamsJax
    theta: dict[str, JaxArray]
    history: list[float]
    explore_overrides: dict[str, float]
    explore_losses: FloatArray


def identify(  # noqa: PLR0913
    base: JRParamsJax,
    free_names: Sequence[str],
    y_data: FloatArray | list[FloatArray],
    dt: float,
    *,
    window: int = 200,
    burn_in: int = 0,
    controls: FloatArray | None = None,
    x0: FloatArray | None = None,
    w_max: float | None = None,
    weights: LossWeights | None = None,
    stat_cfg: StatConfig | None = None,
    refine_cfg: RefineConfig | None = None,
    a_grid: Sequence[float] | None = None,
    k_grid: Sequence[float] | None = None,
    input_grid: Sequence[float] | None = None,
) -> IdentifyResult:
    """Explore global knobs then gradient-refine the reduced-model parameters.

    Parameters
    ----------
    base
        Base reduced-model parameters (fixed ``eeg_gain``, ``delay_steps``, ``sigma``...).
    free_names
        Subset of ``{"A", "w_weights", "K", "mean_input"}`` to identify.
    y_data
        Measured (uncontrolled) EEG, shape ``(n_channels, T)``, or a sequence of such
        recordings of the *same regime* (different noise seeds). A sequence is averaged
        into a single mean PSD / FC target via :func:`average_targets` (all recordings
        must be the same length); set the time-domain loss weight to 0 in that case, as
        the deterministic model cannot match any single realisation's phase.
    dt
        Integration step in seconds.
    window
        Truncated-backprop window length in steps.
    burn_in
        Leading samples discarded before computing statistics (drops the model's
        ``x0``-to-limit-cycle transient); applied identically to data and model.
    controls
        Optional node-projected control schedule, shape ``(T, N)``; defaults to zeros.
    x0
        Optional initial state, shape ``(6, N)``; defaults to zeros.
    w_max
        Upper bound per connection weight; defaults to ``10 x`` the max base weight (or 1).
    weights, stat_cfg, refine_cfg
        Loss weights, Welch settings, and optimiser settings (sensible defaults used).
    a_grid, k_grid, input_grid
        Optional exploration grids for the global ``A``, ``K`` and ``mean_input``; if any
        is given, the best combination is used as the refinement init.

    Returns
    -------
    IdentifyResult
        Identified parameters, raw variables, loss history, and exploration outcome.
    """
    weights = weights or LossWeights()
    stat_cfg = stat_cfg or StatConfig()
    refine_cfg = refine_cfg or RefineConfig()
    n = base.n_nodes
    y_list = cast("list[FloatArray]", y_data if isinstance(y_data, list) else [y_data])
    n_steps = int(y_list[0].shape[1])

    x0_arr = jnp.zeros((6, n), dtype=jnp.float64) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)
    u_arr = jnp.zeros((n_steps, n), dtype=jnp.float64) if controls is None else jnp.asarray(controls, dtype=jnp.float64)
    if w_max is None:
        w_max = 10.0 * (float(jnp.max(jnp.abs(base.w_weights))) or 1.0)

    targets_list = [build_targets(y, dt, stat_cfg, burn_in) for y in y_list]
    targets = targets_list[0] if len(targets_list) == 1 else average_targets(targets_list)
    stat_loss = make_stat_loss(targets, weights, x0_arr, u_arr, dt, window, burn_in)

    overrides: dict[str, float] = {}
    losses = np.zeros(0, dtype=np.float64)
    if a_grid is not None or k_grid is not None or input_grid is not None:
        overrides, losses = explore_globals(base, stat_loss, a_grid=a_grid, k_grid=k_grid, input_grid=input_grid)
        base = replace(
            base,
            A=jnp.broadcast_to(jnp.asarray(overrides["A"]), (n,)),
            K=jnp.asarray(overrides["K"]),
            mean_input=jnp.broadcast_to(jnp.asarray(overrides["mean_input"]), (n,)),
        )

    theta0 = pack_theta(base, free_names, w_max=w_max)

    def loss_theta(theta: dict[str, JaxArray]) -> JaxArray:
        return stat_loss(build_params(theta, base, free_names, w_max=w_max))

    theta, history = refine(loss_theta, theta0, refine_cfg)
    params = build_params(theta, base, free_names, w_max=w_max)
    return IdentifyResult(
        params=params, theta=theta, history=history, explore_overrides=overrides, explore_losses=losses
    )
