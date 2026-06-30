"""Unified short-horizon prediction API for comparing system-identification methods.

Two identified models live in this project -- the JAX reduced PCA model
(:mod:`neuro.sysid_jax`) and an Equinox MLP predictor (``scripts/run_nn_predictor_jax.py``).
They are fit and evaluated in isolation on different data, channels, units, and time
steps. This module wraps each behind one :class:`Predictor` protocol so a notebook can
compare them on identical held-out test windows.

Conditioning is **measured-EEG only** (no hidden plant state): at each test start ``t0``
every predictor rebuilds its state from a window of recent *measured* EEG, then runs
open-loop to the horizon, applying the recorded future controls through the physical
tES projection ``gamma`` (the data is a persistently-exciting tES recording). The two
reduced physics models pre-load their slowly-varying parameters from train-only
artifacts and only build the per-window *state* from the test context.

Metrics are **scale-invariant** (NRMSE, Pearson correlation, correlation-based FC)
because the EEG units are arbitrary (see the project's uncalibrated-units note) and the
methods output different amplitude scales.

64-bit JAX is required; call :func:`neuro.jansen_rit_jax.enable_x64` first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import eeg_jax, from_jansen_rit_params
from neuro.sysid_jax import rollout_tbptt

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.jansen_rit_jax import JRParamsJax

FloatArray = npt.NDArray[np.float64]
JaxArray = jax.Array

_N_STATE_ROWS = 6  # Jansen-Rit state rows x1..x6
_EPS = 1e-12


def get_activation(name: str) -> Callable[[JaxArray], JaxArray]:
    """Return the JAX activation function by name."""
    if name == "relu":
        return jax.nn.relu
    if name == "tanh":
        return jax.nn.tanh
    if name == "softplus":
        return jax.nn.softplus
    msg = f"Unsupported activation: {name}"
    raise ValueError(msg)


# --------------------------------------------------------------------------------------
# Window container
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionWindow:
    """Measurements a predictor may use at one test start ``t0``.

    No hidden plant state is exposed -- each predictor reconstructs its own state from
    ``y_ctx``. All arrays are given at the data sampling step ``dt_data``; predictors
    downsample to their own native step internally.

    Attributes
    ----------
    y_ctx
        Measured EEG over the context window ending at ``t0``, shape ``(n_ch, ctx)``.
    u_ctx
        Per-electrode controls over the context, shape ``(ctx, n_elec)``.
    u_future
        Recorded/planned controls over the horizon, shape ``(H, n_elec)``.
    dt_data
        Sampling step (seconds) of ``y_ctx`` / ``u_ctx`` / ``u_future``.
    """

    y_ctx: FloatArray
    u_ctx: FloatArray
    u_future: FloatArray
    dt_data: float


@runtime_checkable
class Predictor(Protocol):
    """Common interface: build state from a context window, predict open-loop."""

    name: str
    dt: float

    def predict(self, window: PredictionWindow, horizon_s: float) -> FloatArray:
        """Return predicted EEG ``(n_ch, n_steps)`` at ``self.dt`` over ``horizon_s``."""
        ...


# --------------------------------------------------------------------------------------
# Resampling helpers
# --------------------------------------------------------------------------------------
def _ds_factor(target_dt: float, data_dt: float) -> int:
    """Integer decimation factor mapping ``data_dt`` up to ``target_dt`` (>= 1)."""
    return max(1, round(target_dt / data_dt))


def _resample_controls(u: FloatArray, factor: int, n_steps: int) -> FloatArray:
    """Decimate a ``(T, n_elec)`` control series and pad/truncate to ``n_steps`` rows."""
    u_ds = np.asarray(u, dtype=np.float64)[::factor]
    if u_ds.shape[0] >= n_steps:
        return u_ds[:n_steps]
    pad = np.zeros((n_steps - u_ds.shape[0], u_ds.shape[1]), dtype=np.float64)
    return np.concatenate([u_ds, pad], axis=0)


def decimate_to(signal: FloatArray, native_dt: float, analysis_dt: float) -> FloatArray:
    """Decimate a ``(n_ch, T)`` signal from ``native_dt`` onto the ``analysis_dt`` grid.

    Parameters
    ----------
    signal : FloatArray
        Input signal array, shape ``(n_channels, n_samples)``.
    native_dt : float
        The original sampling interval of the signal.
    analysis_dt : float
        The target sampling interval to decimate to.

    Returns
    -------
    FloatArray
        The decimated signal array, shape ``(n_channels, n_samples_decimated)``.
    """
    return np.asarray(signal, dtype=np.float64)[:, :: _ds_factor(analysis_dt, native_dt)]


class JaxSysidPredictor:
    """Reduced PCA Jansen-Rit model identified by :mod:`neuro.sysid_jax`.

    The model does **not** fit per window. Instead it is run forward through the context
    window (from a zero state, driven by the recorded controls) so its internal state
    settles past the ``x0`` -> limit-cycle transient; the horizon prediction is the
    continued deterministic rollout. Being deterministic and noise-free, it tracks the
    model's own attractor under the recorded stimulation rather than a specific noise
    realisation -- a stabilised free-run, not data assimilation.
    """

    name = "jax_sysid"

    def __init__(
        self,
        params: JRParamsJax,
        eeg_gain: FloatArray,
        gamma_modes: FloatArray,
        dt: float,
    ) -> None:
        self._params = params
        self._eeg_gain = jnp.asarray(eeg_gain, dtype=jnp.float64)
        self._gamma_modes = jnp.asarray(gamma_modes, dtype=jnp.float64)
        self.dt = float(dt)
        self._n_nodes = params.n_nodes

    @classmethod
    def load(cls, artifact: str | Path, gamma_full: FloatArray, **_kwargs: object) -> JaxSysidPredictor:
        """Rebuild the predictor from a ``run_sysid_jax`` ``.npz`` and the full ``gamma``.

        ``gamma_full`` is the ``(n_elec, R)`` physical tES projection over the full
        connectome; it is projected into the saved PCA mode basis so the reduced model
        can respond to the recorded stimulation.
        """
        with np.load(artifact) as res:
            a_syn = np.asarray(res["A"], dtype=np.float64)
            w_weights = np.asarray(res["w_weights"], dtype=np.float64)
            k_global = float(res["K"])
            mean_input = np.asarray(res["mean_input"], dtype=np.float64)
            eeg_gain = np.asarray(res["eeg_gain"], dtype=np.float64)
            components = np.asarray(res["components"], dtype=np.float64)
            delay_steps = np.asarray(res["delay_steps"], dtype=np.int64)
            dt = float(res["dt"]) if "dt" in res else 1e-3
        n = int(a_syn.shape[0])
        gamma_modes = np.atleast_2d(np.asarray(gamma_full, dtype=np.float64)) @ components.T
        jr = JansenRitParams(
            A=a_syn,
            mean_input=mean_input,
            sigma=0.0,
            w_weights=w_weights,
            delay_steps=delay_steps,
            eeg_gain=eeg_gain,
            gamma=gamma_modes,
            K=k_global,
        )
        params = from_jansen_rit_params(jr, n)
        return cls(params, eeg_gain, gamma_modes, dt)

    def predict(self, window: PredictionWindow, horizon_s: float) -> FloatArray:
        """Stabilise the state over the context, then continue the rollout for the horizon."""
        factor = _ds_factor(self.dt, window.dt_data)
        ctx_t = np.asarray(window.y_ctx)[:, ::factor].shape[1]
        u_ctx = _resample_controls(window.u_ctx, factor, ctx_t)
        n_steps = round(horizon_s / self.dt)
        u_fut = _resample_controls(window.u_future, factor, n_steps)

        u_all = jnp.asarray(np.concatenate([u_ctx, u_fut], axis=0), dtype=jnp.float64) @ self._gamma_modes
        x0 = jnp.zeros((_N_STATE_ROWS, self._n_nodes), dtype=jnp.float64)
        x_traj = rollout_tbptt(x0, u_all, self._params, self.dt, ctx_t + n_steps)
        eeg = np.asarray(eeg_jax(x_traj, self._eeg_gain))  # (n_ch, ctx_t + n_steps)
        return eeg[:, ctx_t:]


class AutoregressivePredictor(eqx.Module):
    """Wrapper that unrolls a 1-step MLP model over a prediction horizon.

    Attributes
    ----------
    model : eqx.nn.MLP
        The underlying 1-step MLP model.
    n_y : int
        The number of past EEG (output) steps the model needs as history.
    n_u : int
        The number of past control inputs the model needs as history.
    horizon : int
        The number of future steps to predict.
    C_y : int
        The number of EEG (output) channels.
    C_u : int
        The number of control (input) channels.
    """

    model: eqx.nn.MLP
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    C_y: int = eqx.field(static=True)
    C_u: int = eqx.field(static=True)
    activation: str = eqx.field(static=True, default="relu")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the autoregressive rollout."""
        y_past_flat = x[: self.n_y * self.C_y]
        u_past_flat = x[self.n_y * self.C_y : self.n_y * self.C_y + self.n_u * self.C_u]
        u_future_flat = x[self.n_y * self.C_y + self.n_u * self.C_u :]

        y_window = y_past_flat.reshape((self.n_y, self.C_y))
        u_window = u_past_flat.reshape((self.n_u, self.C_u))
        u_future = u_future_flat.reshape((self.horizon, self.C_u))

        def scan_fn(
            carry: tuple[jax.Array, jax.Array], u_curr: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            y_w, u_w = carry
            new_u_w = jnp.concatenate([u_w[1:], u_curr[None, :]], axis=0)
            mlp_input = jnp.concatenate([y_w.flatten(), new_u_w.flatten()])
            y_next = self.model(mlp_input)
            new_y_w = jnp.concatenate([y_w[1:], y_next[None, :]], axis=0)
            return (new_y_w, new_u_w), y_next

        _, y_preds = jax.lax.scan(scan_fn, (y_window, u_window), u_future)
        return y_preds.flatten()


@dataclass(frozen=True)
class MLPArtifact:
    """Loaded ``run_nn_predictor_jax`` artifact: the live predictor, native dt, and scalers.

    The canonical artifact representation, shared by the training script (:meth:`save`),
    :class:`NNPredictor` (:meth:`load`), and the CasADi port in
    :mod:`neuro.nn_predictor_casadi`. Architecture sizes (``n_y``, ``n_u``, ``horizon``,
    ``n_channels``, ``n_controls``) are not duplicated here -- they are read straight off
    ``model``, the single source of truth once it is built or deserialised.

    Attributes
    ----------
    model : AutoregressivePredictor
        The trained (or freshly built) predictor.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    u_mean, u_scale, y_mean, y_scale : FloatArray
        Per-channel ``StandardScaler`` statistics used to z-score controls/outputs.
    """

    model: AutoregressivePredictor
    dt: float
    downsample: int
    u_mean: FloatArray
    u_scale: FloatArray
    y_mean: FloatArray
    y_scale: FloatArray

    @property
    def n_y(self) -> int:
        """Number of past EEG (output) steps in the model's history window."""
        return self.model.n_y

    @property
    def n_u(self) -> int:
        """Number of past control (input) steps in the model's history window."""
        return self.model.n_u

    @property
    def horizon(self) -> int:
        """Direct-prediction horizon of the underlying model."""
        return self.model.horizon

    @property
    def n_channels(self) -> int:
        """Number of EEG output channels."""
        return self.model.C_y

    @property
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.model.C_u

    @classmethod
    def load(cls, artifact: str | Path) -> MLPArtifact:
        """Load the 3-file artifact (``.eqx``/``.json``/``.scalers.npz``) from disk."""
        artifact = Path(artifact)
        meta: dict[str, Any] = json.loads(artifact.with_suffix(".json").read_text())
        with np.load(artifact.with_suffix(".scalers.npz")) as sc:
            scalers = {k: np.asarray(sc[k], dtype=np.float64) for k in ("u_mean", "u_scale", "y_mean", "y_scale")}

        activation_name = meta.get("activation", "relu")
        mlp = eqx.nn.MLP(
            in_size=meta["in_size"],
            out_size=meta["out_size"],
            width_size=meta["hidden_size"],
            depth=meta["depth"],
            activation=get_activation(activation_name),
            key=jax.random.PRNGKey(0),
        )
        skeleton = AutoregressivePredictor(
            model=mlp,
            n_y=meta["n_y"],
            n_u=meta["n_u"],
            horizon=meta["horizon"],
            C_y=meta["n_channels"],
            C_u=meta["n_controls"],
            activation=activation_name,
        )
        model = eqx.tree_deserialise_leaves(str(artifact), skeleton)

        return cls(model=model, dt=float(meta["dt"]), downsample=int(meta["downsample"]), **scalers)

    def save(self, artifact: str | Path) -> None:
        """Persist the predictor (eqx leaves) plus a JSON sidecar and the scaler arrays."""
        artifact = Path(artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(artifact), self.model)
        mlp = self.model.model
        meta = {
            "in_size": int(mlp.in_size),
            "out_size": int(mlp.out_size),
            "hidden_size": int(mlp.width_size),
            "depth": int(mlp.depth),
            "activation": self.model.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "downsample": self.downsample,
            "dt": self.dt,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
        }
        artifact.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        np.savez(
            artifact.with_suffix(".scalers.npz"),
            u_mean=self.u_mean,
            u_scale=self.u_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale,
        )


def zscore(x: Any, mean: Any, scale: Any) -> Any:  # noqa: ANN401
    """Standardize ``x`` using precomputed per-channel ``mean``/``scale``.

    Works unchanged for NumPy/JAX arrays and CasADi ``SX``/``MX`` symbols, since all three
    overload elementwise ``-``/``/`` identically.
    """
    return (x - mean) / scale


def unzscore(x: Any, mean: Any, scale: Any) -> Any:  # noqa: ANN401
    """Invert :func:`zscore`, mapping standardized values back to raw units."""
    return x * scale + mean


class NNPredictor:
    """Equinox MLP trained for direct ``horizon``-step EEG prediction.

    Native conditioning: the input is ``[y_past (n_y), u_past (n_u), u_future (horizon)]``
    drawn from the tail of the context window plus the recorded future controls. Outputs
    are inverse-scaled to raw EEG units.
    """

    name = "nn"

    def __init__(  # noqa: PLR0913
        self,
        model: eqx.Module,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        dt: float,
        scalers: dict[str, FloatArray],
    ) -> None:
        self._model = model
        self._n_y = n_y
        self._n_u = n_u
        self._horizon = horizon
        self._n_ch = n_channels
        self.dt = dt
        self._u_mean = scalers["u_mean"]
        self._u_scale = scalers["u_scale"]
        self._y_mean = scalers["y_mean"]
        self._y_scale = scalers["y_scale"]

    @classmethod
    def load(cls, artifact: str | Path, **_kwargs: object) -> NNPredictor:
        """Rebuild the predictor from a saved :class:`MLPArtifact`."""
        art = MLPArtifact.load(artifact)
        return cls(
            art.model,
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            dt=art.dt,
            scalers={"u_mean": art.u_mean, "u_scale": art.u_scale, "y_mean": art.y_mean, "y_scale": art.y_scale},
        )

    def predict(self, window: PredictionWindow, horizon_s: float) -> FloatArray:
        """Assemble the network input from the context tail, predict, inverse-scale."""
        factor = _ds_factor(self.dt, window.dt_data)
        y_abs = np.asarray(window.y_ctx, dtype=np.float64)[:, ::factor].T  # (ctx, n_ch), absolute
        u_abs = _resample_controls(window.u_ctx, factor, y_abs.shape[0])  # (ctx, n_elec)
        n_steps = round(horizon_s / self.dt)

        num_chunks = int(np.ceil(n_steps / self._horizon)) if n_steps > 0 else 0
        if num_chunks == 0:
            return np.empty((self._n_ch, 0), dtype=np.float64)

        total_pred_steps = num_chunks * self._horizon
        u_fut_abs_all = _resample_controls(window.u_future, factor, total_pred_steps)

        y_abs_current = y_abs.copy()
        u_abs_current = u_abs.copy()
        y_pred_abs_all = []

        for i in range(num_chunks):
            chunk_u_fut_abs = u_fut_abs_all[i * self._horizon : (i + 1) * self._horizon]

            y_src, u_src, u_fut = y_abs_current, u_abs_current, chunk_u_fut_abs

            if y_src.shape[0] < self._n_y or u_src.shape[0] < self._n_u:
                msg = f"context too short for NN history (need n_y={self._n_y}, n_u={self._n_u}; got {y_src.shape[0]})"
                raise ValueError(msg)

            y_past = y_src[-self._n_y :]
            u_past = u_src[-self._n_u :]

            y_past_s = zscore(y_past, self._y_mean, self._y_scale)
            u_past_s = zscore(u_past, self._u_mean, self._u_scale)
            u_fut_s = zscore(u_fut, self._u_mean, self._u_scale)

            x_scaled = np.concatenate([y_past_s.flatten(), u_past_s.flatten(), u_fut_s.flatten()])
            y_scaled = np.asarray(self._model(jnp.asarray(x_scaled, dtype=jnp.float64)))  # type: ignore
            y_pred_scaled = y_scaled.reshape(self._horizon, self._n_ch)
            y_pred = unzscore(y_pred_scaled, self._y_mean, self._y_scale)

            y_pred_abs_all.append(y_pred)

            y_abs_current = np.concatenate([y_abs_current, y_pred], axis=0)
            u_abs_current = np.concatenate([u_abs_current, chunk_u_fut_abs], axis=0)

        y_pred_final = np.concatenate(y_pred_abs_all, axis=0)
        return y_pred_final.T[:, :n_steps]


def nrmse(pred: FloatArray, true: FloatArray) -> float:
    """Channel-mean RMSE normalised by each channel's true-window std (lower better).

    Parameters
    ----------
    pred : FloatArray
        Predicted EEG window, shape ``(n_channels, n_samples)``.
    true : FloatArray
        True EEG window, shape ``(n_channels, n_samples)``.

    Returns
    -------
    float
        The channel-mean normalised RMSE.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    std = np.std(true, axis=1) + _EPS
    return float(np.mean(rmse / std))


def pearson_corr(pred: FloatArray, true: FloatArray) -> float:
    """Pooled Pearson correlation over all channels/samples of a window.

    Parameters
    ----------
    pred : FloatArray
        Predicted EEG window, shape ``(n_channels, n_samples)``.
    true : FloatArray
        True EEG window, shape ``(n_channels, n_samples)``.

    Returns
    -------
    float
        The Pearson correlation coefficient between the flattened arrays.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    if pred.std() < _EPS or true.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def error_vs_leadtime(preds: list[FloatArray], trues: list[FloatArray]) -> FloatArray:
    """Std-normalised RMSE per lead step, averaged over windows and channels.

    Parameters
    ----------
    preds : list of FloatArray
        List of prediction windows, where each array has shape ``(n_channels, H)``.
    trues : list of FloatArray
        List of truth windows, where each array has shape ``(n_channels, H)``.

    Returns
    -------
    FloatArray
        NRMSE as a function of lead step, shape ``(H,)``.
    """
    errs = []
    for pred, true in zip(preds, trues, strict=True):
        std = np.std(true, axis=1, keepdims=True) + _EPS
        errs.append(np.sqrt(np.mean(((pred - true) / std) ** 2, axis=0)))
    return np.mean(np.stack(errs, axis=0), axis=0)


def fc(eeg: FloatArray) -> FloatArray:
    """Channel-by-channel functional-connectivity (correlation) matrix.

    Parameters
    ----------
    eeg : FloatArray
        EEG signal window, shape ``(n_channels, n_samples)``.

    Returns
    -------
    FloatArray
        The Pearson correlation matrix, shape ``(n_channels, n_channels)``.
    """
    return np.corrcoef(np.asarray(eeg, dtype=np.float64))


def fc_error(pred_fc: FloatArray, true_fc: FloatArray) -> float:
    """Mean-squared error of the off-diagonal FC entries.

    Parameters
    ----------
    pred_fc : FloatArray
        Predicted functional connectivity matrix, shape ``(n_channels, n_channels)``.
    true_fc : FloatArray
        True functional connectivity matrix, shape ``(n_channels, n_channels)``.

    Returns
    -------
    float
        The mean squared error of the upper triangular off-diagonal elements.
    """
    iu = np.triu_indices(true_fc.shape[0], k=1)
    return float(np.mean((pred_fc[iu] - true_fc[iu]) ** 2))


def fc_pattern_corr(pred_fc: FloatArray, true_fc: FloatArray) -> float:
    """Correlation between the off-diagonal FC patterns (spatial agreement).

    Parameters
    ----------
    pred_fc : FloatArray
        Predicted functional connectivity matrix, shape ``(n_channels, n_channels)``.
    true_fc : FloatArray
        True functional connectivity matrix, shape ``(n_channels, n_channels)``.

    Returns
    -------
    float
        The Pearson correlation between the upper triangular off-diagonal elements.
    """
    iu = np.triu_indices(true_fc.shape[0], k=1)
    return float(np.corrcoef(pred_fc[iu], true_fc[iu])[0, 1])


def persistence_baseline(window: PredictionWindow, horizon_s: float, analysis_dt: float) -> FloatArray:
    """Constant-hold baseline: repeat the last context sample over the horizon grid.

    Parameters
    ----------
    window : PredictionWindow
        The current test context containing the measured EEG.
    horizon_s : float
        The prediction horizon in seconds.
    analysis_dt : float
        The sampling interval for the predicted signal in seconds.

    Returns
    -------
    FloatArray
        The constant-hold baseline prediction, shape ``(n_channels, n_steps)``,
        where ``n_steps = round(horizon_s / analysis_dt)``.
    """
    n_steps = round(horizon_s / analysis_dt)
    last = np.asarray(window.y_ctx, dtype=np.float64)[:, -1:]
    return np.repeat(last, n_steps, axis=1)
