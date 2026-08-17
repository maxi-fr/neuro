from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from neuro.transforms import Pipeline

if TYPE_CHECKING:
    from neuro.types import FloatArray


def generate_reservoir(  # noqa: PLR0913, PLR0917
    reservoir_size: int,
    spectral_radius: float,
    density: float,
    input_scaling: float,
    in_dim: int,
    seed: int,
) -> tuple[scipy.sparse.csr_matrix, FloatArray]:
    """Generate sparse reservoir weight matrix W_res and dense input weight matrix W_in.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (N).
    spectral_radius : float
        Target spectral radius (rho) for W_res.
    density : float
        Sparsity density d of W_res in (0, 1].
    input_scaling : float
        Scaling gamma for input weights W_in uniform in [-gamma, gamma].
    in_dim : int
        Dimension of reservoir input vector [z; v; 1] = C + m + 1.
    seed : int
        Random seed.

    Returns
    -------
    W_res : scipy.sparse.csr_matrix
        Rescaled reservoir matrix (N, N).
    W_in : FloatArray
        Input weight matrix (N, in_dim).
    """
    rng = np.random.default_rng(seed)

    def data_rvs(size: int) -> FloatArray:
        return rng.uniform(-1.0, 1.0, size)

    w_res_raw = scipy.sparse.random(
        reservoir_size,
        reservoir_size,
        density=density,
        format="csr",
        random_state=rng,
        data_rvs=data_rvs,
    )

    try:
        vals = scipy.sparse.linalg.eigs(w_res_raw, k=1, which="LM", return_eigenvectors=False)
        abs_lam_max = float(np.abs(vals[0]))
    except Exception:  # noqa: BLE001
        vals_dense = np.linalg.eigvals(w_res_raw.toarray())
        abs_lam_max = float(np.max(np.abs(vals_dense)))

    if abs_lam_max == 0.0 or not np.isfinite(abs_lam_max):
        msg = f"Failed to compute non-zero spectral radius for reservoir (got {abs_lam_max})"
        raise ValueError(msg)

    w_res = w_res_raw * (spectral_radius / abs_lam_max)
    w_in = rng.uniform(-input_scaling, input_scaling, size=(reservoir_size, in_dim))
    return w_res, w_in


def harvest_normal_equations(  # noqa: PLR0913, PLR0917
    trajectories: list[tuple[FloatArray, FloatArray]],
    y_pipeline: Pipeline,
    u_pipeline: Pipeline,
    w_res: scipy.sparse.csr_matrix,
    w_in: FloatArray,
    leak_rate: float,
    washout: int,
    noise_sigma: float,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Harvest normal equations G and P continuously over trajectories.

    Parameters
    ----------
    trajectories : list[tuple[FloatArray, FloatArray]]
        List of (u, y) trajectories where y is raw EEG (T, n_eeg_channels) and u is raw control (T, n_controls).
    y_pipeline : Pipeline
        Encoder for y -> z.
    u_pipeline : Pipeline
        Encoder for u -> v.
    w_res : scipy.sparse.csr_matrix
        Reservoir matrix (N, N).
    w_in : FloatArray
        Input weight matrix (N, C + m + 1).
    leak_rate : float
        Leakage rate alpha.
    washout : int
        Number of initial steps to discard per trajectory.
    noise_sigma : float
        Noise injection standard deviation sigma on model-space input z_in.
    seed : int
        Random seed for noise injection.

    Returns
    -------
    G : FloatArray
        State covariance matrix sum (N+1, N+1).
    P : FloatArray
        State-target cross-covariance matrix sum (N+1, C).
    """
    rng = np.random.default_rng(seed)
    N = w_res.shape[0]
    C = y_pipeline.transform(np.asarray(trajectories[0][1][:1], dtype=np.float64)).shape[1]

    G = np.zeros((N + 1, N + 1), dtype=np.float64)
    P = np.zeros((N + 1, C), dtype=np.float64)

    w_in_input_weights = w_in[:, :-1]
    w_in_bias = w_in[:, -1]

    for u_raw, y_raw in trajectories:
        z = y_pipeline.transform(np.asarray(y_raw, dtype=np.float64))
        v = u_pipeline.transform(np.asarray(u_raw, dtype=np.float64))
        T = len(z)
        h = np.zeros(N, dtype=np.float64)

        z_in = z + noise_sigma * rng.standard_normal(z.shape) if noise_sigma > 0 else z

        inputs = np.hstack([z_in, v])
        w_in_seq = inputs @ w_in_input_weights.T + w_in_bias

        n_harvest = max(0, T - washout)
        if n_harvest > 0:
            H_mat = np.empty((n_harvest, N + 1), dtype=np.float64)
            H_mat[:, -1] = 1.0
            Z_mat = z[washout:]

            for t in range(T):
                # h enters the row *before* absorbing (z[t], v[t]), so the target z[t] it is
                # paired with is a genuine one-step-ahead prediction rather than a reconstruction.
                if t >= washout:
                    H_mat[t - washout, :N] = h
                h = (1.0 - leak_rate) * h + leak_rate * np.tanh(w_res @ h + w_in_seq[t])

            G += H_mat.T @ H_mat
            P += H_mat.T @ Z_mat
        else:
            for t in range(T):
                h = (1.0 - leak_rate) * h + leak_rate * np.tanh(w_res @ h + w_in_seq[t])

    return G, P


def solve_ridge(G: FloatArray, P: FloatArray, ridge_lambda: float) -> FloatArray:
    """Solve ridge regression W_out = (G + lambda * I)^-1 P with unregularized bias column.

    Parameters
    ----------
    G : FloatArray
        State covariance matrix (N+1, N+1).
    P : FloatArray
        State-target cross-covariance matrix (N+1, C).
    ridge_lambda : float
        L2 regularization weight lambda.

    Returns
    -------
    W_out : FloatArray
        Readout weight matrix (C, N+1).
    """
    N_plus_1 = G.shape[0]
    reg = ridge_lambda * np.eye(N_plus_1, dtype=np.float64)
    reg[-1, -1] = 0.0  # Do not regularize bias column

    w_out_t = np.linalg.solve(G + reg, P)
    return np.ascontiguousarray(w_out_t.T)


def _append_bias(x: FloatArray) -> FloatArray:
    """Concatenate a constant-1 column onto the last axis of ``x`` (batch dims pass through)."""
    return np.concatenate([x, np.ones((*x.shape[:-1], 1), dtype=np.float64)], axis=-1)


@dataclass(frozen=True)
class ESNPredictor:
    """Numpy ESN: the reservoir recursion and the readout, both in model space."""

    w_in: FloatArray
    w_out: FloatArray
    w_res: scipy.sparse.csr_matrix
    leak_rate: float

    def readout(self, h: FloatArray) -> FloatArray:
        """One-step-ahead model-space prediction z_hat from state ``h`` ``(..., N)`` -> ``(..., C)``."""
        return _append_bias(h) @ self.w_out.T

    def teacher_step(self, h: FloatArray, z: FloatArray, v: FloatArray) -> FloatArray:
        """Advance ``h`` ``(..., N)`` absorbing the model-space input ``(z, v)``."""
        alpha = self.leak_rate
        x_in = _append_bias(np.concatenate([z, v], axis=-1))
        return (1.0 - alpha) * h + alpha * np.tanh(h @ self.w_res.T + x_in @ self.w_in.T)

    def step(self, h: FloatArray, v: FloatArray) -> FloatArray:
        """Advance ``h`` free-running under model-space control ``v``: the readout replaces z."""
        return self.teacher_step(h, self.readout(h), v)


@dataclass(frozen=True)
class ESNArtifact:
    """Loaded ESN artifact containing weight matrices, pipeline transforms, and metadata."""

    w_in: FloatArray
    w_out: FloatArray
    w_res: scipy.sparse.csr_matrix
    dt: float
    downsample: int
    horizon: int
    reservoir_size: int
    leak_rate: float
    spectral_radius: float
    washout: int
    input_scaling: float
    density: float
    noise_sigma: float
    ridge_lambda: float
    seed: int
    y_pipeline: Pipeline
    u_pipeline: Pipeline

    @cached_property
    def predictor(self) -> ESNPredictor:
        """Numpy ESN backing :meth:`prime` and :meth:`rollout`."""
        return ESNPredictor(w_in=self.w_in, w_out=self.w_out, w_res=self.w_res, leak_rate=self.leak_rate)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('esn')."""
        return "esn"

    @property
    def priming_steps(self) -> int:
        """Priming steps needed for history absorption (washout)."""
        return self.washout

    @property
    def n_channels(self) -> int:
        """Number of model-space y output channels C."""
        return self.w_out.shape[0]

    @property
    def n_controls(self) -> int:
        """Number of control input channels m, read off W_in's ``[z; v; 1]`` input width."""
        return self.w_in.shape[1] - self.n_channels - 1

    @property
    def n_eeg_channels(self) -> int:
        """Number of raw EEG channels."""
        pca = self.y_pipeline.pca
        return pca.basis.shape[1] if pca is not None else self.n_channels

    @property
    def meta(self) -> dict[str, Any]:
        """Serializable dictionary representation of artifact metadata."""
        return {
            "model_type": self.model_type,
            "dt": self.dt,
            "downsample": self.downsample,
            "horizon": self.horizon,
            "reservoir_size": self.reservoir_size,
            "leak_rate": self.leak_rate,
            "spectral_radius": self.spectral_radius,
            "washout": self.washout,
            "input_scaling": self.input_scaling,
            "density": self.density,
            "noise_sigma": self.noise_sigma,
            "ridge_lambda": self.ridge_lambda,
            "seed": self.seed,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_eeg_channels": self.n_eeg_channels,
            "y_pipeline": self.y_pipeline.step_tags(),
            "u_pipeline": self.u_pipeline.step_tags(),
        }

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG (..., n_eeg_channels) into model space."""
        return self.y_pipeline.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG (..., n_eeg_channels) from model space."""
        return self.y_pipeline.inverse_transform(np.asarray(z, dtype=np.float64))

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into initial reservoir state h0 = 0.

        Parameters
        ----------
        y_hist : FloatArray
            Raw EEG history (k, n_eeg_channels).
        u_hist : FloatArray
            Raw control history (k, n_controls).

        Returns
        -------
        h : FloatArray
            Primed reservoir state (N,).
        """
        z = self.encode(y_hist)
        v = self.u_pipeline.transform(np.asarray(u_hist, dtype=np.float64))
        esn = self.predictor
        h = np.zeros(self.reservoir_size, dtype=np.float64)

        for t in range(len(z)):
            h = esn.teacher_step(h, z[t], v[t])
        return h

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_eeg_channels)`` and ``(B, k, n_controls)`` -> ``(B, N)``."""
        z = self.encode(y_hists)
        v = self.u_pipeline.transform(np.asarray(u_hists, dtype=np.float64))
        esn = self.predictor
        h = np.zeros((z.shape[0], self.reservoir_size), dtype=np.float64)

        for t in range(z.shape[1]):
            h = esn.teacher_step(h, z[:, t], v[:, t])
        return h

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from state under raw u_future -> (steps, n_eeg_channels).

        Parameters
        ----------
        state : FloatArray
            Initial reservoir state (N,).
        u_future : FloatArray
            Future control actions (steps, n_controls).

        Returns
        -------
        y_preds : FloatArray
            Predicted raw EEG trajectories (steps, n_eeg_channels).
        """
        v_future = self.u_pipeline.transform(np.asarray(u_future, dtype=np.float64))
        esn = self.predictor
        h = np.asarray(state, dtype=np.float64).copy()

        n_steps = len(v_future)
        preds_z = np.zeros((n_steps, self.n_channels), dtype=np.float64)
        for t in range(n_steps):
            z_hat = esn.readout(h)
            preds_z[t] = z_hat
            h = esn.teacher_step(h, z_hat, v_future[t])

        return self.decode(preds_z)

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, N)`` and raw ``(B, steps, n_controls)``.

        Returns ``(B, steps, n_eeg_channels)``.
        """
        v_future = self.u_pipeline.transform(np.asarray(u_futures, dtype=np.float64))
        esn = self.predictor
        h = np.array(states, dtype=np.float64)

        n_batch, n_steps = v_future.shape[0], v_future.shape[1]
        preds_z = np.zeros((n_batch, n_steps, self.n_channels), dtype=np.float64)
        for t in range(n_steps):
            z_hat = esn.readout(h)
            preds_z[:, t] = z_hat
            h = esn.teacher_step(h, z_hat, v_future[:, t])

        return self.decode(preds_z)

    @classmethod
    def load(cls, artifact: str | Path) -> ESNArtifact:
        """Load the single-``.npz`` ESN artifact from disk (``artifact`` is a suffix-less stem)."""
        path = Path(artifact).with_suffix(".npz")
        with np.load(path) as npz:
            meta: dict[str, Any] = json.loads(str(npz["meta"]))
            w_in = np.asarray(npz["W_in"], dtype=np.float64)
            w_out = np.asarray(npz["W_out"], dtype=np.float64)
            w_res_data = np.asarray(npz["W_res.data"], dtype=np.float64)
            w_res_indices = np.asarray(npz["W_res.indices"], dtype=np.int32)
            w_res_indptr = np.asarray(npz["W_res.indptr"], dtype=np.int32)
            w_res_shape = tuple(npz["W_res.shape"])
            w_res = scipy.sparse.csr_matrix((w_res_data, w_res_indices, w_res_indptr), shape=w_res_shape)
            arrays = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files if k.startswith(("y.", "u."))}

        return cls(
            w_in=w_in,
            w_out=w_out,
            w_res=w_res,
            dt=float(meta["dt"]),
            downsample=int(meta["downsample"]),
            horizon=int(meta["horizon"]),
            reservoir_size=int(meta["reservoir_size"]),
            leak_rate=float(meta["leak_rate"]),
            spectral_radius=float(meta["spectral_radius"]),
            washout=int(meta["washout"]),
            input_scaling=float(meta["input_scaling"]),
            density=float(meta["density"]),
            noise_sigma=float(meta["noise_sigma"]),
            ridge_lambda=float(meta["ridge_lambda"]),
            seed=int(meta["seed"]),
            y_pipeline=Pipeline.from_serialized("y", meta["y_pipeline"], arrays),
            u_pipeline=Pipeline.from_serialized("u", meta["u_pipeline"], arrays),
        )

    def save(self, artifact: str | Path) -> None:
        """Persist weights, transforms and metadata into one ``.npz`` (``artifact`` is a stem).

        ``meta`` is stored as a 0-d unicode array holding JSON, so loading needs no ``allow_pickle``.
        """
        path = Path(artifact).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray[Any, Any]] = {
            "meta": np.array(json.dumps(self.meta)),
            "W_in": self.w_in,
            "W_out": self.w_out,
            "W_res.data": self.w_res.data,
            "W_res.indices": self.w_res.indices,
            "W_res.indptr": self.w_res.indptr,
            "W_res.shape": np.array(self.w_res.shape),
        }
        arrays.update(self.y_pipeline.array_dict("y"))
        arrays.update(self.u_pipeline.array_dict("u"))

        np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
