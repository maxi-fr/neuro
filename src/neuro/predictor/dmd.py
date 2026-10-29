from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.predictor.data import build_dataset_for_trajectory
from neuro.types import RidgeFittable

if TYPE_CHECKING:
    from neuro.predictor.base import AutoregressiveModel
    from neuro.types import FloatArray


def dmd(  # noqa: PLR0913 -- X, Y plus SVD truncation, regularization, and diagnostics options
    X: FloatArray,
    Y: FloatArray,
    *,
    rank: int | None = None,
    energy: float | None = None,
    dmd_lambda: float = 0.0,
    return_diagnostics: bool = False,
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, int, float]:
    """Solve the Hankel-DMDc linear operator and affine bias: ``y ≈ W x + b``.

    Computes the truncated SVD on mean-centered inputs ``X̃ = X - x̄``, projects outputs
    ``Ỹ = Y - ȳ`` onto the truncated subspace with optional Tikhonov damping, and recovers the
    affine bias ``b = ȳ - W x̄``.

    Parameters
    ----------
    X : FloatArray
        Input features array of shape ``(n_samples, n_features)``.
    Y : FloatArray
        Target outputs array of shape ``(n_samples, n_outputs)``.
    rank : int | None, optional
        Explicit SVD truncation rank. If specified, overrides ``energy``.
    energy : float | None, optional
        Singular value cumulative energy threshold in ``(0, 1]`` (variance explained).
        Used when ``rank`` is ``None``. Defaults to 0.99 if both ``rank`` and ``energy`` are ``None``.
    dmd_lambda : float, optional
        Tikhonov regularization parameter applied to inverted singular values.
    return_diagnostics : bool, default=False
        Whether to return the retained SVD rank and condition number.

    Returns
    -------
    W : FloatArray
        Linear weight matrix of shape ``(n_outputs, n_features)``.
    b : FloatArray
        Affine bias vector of shape ``(n_outputs,)``.
    r : int
        Retained SVD rank, returned only when ``return_diagnostics=True``.
    cond : float
        Condition number of retained singular values, returned only when ``return_diagnostics=True``.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)

    x_mean = np.mean(x_arr, axis=0)
    y_mean = np.mean(y_arr, axis=0)
    x_tilde = x_arr - x_mean
    y_tilde = y_arr - y_mean

    m, n = x_tilde.shape
    if m >= n:
        c_mat = x_tilde.T @ x_tilde
        evals, evecs = np.linalg.eigh(c_mat)
        idx = np.argsort(evals)[::-1]
        evals = np.maximum(evals[idx], 0.0)
        s = np.sqrt(evals)
        vt = evecs[:, idx].T
        use_gram = True
        u_mat = None
    else:
        u_mat, s, vt = np.linalg.svd(x_tilde, full_matrices=False)
        use_gram = False

    k = len(s)
    if k == 0:
        msg = "Cannot fit DMD on empty features."
        raise ValueError(msg)

    if rank is not None:
        r = min(int(rank), k)
    else:
        target_energy = float(energy) if energy is not None else 0.99
        s_sq = s**2
        total_energy = np.sum(s_sq)
        if total_energy <= 0.0:
            r = 1
        else:
            cum_energy = np.cumsum(s_sq) / total_energy
            r = int(np.searchsorted(cum_energy, target_energy)) + 1

    r = max(1, min(r, k))

    s_r = s[:r]
    vt_r = vt[:r, :]

    s_inv = s_r / (s_r**2 + float(dmd_lambda)) if dmd_lambda > 0.0 else 1.0 / np.where(s_r > 0.0, s_r, 1e-12)

    if use_gram:
        safe_s_r = np.where(s_r > 0.0, s_r, 1e-12)
        yx = y_tilde.T @ x_tilde
        w = np.ascontiguousarray((yx @ vt_r.T * (s_inv / safe_s_r)[None, :]) @ vt_r)
    else:
        u_r = u_mat[:, :r]
        w = np.ascontiguousarray((y_tilde.T @ u_r * s_inv[None, :]) @ vt_r)

    b = np.ascontiguousarray(y_mean - w @ x_mean)
    if return_diagnostics:
        cond = float(s_r[0] / max(s_r[-1], 1e-12))
        return w, b, r, cond
    return w, b


class DmdTrainer:
    """Closed-form Hankel-DMDc Trainer: fits the linear readout and bias of depth-0 Predictors."""

    def __init__(
        self,
        *,
        rank: int | None = None,
        energy: float | None = None,
        dmd_lambda: float = 0.0,
    ) -> None:
        """Store SVD rank, cumulative energy threshold, and Tikhonov regularization."""
        self.rank = int(rank) if rank is not None else None
        self.energy = float(energy) if energy is not None else None
        self.dmd_lambda = float(dmd_lambda)
        self.retained_rank: int | None = None
        self.condition_number: float | None = None

    def fit(
        self,
        model: AutoregressiveModel,
        trajectories: list[tuple[FloatArray, FloatArray]],
    ) -> AutoregressiveModel:
        """Fit flattened standardized waveform or Observable outputs with Hankel-DMDc."""
        if getattr(model, "depth", 0) > 0 or not isinstance(model, RidgeFittable):
            msg = f"DmdTrainer requires a depth-0 RidgeFittable model, got {type(model).__name__}."
            raise TypeError(msg)
        if getattr(model, "residual", False):
            msg = "DmdTrainer requires direct prediction (model.residual=False)."
            raise ValueError(msg)

        c = model.n_outputs
        m = model.n_controls
        y_len = model.n_y * c
        x_list: list[FloatArray] = []
        y_list: list[FloatArray] = []

        for u_raw, y_raw in trajectories:
            y_arr = model.y_std.transform(np.asarray(y_raw, dtype=np.float64))
            if y_arr.ndim > 2:  # noqa: PLR2004 -- 2D check for observable frames
                y_arr = y_arr.reshape(len(y_arr), -1)
            x_traj, y_traj = build_dataset_for_trajectory(
                model.u_std.transform(np.asarray(u_raw, dtype=np.float64)),
                y_arr,
                model.n_y,
                model.n_u,
                model.horizon,
            )
            x_1step = np.hstack([x_traj[:, :y_len], x_traj[:, y_len + m : y_len + (model.n_u + 1) * m]])
            targets = y_traj[:, :c]
            x_list.append(np.asarray(x_1step, dtype=np.float64))
            y_list.append(np.asarray(targets, dtype=np.float64))

        if not x_list:
            msg = "trajectories cannot be empty for DMD fitting."
            raise ValueError(msg)

        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        w, b, r, cond = dmd(
            x_all,
            y_all,
            rank=self.rank,
            energy=self.energy,
            dmd_lambda=self.dmd_lambda,
            return_diagnostics=True,
        )
        self.retained_rank = r
        self.condition_number = cond

        readout = np.hstack([w, b[:, None]])
        model.install_readout(readout)
        return model
