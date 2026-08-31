from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.predictor.data import build_dataset_for_trajectory

if TYPE_CHECKING:
    from neuro.predictor.module import AutoregressiveMLP
    from neuro.types import FloatArray


def dmd(
    X: FloatArray,
    Y: FloatArray,
    *,
    rank: int | None = None,
    energy: float | None = None,
    dmd_lambda: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
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

    Returns
    -------
    W : FloatArray
        Linear weight matrix of shape ``(n_outputs, n_features)``.
    b : FloatArray
        Affine bias vector of shape ``(n_outputs,)``.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)

    x_mean = np.mean(x_arr, axis=0)
    y_mean = np.mean(y_arr, axis=0)
    x_tilde = x_arr - x_mean
    y_tilde = y_arr - y_mean

    u_mat, s, vt = np.linalg.svd(x_tilde, full_matrices=False)
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

    u_r = u_mat[:, :r]
    s_r = s[:r]
    vt_r = vt[:r, :]

    s_inv = s_r / (s_r**2 + float(dmd_lambda)) if dmd_lambda > 0.0 else 1.0 / np.where(s_r > 0.0, s_r, 1e-12)

    w = np.ascontiguousarray((y_tilde.T @ u_r * s_inv[None, :]) @ vt_r)
    b = np.ascontiguousarray(y_mean - w @ x_mean)
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

    def fit(
        self,
        model: AutoregressiveMLP,
        trajectories: list[tuple[FloatArray, FloatArray]],
    ) -> AutoregressiveMLP:
        """Fit ``model``'s readout via Hankel-DMDc and install the weights and affine bias."""
        if getattr(model, "depth", 0) > 0 or not hasattr(model, "install_readout"):
            msg = f"DmdTrainer requires a depth-0 model with install_readout, got {type(model).__name__}."
            raise TypeError(msg)

        c = model.n_outputs
        m = model.n_controls
        y_len = model.n_y * c
        x_list: list[FloatArray] = []
        y_list: list[FloatArray] = []

        for u_raw, y_raw in trajectories:
            x_traj, y_traj = build_dataset_for_trajectory(
                model.u_std.transform(np.asarray(u_raw, dtype=np.float64)),
                model.y_std.transform(np.asarray(y_raw, dtype=np.float64)),
                model.n_y,
                model.n_u,
                model.horizon,
            )
            x_1step = np.hstack([x_traj[:, :y_len], x_traj[:, y_len + m : y_len + (model.n_u + 1) * m]])
            targets = y_traj[:, :c]
            if model.residual:
                targets = targets - x_traj[:, y_len - c : y_len]
            x_list.append(np.asarray(x_1step, dtype=np.float64))
            y_list.append(np.asarray(targets, dtype=np.float64))

        if not x_list:
            msg = "trajectories cannot be empty for DMD fitting."
            raise ValueError(msg)

        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        w, b = dmd(
            x_all,
            y_all,
            rank=self.rank,
            energy=self.energy,
            dmd_lambda=self.dmd_lambda,
        )

        readout = np.hstack([w, b[:, None]])
        model.install_readout(readout)
        return model
