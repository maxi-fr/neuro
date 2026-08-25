from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from neuro.transforms import Standardizer
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
    y_std: Standardizer,
    u_std: Standardizer,
    w_res: scipy.sparse.csr_matrix,
    w_in: FloatArray,
    leak_rate: float,
    priming_steps: int,
    noise_sigma: float,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Harvest normal equations G and P continuously over trajectories.

    The incumbent closed-form reference the torch ESN's ``design_normal_equations`` reproduces
    at ``noise_sigma = 0``; it stays as the one-time scipy preprocessing twin of the module's
    capability, not as a runtime.

    Parameters
    ----------
    trajectories : list[tuple[FloatArray, FloatArray]]
        List of (u, y) trajectories where y is raw EEG (T, n_channels) and u is raw control (T, n_controls).
    y_std : Standardizer
        Encoder for y -> z.
    u_std : Standardizer
        Encoder for u -> v.
    w_res : scipy.sparse.csr_matrix
        Reservoir matrix (N, N).
    w_in : FloatArray
        Input weight matrix (N, C + m + 1).
    leak_rate : float
        Leakage rate alpha.
    priming_steps : int
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
    C = y_std.transform(np.asarray(trajectories[0][1][:1], dtype=np.float64)).shape[1]

    G = np.zeros((N + 1, N + 1), dtype=np.float64)
    P = np.zeros((N + 1, C), dtype=np.float64)

    w_in_input_weights = w_in[:, :-1]
    w_in_bias = w_in[:, -1]

    for u_raw, y_raw in trajectories:
        z = y_std.transform(np.asarray(y_raw, dtype=np.float64))
        v = u_std.transform(np.asarray(u_raw, dtype=np.float64))
        T = len(z)
        h = np.zeros(N, dtype=np.float64)

        z_in = z + noise_sigma * rng.standard_normal(z.shape) if noise_sigma > 0 else z

        inputs = np.hstack([z_in, v])
        w_in_seq = inputs @ w_in_input_weights.T + w_in_bias

        n_harvest = max(0, T - priming_steps)
        if n_harvest > 0:
            H_mat = np.empty((n_harvest, N + 1), dtype=np.float64)
            H_mat[:, -1] = 1.0
            Z_mat = z[priming_steps:]

            for t in range(T):
                # h enters the row *before* absorbing (z[t], v[t]), so the target z[t] it is
                # paired with is a genuine one-step-ahead prediction rather than a reconstruction.
                if t >= priming_steps:
                    H_mat[t - priming_steps, :N] = h
                h = (1.0 - leak_rate) * h + leak_rate * np.tanh(w_res @ h + w_in_seq[t])

            G += H_mat.T @ H_mat
            P += H_mat.T @ Z_mat
        else:
            for t in range(T):
                h = (1.0 - leak_rate) * h + leak_rate * np.tanh(w_res @ h + w_in_seq[t])

    return G, P


def solve_ridge(G: FloatArray, P: FloatArray, ridge_lambda: float) -> FloatArray:
    """Solve ridge regression W_out = (G + lambda * I)^-1 P with unregularized bias column.

    The incumbent closed-form reference the torch ESN's ``solve_ridge`` reproduces to LAPACK
    precision; it stays as the one-time scipy preprocessing twin, not as a runtime.

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
        Readout weight matrix (C, N+1) mapping [h; 1] -> z.
    """
    N_plus_1 = G.shape[0]
    reg = ridge_lambda * np.eye(N_plus_1, dtype=np.float64)
    reg[-1, -1] = 0.0  # Do not regularize bias column

    w_out_t = np.linalg.solve(G + reg, P)
    return np.ascontiguousarray(w_out_t.T)
