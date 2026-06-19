# ruff: noqa: N806, N803
"""CasADi symbolic implementation of the Jansen-Rit neural mass model.

Provides differentiable RHS and Heun-step symbolic ``ca.SX`` expressions suitable
for embedding in optimization problems (MPC, NLP). Noise is not modelled -- the
stochastic ``sigma`` term lives in ``jansen_rit.py`` and has no place in a
differentiable shooting formulation.

State layout
------------
The symbolic state ``X`` is a ``6 x N`` matrix: row ``k`` holds state variable
``x_{k+1}`` across all ``N`` nodes (pyramidal ``x1``, excitatory ``x2``,
inhibitory ``x3`` and their derivatives ``x4, x5, x6``)::

    X[0, :]  =  x1  (pyramidal)
    X[1, :]  =  x2  (excitatory)
    X[2, :]  =  x3  (inhibitory)
    X[3, :]  =  x4
    X[4, :]  =  x5
    X[5, :]  =  x6

This is exactly ``np.reshape(x_aug[:6*N], (6, N))`` (C order) from the estimator,
so flattening a CasADi output in C order reproduces the estimator's dynamic-state
slice and can be compared directly against ``_fx_step_jit`` / ``_jr_rhs_jit``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca

if TYPE_CHECKING:
    import numpy as np

    from neuro.jansen_rit import JansenRitParams


def sigmoid(v: ca.SX | float, params: JansenRitParams) -> ca.SX:
    """Compute the sigmoidal transformation mapping average membrane potential to firing rate.

    Parameters
    ----------
    v
        Average membrane potential, as a scalar or CasADi symbolic expression.
    params
        Configuration dataclass holding the sigmoid constants ``e0``, ``v0``, ``r``.

    Returns
    -------
    ca.SX
        Symbolic firing rate ``2 e0 / (1 + exp(r (v0 - v)))``.
    """
    return 2 * params.e0 / (1.0 + ca.exp(params.r * (params.v0 - v)))


def f_rhs(X: ca.SX, coupling: ca.SX, u: ca.SX | float, A: ca.SX, params: JansenRitParams) -> ca.SX:
    """Compute the continuous-time right-hand side (RHS) derivatives for the network nodes.

    Parameters
    ----------
    X
        Current node states as a ``6 x N`` symbolic matrix.
    coupling
        Structural coupling row vector of shape ``1 x N``. Already includes the
        firing-rate sigmoid and global gain ``K`` (see :func:`get_network_coupling`).
    u
        External tES input on the pyramidal population, scalar or shape ``1 x N``.
    A
        Excitatory synaptic gain, scalar or shape ``1 x N`` (broadcast over the rows).
    params
        Configuration dataclass holding the biological and structural constants.

    Returns
    -------
    ca.SX
        A ``6 x N`` symbolic matrix of the state derivatives ``dX/dt``.
    """
    B, a, b = params.B, params.a, params.b
    C1, C2, C3, C4 = params.C1, params.C2, params.C3, params.C4
    mean_input = params.mean_input

    x1, x2, x3, x4, x5, x6 = X[0, :], X[1, :], X[2, :], X[3, :], X[4, :], X[5, :]
    N: int = X.shape[1]

    rhs = ca.SX(6, N)
    rhs[0, :] = x4
    rhs[1, :] = x5
    rhs[2, :] = x6
    rhs[3, :] = A * a * sigmoid(x2 - x3 + u, params) - 2 * a * x4 - a**2 * x1
    rhs[4, :] = A * a * (mean_input + C2 * sigmoid(C1 * x1, params) + coupling) - 2 * a * x5 - a**2 * x2
    rhs[5, :] = B * b * (C4 * sigmoid(C3 * x1, params)) - 2 * b * x6 - b**2 * x3

    return rhs


def get_network_coupling(
    X_history_list: list[ca.SX], K: ca.SX, W: ca.SX, D: np.ndarray, params: JansenRitParams
) -> ca.SX:
    """Calculate the structural delayed-coupling inputs acting on all nodes in the network.

    Implements ``c_i = K * sum_j W[i, j] * S(x2_j - x3_j)`` evaluated at the delayed
    time ``t - D[i, j]``, matching the paper's ``K sum_j w_ij S(x2 - x3)`` term and the
    numba reference (``_dynamics_history_coupling_jit`` / ``_fx_step_jit``).

    Parameters
    ----------
    X_history_list
        Chronologically ordered list of ``6 x N`` state matrices, where index 0 is
        time ``t``, index 1 is ``t - 1``, etc. Must hold at least ``int(D.max()) + 1``
        entries.
    K
        Global coupling scaling factor symbol.
    W
        ``N x N`` structural connectivity weight matrix symbol.
    D
        ``N x N`` static numpy matrix of exact integer integration-step delays.
    params
        Configuration dataclass holding the sigmoid constants ``e0``, ``v0``, ``r``.

    Returns
    -------
    ca.SX
        A ``1 x N`` symbolic row vector of total structural coupling incoming to each node.
    """
    N: int = W.shape[0]
    coupling_vector = ca.SX(1, N)

    for i in range(N):
        c_i: ca.SX | float = 0
        for j in range(N):
            delay_steps: int = int(D[i, j])
            X_past: ca.SX = X_history_list[delay_steps]

            c_i += K * W[i, j] * sigmoid(X_past[1, j] - X_past[2, j], params)

        coupling_vector[i] = c_i

    return coupling_vector


def heun_step(  # noqa: PLR0913
    X_history_list: list[ca.SX],
    u: ca.SX | float,
    K: ca.SX,
    W: ca.SX,
    D: np.ndarray,
    A: ca.SX,
    params: JansenRitParams,
    dt: float,
) -> ca.SX:
    """Advance the delayed Jansen-Rit network states by one discrete step using Heun's method.

    Parameters
    ----------
    X_history_list
        Chronologically ordered list of ``6 x N`` state matrices, where index 0 is the
        current state snapshot ``X_t``.
    u
        External tES input vector or constant applied to the node populations.
    K
        Global coupling scaling factor symbol.
    W
        ``N x N`` structural connectivity weight matrix symbol.
    D
        ``N x N`` static numpy matrix of exact integer integration-step delays.
    A
        Excitatory synaptic gain, scalar or shape ``1 x N``.
    params
        Configuration dataclass holding the biological and structural constants.
    dt
        Step size for numerical integration in seconds.

    Returns
    -------
    ca.SX
        A ``6 x N`` symbolic matrix of the integrated next-step states ``X_{t+1}``.
    """
    X_curr: ca.SX = X_history_list[0]

    coupling: ca.SX = get_network_coupling(X_history_list, K, W, D, params)
    k1: ca.SX = f_rhs(X_curr, coupling, u, A, params)

    X_pred: ca.SX = X_curr + dt * k1

    # Note: If true predictor-corrector delay updates are needed here later,
    # you would append X_pred to a copied state list before querying coup_pred.
    # X_history_pred = [X_pred] + X_history_list[:-1]  # noqa: ERA001
    # coupling = get_network_coupling(X_history_pred, K, W, D, params)  # noqa: ERA001

    k2: ca.SX = f_rhs(X_pred, coupling, u, A, params)

    return X_curr + 0.5 * dt * (k1 + k2)
