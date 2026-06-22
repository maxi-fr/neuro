"""CasADi symbolic implementation of the Jansen-Rit neural mass model.

Provides differentiable RHS and Heun-step symbolic ``ca.SX`` expressions suitable
for embedding in optimization problems (MPC, NLP). Noise is not modelled -- the
stochastic ``sigma`` term lives in ``jansen_rit.py`` and has no place in a
differentiable shooting formulation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class JRSymbolicParams:
    """Jansen-Rit parameter set, typed for CasADi symbolic graphs and numpy evaluation."""

    # Node-level arrays
    A: ca.SX | np.ndarray
    mean_input: ca.SX | np.ndarray
    eeg_gain: ca.SX | np.ndarray

    # Edge-level matrix
    w_weights: ca.SX | np.ndarray

    # Scalars
    B: ca.SX | float
    a: ca.SX | float
    b: ca.SX | float
    C1: ca.SX | float
    C2: ca.SX | float
    C3: ca.SX | float
    C4: ca.SX | float
    e0: ca.SX | float
    v0: ca.SX | float
    r: ca.SX | float
    sigma: ca.SX | float
    K: ca.SX | float

    # Pure structural
    delay_steps: ca.SX | np.ndarray
    gamma: ca.SX | np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))


def sigmoid(v: ca.SX | ca.MX | float, params: JRSymbolicParams) -> ca.SX | ca.MX:
    """Compute the sigmoidal transformation mapping average membrane potential to firing rate.

    Parameters
    ----------
    v : ca.SX | ca.MX | float
        Average membrane potential.
    params : JRSymbolicParams
        Symbolic parameters for the network.

    Returns
    -------
    ca.SX | ca.MX
        Average firing rate.
    """
    return 2 * params.e0 / (1.0 + ca.exp(params.r * (params.v0 - v)))


def f_rhs(
    X: ca.SX | ca.MX, coupling: ca.SX | ca.MX, u: ca.SX | ca.MX | float | np.ndarray, params: JRSymbolicParams
) -> ca.SX | ca.MX:
    """Compute the continuous-time right-hand side (RHS) derivatives for the network nodes.

    Parameters
    ----------
    X : ca.SX | ca.MX
        Current state matrix of shape (6, N).
    coupling : ca.SX | ca.MX
        Network coupling input vector of shape (1, N).
    u : ca.SX | ca.MX | float
        External control input projected to the nodes, of shape (1, N) or scalar.
    params : JRSymbolicParams
        Symbolic parameters for the network.

    Returns
    -------
    ca.SX | ca.MX
        The RHS state derivatives, of the same shape and type as X.
    """
    B, a, b = params.B, params.a, params.b
    C1, C2, C3, C4 = params.C1, params.C2, params.C3, params.C4
    mean_input = params.mean_input
    A = params.A

    x1, x2, x3, x4, x5, x6 = X[0, :], X[1, :], X[2, :], X[3, :], X[4, :], X[5, :]
    N: int = X.shape[1]

    rhs = type(X)(6, N)
    rhs[0, :] = x4
    rhs[1, :] = x5
    rhs[2, :] = x6
    rhs[3, :] = A * a * sigmoid(x2 - x3 + u, params) - 2 * a * x4 - a**2 * x1
    rhs[4, :] = A * a * (mean_input + C2 * sigmoid(C1 * x1, params) + coupling) - 2 * a * x5 - a**2 * x2
    rhs[5, :] = B * b * (C4 * sigmoid(C3 * x1, params)) - 2 * b * x6 - b**2 * x3

    return rhs


def get_network_coupling(X_history_list: Sequence[ca.SX | ca.MX], params: JRSymbolicParams) -> ca.SX | ca.MX:
    """Calculate the structural delayed-coupling inputs acting on all nodes in the network.

    Parameters
    ----------
    X_history_list : Sequence[ca.SX | ca.MX]
        List of past state matrices, ordered from newest (index 0) to oldest.
    params : JRSymbolicParams
        Symbolic parameters containing connectivity weights and delay steps.

    Returns
    -------
    ca.SX | ca.MX
        The delayed coupling input vector of shape (1, N).
    """
    K = params.K
    W = params.w_weights
    D = np.asarray(params.delay_steps)

    N: int = X_history_list[0].shape[1]

    # Group edges by their integer delay so each unique delay contributes a single masked
    # matmul ``W_d @ S(lfp(X[t-d]))`` instead of an O(N^2) scalar double loop. ``S(lfp)`` for
    # a given delay is built once over all nodes (the loop rebuilt it once per source node).
    # Same arithmetic, regrouped; the graph shrinks from O(N^2) nodes to O(#unique_delays).
    coupling_vector = type(X_history_list[0])(1, N)
    for d in np.unique(D):
        mask = ca.DM((d == D).astype(np.float64))  # (N, N) numeric edge mask
        w_d = ca.times(W, mask)  # elementwise; W may be numeric or symbolic
        s_past = sigmoid(lfp(X_history_list[int(d)]), params).T  # (N, 1)
        coupling_vector = coupling_vector + ca.mtimes(w_d, s_past).T  # (1, N)

    return K * coupling_vector


def heun_step(
    X_history_list: Sequence[ca.SX | ca.MX],
    u: ca.SX | ca.MX | float | np.ndarray,
    params: JRSymbolicParams,
    dt: float,
) -> ca.SX | ca.MX:
    """Advance the delayed Jansen-Rit network states by one discrete step using Heun's method.

    Parameters
    ----------
    X_history_list : Sequence[ca.SX | ca.MX]
        List of past state matrices, ordered from newest (index 0) to oldest.
    u : ca.SX | ca.MX | float
        External control input projected to the nodes.
    params : JRSymbolicParams
        Symbolic parameters for the network.
    dt : float
        Integration time step.

    Returns
    -------
    ca.SX | ca.MX
        The next state matrix of shape (6, N).
    """
    X_curr = X_history_list[0]

    coupling = get_network_coupling(X_history_list, params)
    k1 = f_rhs(X_curr, coupling, u, params)

    X_pred = X_curr + dt * k1
    k2 = f_rhs(X_pred, coupling, u, params)

    return X_curr + 0.5 * dt * (k1 + k2)


def lfp(X: ca.SX | ca.MX) -> ca.SX | ca.MX:
    """Observed output ``y = x2 - x3`` from a state trajectory or state.

    Parameters
    ----------
    X : ca.SX | ca.MX
        Current state matrix of shape (6, N).

    Returns
    -------
    ca.SX | ca.MX
        The scalar LFP output per node, of shape (1, N).
    """
    return X[1, :] - X[2, :]


def eeg(
    X: ca.SX | ca.MX, gain: ca.SX | ca.MX | np.ndarray, selected_channels: np.ndarray | list[int] | None = None
) -> ca.SX | ca.MX:
    """Measure the EEG output.

    Parameters
    ----------
    X : ca.SX | ca.MX
        Current state matrix of shape (6, N).
    gain : ca.SX | ca.MX | np.ndarray
        Leadfield measurement matrix.
    selected_channels : np.ndarray | list[int] | None
        Optional subset of channels to output.

    Returns
    -------
    ca.SX | ca.MX
        The measured output vector.
    """
    node_lfp = lfp(X)  # shape 1 x N
    y = ca.mtimes(gain, node_lfp.T)

    if selected_channels is not None:
        y = y[selected_channels, :]

    return y


def project_control(
    u_elec: ca.SX | ca.MX | float | np.ndarray, gamma_2d: np.ndarray | ca.SX | ca.MX
) -> ca.SX | ca.MX | np.ndarray:
    """Project per-electrode tES current ``u`` onto nodes via ``gamma``.

    Parameters
    ----------
    u_elec : ca.SX | ca.MX | float | np.ndarray
        Control input applied to the electrodes.
    gamma_2d : np.ndarray | ca.SX | ca.MX
        Spatial projection matrix from electrodes to brain nodes; symbolic when
        ``gamma`` is a SysID decision variable.

    Returns
    -------
    ca.SX | ca.MX | np.ndarray
        The projected input to each node, of shape (1, N); ``np.ndarray`` only for
        the all-numeric path, which still broadcasts into a symbolic graph.
    """
    u_elec_t = u_elec
    if isinstance(u_elec, (ca.SX, ca.MX)):
        if u_elec.shape[0] != 1 and u_elec.shape[1] == 1:
            u_elec_t = u_elec.T
        return ca.mtimes(u_elec_t, gamma_2d)

    u_vec = np.asarray(u_elec, dtype=np.float64).flatten()
    if not np.any(u_vec):
        return ca.SX(0.0)
    return u_vec @ gamma_2d


def rollout(
    x0_history: Sequence[ca.SX | ca.MX],
    controls: Sequence[ca.SX | ca.MX | float],
    params: JRSymbolicParams,
    dt: float,
) -> tuple[list[ca.SX | ca.MX], list[ca.SX | ca.MX]]:
    """Roll out the dynamics over a horizon.

    Parameters
    ----------
    x0_history : Sequence[ca.SX | ca.MX]
        Initial state history list.
    controls : Sequence[ca.SX | ca.MX | float]
        Sequence of control inputs over the horizon.
    params : JRSymbolicParams
        Symbolic parameters for the network.
    dt : float
        Integration time step.

    Returns
    -------
    tuple[list[ca.SX | ca.MX], list[ca.SX | ca.MX]]
        A tuple containing the sequence of states and the sequence of LFP outputs.
    """
    X_seq = []
    Y_seq = []

    current_history = list(x0_history)

    for u in controls:
        x_next = heun_step(current_history, u, params, dt)
        X_seq.append(x_next)
        Y_seq.append(lfp(x_next))

        current_history.insert(0, x_next)
        current_history.pop()

    return X_seq, Y_seq
