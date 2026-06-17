# ruff: noqa: N806
"""CasADi symbolic implementation of the Jansen-Rit neural mass model.

Provides differentiable RHS and Heun-step CasADi Functions suitable for
embedding in optimization problems (MPC, NLP). Noise is not modelled -- the
stochastic ``sigma`` term lives in ``jansen_rit.py`` and has no place in a
differentiable shooting formulation.

State layout
------------
``x`` is a flat vector of length ``6 * n_nodes`` following the same C-order
convention as the estimator's augmented-state encoding::

    x[k * n_nodes : (k+1) * n_nodes]  =  state variable x_{k+1} for all nodes

For a single node (``n_nodes = 1``) this reduces to ``[x1, x2, x3, x4, x5, x6]``,
identical to the single-node output of ``_jr_rhs_jit``.  For ``n_nodes = N``:

    x[0*N : 1*N]  =  x1  (pyramidal)
    x[1*N : 2*N]  =  x2  (excitatory)
    x[2*N : 3*N]  =  x3  (inhibitory)
    x[3*N : 4*N]  =  x4
    x[4*N : 5*N]  =  x5
    x[5*N : 6*N]  =  x6

This matches ``np.reshape(x_aug[:6*N], (6, N)).flatten()`` (C order) from the
estimator, so CasADi outputs can be compared directly against ``_fx_step_jit``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from neuro.jansen_rit import JansenRitParams


def build_jr_rhs(params: JansenRitParams, n_nodes: int = 1) -> ca.Function:
    """Build a CasADi symbolic right-hand side for the Jansen-Rit ODE.

    Parameters
    ----------
    params:
        Jansen-Rit parameters.  ``params.A`` may be a scalar or a 1-D array of
        length ``n_nodes``; all other parameters must be scalars.
    n_nodes:
        Number of network nodes.

    Returns
    -------
    ca.Function
        ``jr_rhs(x, coupling, u_tes) -> dx``
        where every argument/result is a vector of the sizes shown below:

        * ``x``        - ``(6 * n_nodes,)`` current state
        * ``coupling`` - ``(n_nodes,)``      network coupling term
        * ``u_tes``    - ``(n_nodes,)``      tES input on pyramidal population
        * ``dx``       - ``(6 * n_nodes,)``  time derivative
    """
    x = ca.SX.sym("x", 6 * n_nodes)
    coupling = ca.SX.sym("coupling", n_nodes)
    u_tes = ca.SX.sym("u_tes", n_nodes)

    A_arr = (
        np.full(n_nodes, np.asarray(params.A, dtype=np.float64).item())
        if np.isscalar(params.A)
        else np.asarray(params.A, dtype=np.float64)
    )

    B, a, b = params.B, params.a, params.b
    C1, C2, C3, C4 = params.C1, params.C2, params.C3, params.C4
    e0, v0, r, mean_input = params.e0, params.v0, params.r, params.mean_input

    dx_parts: list[ca.SX] = []
    for i in range(n_nodes):
        x1_i = x[0 * n_nodes + i]
        x2_i = x[1 * n_nodes + i]
        x3_i = x[2 * n_nodes + i]
        x4_i = x[3 * n_nodes + i]
        x5_i = x[4 * n_nodes + i]
        x6_i = x[5 * n_nodes + i]
        A_i = float(A_arr[i])

        sig_out = 2.0 * e0 / (1.0 + ca.exp(r * (v0 - (x2_i - x3_i + u_tes[i]))))
        sig_exc = 2.0 * e0 / (1.0 + ca.exp(r * (v0 - C1 * x1_i)))
        sig_inh = 2.0 * e0 / (1.0 + ca.exp(r * (v0 - C3 * x1_i)))

        dx_parts.append(x4_i)  # dx1
        dx_parts.append(x5_i)  # dx2
        dx_parts.append(x6_i)  # dx3
        dx_parts.append(A_i * a * sig_out - 2.0 * a * x4_i - a**2 * x1_i)  # dx4
        dx_parts.append(A_i * a * (mean_input + C2 * sig_exc + coupling[i]) - 2.0 * a * x5_i - a**2 * x2_i)  # dx5
        dx_parts.append(B * b * C4 * sig_inh - 2.0 * b * x6_i - b**2 * x3_i)  # dx6

    # Reassemble in variable-first order: [dx1_0,..,dx1_{N-1}, dx2_0,.., dx6_{N-1}]
    dx = ca.vertcat(*[dx_parts[6 * i + k] for k in range(6) for i in range(n_nodes)])

    return ca.Function(
        "jr_rhs",
        [x, coupling, u_tes],
        [dx],
        ["x", "coupling", "u_tes"],
        ["dx"],
    )


def build_heun_step(params: JansenRitParams, dt: float, n_nodes: int = 1) -> ca.Function:
    """Build a CasADi deterministic fixed-step Heun integrator.

    Implements ``x_next = x + 0.5 * dt * (f(x) + f(x + dt * f(x)))``, the
    deterministic (noise-free) Heun scheme -- identical to :func:`heun_step`
    from ``jansen_rit.py`` when called with ``sigma = 0``.

    Parameters
    ----------
    params:
        Jansen-Rit parameters.
    dt:
        Integration step in seconds.
    n_nodes:
        Number of network nodes.

    Returns
    -------
    ca.Function
        ``heun_step(x, coupling, u_tes) -> x_next``
        with the same argument sizes as :func:`build_jr_rhs`.
    """
    rhs = build_jr_rhs(params, n_nodes)

    x = ca.SX.sym("x", 6 * n_nodes)
    coupling = ca.SX.sym("coupling", n_nodes)
    u_tes = ca.SX.sym("u_tes", n_nodes)

    f0 = rhs(x, coupling, u_tes)
    x_pred = x + dt * f0
    f1 = rhs(x_pred, coupling, u_tes)
    x_next = x + 0.5 * dt * (f0 + f1)

    return ca.Function(
        "heun_step",
        [x, coupling, u_tes],
        [x_next],
        ["x", "coupling", "u_tes"],
        ["x_next"],
    )
