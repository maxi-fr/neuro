from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Self

import casadi as ca
import numpy as np
from scipy.signal.windows import hann

from neuro.spectral import LOG_FLOOR, PsdEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.types import FloatArray, SymbolicModel


def _l1_epigraph(u_vars: list[ca.MX], w_l1: float) -> tuple[list[ca.MX], ca.MX, ca.MX]:
    """Epigraph reformulation of the horizon-mean L1 penalty ``(w_l1 / H) * sum(|u_k|)`` into linear terms.

    A slack ``t_k`` is added per control node so the non-smooth L1 penalty becomes a *linear*
    cost ``(w_l1 / len(u_vars)) * sum(t_k)`` plus the linear inequalities ``t_k >= |u_k|`` (i.e.
    ``t_k - u_k >= 0`` and ``t_k + u_k >= 0``). Keeping the objective quadratic lets ``ca.qpsol``
    extract a constant Hessian (OSQP/qpOASES) and keeps the IPOPT graph smooth; the active-set QP
    can then drive controls to exact zero. Returns the slack variables, the (linear) cost term,
    and the stacked ``>= 0`` inequalities ``[t_k - u_k, t_k + u_k]``.
    """
    slacks = [ca.MX.sym(f"t_{k}", u.numel()) for k, u in enumerate(u_vars)]
    cost = (w_l1 / len(u_vars)) * ca.sum1(ca.vertcat(*slacks))
    g = []
    for t, u in zip(slacks, u_vars, strict=True):
        g += [t - u, t + u]
    return slacks, cost, ca.vertcat(*g)


def _sum_to_zero(u_vars: list[ca.MX]) -> ca.MX:
    """Kirchhoff current-law equality: the per-electrode currents sum to zero at each step."""
    return ca.vertcat(*[ca.sum1(u) for u in u_vars])


def _spectral_hinge_cost(y_nodes: list[ca.MX], envelope: PsdEnvelope, horizon: int) -> ca.MX:
    """Mean squared one-sided log excess of the predicted spectrum over ``envelope``.

    CasADi has no FFT, so the periodogram is an explicit DFT matrix product. The reduction is a mean
    over ``(window, channel, bin)`` -- never over windows alone -- so a hot sub-window cannot be
    cancelled by a cold one and ``w_psd`` stays independent of the window count.
    """
    n_ch, n_bins = envelope.power.shape
    length, hop = envelope.window, envelope.hop
    if horizon < length:
        msg = f"horizon ({horizon}) is shorter than the envelope window ({length})"
        raise ValueError(msg)
    n_model_ch = y_nodes[0].shape[0]
    if n_ch != n_model_ch:
        msg = f"envelope has {n_ch} channels but the model outputs {n_model_ch}; the cost must not subset channels"
        raise ValueError(msg)
    n_windows = (horizon - length) // hop + 1

    w_hann = hann(length, sym=False)
    bins = np.arange(n_bins)[:, None]
    samples = np.arange(length)
    dft_cos = np.cos(2 * np.pi * bins * samples / length).T  # (length, n_bins)
    dft_sin = -np.sin(2 * np.pi * bins * samples / length).T

    # One-sided: every bin carries its negative-frequency twin, except DC and (even length) Nyquist.
    fold = np.full(n_bins, 2.0)
    fold[0] = 1.0
    if length % 2 == 0:
        fold[-1] = 1.0
    scale = fold / (envelope.fs * np.sum(w_hann**2))

    log_ref = ca.MX(np.log(envelope.power))
    taper = ca.repmat(ca.MX(w_hann.reshape(1, length)), n_ch, 1)
    scale_row = ca.repmat(ca.MX(scale.reshape(1, n_bins)), n_ch, 1)

    y_all = ca.horzcat(*y_nodes)  # (n_ch, horizon)
    total_hinge = ca.MX(0)
    for m in range(n_windows):
        y_win = y_all[:, m * hop : m * hop + length]
        y_detrended = y_win - ca.repmat(ca.sum2(y_win) / length, 1, length)
        y_tapered = y_detrended * taper
        power = ((y_tapered @ ca.MX(dft_cos)) ** 2 + (y_tapered @ ca.MX(dft_sin)) ** 2) * scale_row
        hinge = ca.fmax(0.0, ca.log(power + LOG_FLOOR) - log_ref)
        total_hinge = total_hinge + ca.sum1(ca.sum2(hinge**2))

    return total_hinge / (n_windows * n_ch * n_bins)


def _rollout_cost(  # noqa: PLR0913
    model: SymbolicModel,
    *,
    get_phi: Callable[[int], ca.MX],
    u_vars: list[ca.MX],
    n_segments: int,
    horizon: int,
    shooting_depth: int,
    w_y: float,
    w_y_terminal: float | None,
    w_u: float,
) -> tuple[ca.MX, list[ca.MX], list[ca.MX]]:
    """Roll the model over the horizon; returns the horizon-mean stagewise cost, defects and outputs."""
    cost: ca.MX = ca.MX(0)
    defects: list[ca.MX] = []
    y_nodes: list[ca.MX] = []
    for k in range(n_segments + 1):
        x_curr = get_phi(k)
        for step in range(k * shooting_depth, min((k + 1) * shooting_depth, horizon)):
            u_curr = u_vars[step]
            x_next = model.f_step(x_curr, u_curr)
            y_next = model.f_out(x_next)
            y_nodes.append(y_next)

            is_terminal = step == horizon - 1 and w_y_terminal is not None
            w_y_step = w_y_terminal if is_terminal else w_y
            cost = cost + w_y_step * ca.sumsqr(y_next) + w_u * ca.sumsqr(u_curr)
            x_curr = x_next

        if k < n_segments:
            defects.append(x_curr - get_phi(k + 1))
    return cost / horizon, defects, y_nodes


@dataclasses.dataclass(frozen=True)
class MPCNlp:
    """Symbolic NLP formulation and decision variable / constraint bounds."""

    nlp: dict[str, ca.MX]
    lbx: FloatArray
    ubx: FloatArray
    lbg: FloatArray | float
    ubg: FloatArray | float

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        model: SymbolicModel,
        *,
        horizon: int,
        shooting_depth: int,
        n_controls: int,
        u_max: FloatArray,
        w_y: float,
        w_y_terminal: float | None = None,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        w_psd: float = 0.0,
        psd_envelope: PsdEnvelope | None = None,
    ) -> Self:
        """Build the PCMS multiple-shooting symbolic NLP graph and bounds."""
        n_state = model.state_shape[0]
        n_ctrl, h = n_controls, horizon
        D = shooting_depth

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        n_segments = (h - 1) // D
        phi_vars = [ca.MX.sym(f"phi_{k}", n_state) for k in range(1, n_segments + 1)]

        def get_phi(idx: int) -> ca.MX:
            return x0_p if idx == 0 else phi_vars[idx - 1]

        cost, defects, y_nodes = _rollout_cost(
            model,
            get_phi=get_phi,
            u_vars=u_vars,
            n_segments=n_segments,
            horizon=h,
            shooting_depth=D,
            w_y=w_y,
            w_y_terminal=w_y_terminal,
            w_u=w_u,
        )

        if w_psd > 0:
            if psd_envelope is None:
                msg = "psd_envelope must be provided when w_psd > 0"
                raise ValueError(msg)
            cost = cost + w_psd * _spectral_hinge_cost(y_nodes, psd_envelope, h)

        x_parts = [*u_vars, *phi_vars]
        g_parts = [*defects, _sum_to_zero(u_vars)]
        n_eq = len(defects) * n_state + h
        n_phi_vars = len(phi_vars) * n_state
        lbx = np.concatenate([np.tile(-u_max, h), np.full(n_phi_vars, -np.inf)])
        ubx = np.concatenate([np.tile(u_max, h), np.full(n_phi_vars, np.inf)])

        if w_u_l1 > 0:
            slacks, l1_cost, l1_g = _l1_epigraph(u_vars, w_u_l1)
            cost = cost + l1_cost
            x_parts += slacks
            g_parts.append(l1_g)
            n_l1 = l1_g.numel()
            lbg = np.concatenate([np.zeros(n_eq), np.zeros(n_l1)])
            ubg = np.concatenate([np.zeros(n_eq), np.full(n_l1, np.inf)])
            lbx = np.concatenate([lbx, np.zeros(h * n_ctrl)])
            ubx = np.concatenate([ubx, np.full(h * n_ctrl, np.inf)])
        else:
            lbg = ubg = 0.0

        x_nlp = ca.vertcat(*x_parts)
        g_nlp = ca.vertcat(*g_parts) if g_parts else ca.MX(0)
        nlp = {"x": x_nlp, "f": cost, "g": g_nlp, "p": x0_p}
        return cls(nlp=nlp, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
