from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import osqp
import scipy.sparse as sp

from neuro.geometry import sensor_positions_mm
from neuro.stimulation.analytical import compute_gamma
from neuro.stimulation.base import StimulationModel, check_n_controls
from neuro.stimulation.roast_3d import load_roast_3d_field_projection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.connectome import Connectome
    from neuro.types import FloatArray, IntArray, StrArray

_logger = logging.getLogger(__name__)

_MIN_KCL_ELECTRODES = 2
_CURRENT_ZERO_TOL = 1e-5


class SelectedMontageStim(StimulationModel):
    """Stimulation model applying regional drives from a sparse subset of electrodes."""

    def __init__(self, L_subset: FloatArray, active_labels: StrArray) -> None:
        """Initialize with sub-projection matrix of shape ``(n_nodes, n_active)``."""
        self.L_subset = np.asarray(L_subset, dtype=np.float64)
        self.control_labels = np.asarray(active_labels, dtype=np.str_)
        self.n_controls = len(self.control_labels)

    def project(self, u: FloatArray) -> FloatArray:
        """Compute regional somatic drive of shape ``(n_nodes,)``."""
        check_n_controls(u, self.n_controls)
        return self.L_subset @ u


@dataclass(frozen=True, slots=True)
class SparseMontageCandidate:
    """A sparse stimulation montage identified by L1 screening and unregularized refitting."""

    active_indices: IntArray
    active_labels: StrArray
    n_active: int
    lam: float
    u_l1: FloatArray
    u_refit: FloatArray
    full_u_refit: FloatArray
    mse_l1: float
    mse_refit: float
    s_refit: FloatArray
    kcl_residual: float


def build_target_field(
    connectome: Connectome,
    target_regions: Sequence[str],
    s0: float = 0.8,
) -> FloatArray:
    """Regional target drive vector of shape ``(n_nodes,)``, with -s0 at targets and 0 elsewhere."""
    s_star = np.zeros(len(connectome.region_labels), dtype=np.float64)
    for region in target_regions:
        if region in connectome.region_index:
            s_star[connectome.region_index[region]] = -abs(s0)
    return s_star


def solve_l1_montage_qp(  # noqa: PLR0913 -- solver takes bounds and convergence tolerances
    L: FloatArray,
    s_star: FloatArray,
    lam: float,
    *,
    i_max: float = 2.0,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
) -> FloatArray:
    """Solve L1-regularized QP for sparse Control Currents u of shape ``(n_channels,)``.

    Minimizes ``0.5 * ||L u - s*||_2^2 + lam * ||u||_1`` subject to ``sum(u) = 0``
    and ``|u_p| <= i_max`` via non-negative variable splitting in OSQP.
    """
    n_nodes, n_channels = L.shape
    if s_star.shape != (n_nodes,):
        msg = f"s_star shape {s_star.shape} does not match n_nodes {n_nodes}"
        raise ValueError(msg)

    h = L.T @ L
    l_s = L.T @ s_star

    p_dense = np.block([[h, -h], [-h, h]])
    q = np.concatenate([-l_s + lam, l_s + lam])

    a_kcl = np.concatenate([np.ones(n_channels), -np.ones(n_channels)]).reshape(1, 2 * n_channels)
    a_bounds = np.eye(2 * n_channels)
    a_dense = np.vstack([a_kcl, a_bounds])

    l_bound = np.concatenate([[0.0], np.zeros(2 * n_channels)])
    u_bound = np.concatenate([[0.0], np.full(2 * n_channels, i_max)])

    p_sparse = sp.triu(sp.csc_matrix(p_dense), format="csc")
    a_sparse = sp.csc_matrix(a_dense)

    prob = osqp.OSQP()
    prob.setup(
        P=p_sparse,
        q=q,
        A=a_sparse,
        l=l_bound,
        u=u_bound,
        verbose=False,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=10000,
    )
    res = prob.solve(raise_error=False)
    if res.x is None:
        return np.zeros(n_channels, dtype=np.float64)

    z = np.asarray(res.x, dtype=np.float64)
    u = z[:n_channels] - z[n_channels:]
    u[np.abs(u) < _CURRENT_ZERO_TOL] = 0.0
    return u


def refit_active_montage_qp(  # noqa: PLR0913 -- solver takes bounds and convergence tolerances
    L: FloatArray,
    s_star: FloatArray,
    active_indices: IntArray | Sequence[int],
    *,
    i_max: float = 2.0,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
) -> FloatArray:
    """Solve unregularized QP for active electrodes u_active of shape ``(n_active,)``.

    Minimizes ``0.5 * ||L_S u_S - s*||_2^2`` subject to ``sum(u_S) = 0``
    and ``|u_{S, p}| <= i_max``.
    """
    indices = np.asarray(active_indices, dtype=np.intp)
    k = len(indices)
    if k < _MIN_KCL_ELECTRODES:
        return np.zeros(k, dtype=np.float64)

    l_s = L[:, indices]
    p_dense = l_s.T @ l_s
    q = -l_s.T @ s_star

    a_kcl = np.ones((1, k))
    a_bounds = np.eye(k)
    a_dense = np.vstack([a_kcl, a_bounds])

    l_bound = np.concatenate([[0.0], np.full(k, -i_max)])
    u_bound = np.concatenate([[0.0], np.full(k, i_max)])

    p_sparse = sp.triu(sp.csc_matrix(p_dense), format="csc")
    a_sparse = sp.csc_matrix(a_dense)

    prob = osqp.OSQP()
    prob.setup(
        P=p_sparse,
        q=q,
        A=a_sparse,
        l=l_bound,
        u=u_bound,
        verbose=False,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=10000,
    )
    res = prob.solve(raise_error=False)
    if res.x is None:
        return np.zeros(k, dtype=np.float64)

    return np.asarray(res.x, dtype=np.float64)


def sweep_sparse_montages(  # noqa: PLR0913 -- sweep accepts tuning grid bounds and thresholds
    L: FloatArray,
    s_star: FloatArray,
    channel_labels: StrArray,
    *,
    min_electrodes: int = 2,
    max_electrodes: int = 5,
    i_max: float = 2.0,
    lam_min: float = 1e-4,
    lam_max: float = 1e1,
    n_lambdas: int = 40,
    threshold: float = 1e-3,
) -> list[SparseMontageCandidate]:
    """Sweep lambda to discover distinct active electrode montages within budget."""
    lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambdas)
    candidates: list[SparseMontageCandidate] = []
    seen_supports: set[tuple[int, ...]] = set()

    n_channels = L.shape[1]

    for lam in lambdas:
        u_l1 = solve_l1_montage_qp(L, s_star, lam=float(lam), i_max=i_max)
        active = np.where(np.abs(u_l1) > threshold)[0]
        k = len(active)

        if min_electrodes <= k <= max_electrodes:
            support_key = tuple(sorted(int(idx) for idx in active))
            if support_key not in seen_supports:
                seen_supports.add(support_key)

                u_refit = refit_active_montage_qp(L, s_star, active, i_max=i_max)
                full_u_refit = np.zeros(n_channels, dtype=np.float64)
                full_u_refit[active] = u_refit

                s_l1 = L @ u_l1
                s_refit = L @ full_u_refit

                mse_l1 = float(np.mean((s_l1 - s_star) ** 2))
                mse_refit = float(np.mean((s_refit - s_star) ** 2))
                kcl_res = float(np.abs(np.sum(u_refit)))

                candidates.append(
                    SparseMontageCandidate(
                        active_indices=np.asarray(active, dtype=np.intp),
                        active_labels=channel_labels[active],
                        n_active=k,
                        lam=float(lam),
                        u_l1=u_l1,
                        u_refit=u_refit,
                        full_u_refit=full_u_refit,
                        mse_l1=mse_l1,
                        mse_refit=mse_refit,
                        s_refit=s_refit,
                        kcl_residual=kcl_res,
                    )
                )

    candidates.sort(key=lambda c: (c.n_active, c.mse_refit))
    return candidates


def load_field_projection_matrix(
    path: str | Path,
    connectome: Connectome,
    min_channels_for_roast: int = 10,
) -> tuple[FloatArray, StrArray, bool, str]:
    """Load regional Field Projection matrix of shape ``(n_nodes, n_channels)``."""
    p = Path(path)
    if p.exists():
        try:
            projection_e, channel_labels, file_regions, normals = load_roast_3d_field_projection(p)
            if len(channel_labels) >= min_channels_for_roast and np.array_equal(file_regions, connectome.region_labels):
                polarization_length_mm = 0.35
                gamma = polarization_length_mm * np.einsum("kid,id->ki", projection_e, normals)
                l_stim = gamma.T
                desc = f"ROAST 3D FEM ({len(channel_labels)} channels) from {p.name}"
                return l_stim, channel_labels, False, desc
        except Exception as exc:  # noqa: BLE001 -- fallback gracefully if file is incompatible
            _logger.debug("ROAST loading failed, falling back to analytical model: %s", exc)

    labels, _ = sensor_positions_mm()
    gamma = compute_gamma(connectome.centres, list(labels))
    l_stim = gamma.T
    desc = f"Analytical Coulomb volume potential (62 scalp channels) [Fallback: {p.name} has < {min_channels_for_roast} channels]"
    return l_stim, np.asarray(labels, dtype=np.str_), True, desc
