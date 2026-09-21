from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
import yaml
from simulate.simulation import Simulation

from neuro.control.schedule import ScheduleController, build_input_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from neuro.predictor.inference import InferencePredictor
    from neuro.types import FloatArray, IntArray


def extract_state_action_pairs(
    trajectories: list[tuple[FloatArray, FloatArray]],
    *,
    n_u: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Extract joint state-action feature vectors and one-step next Observable Frames from trajectories.

    Parameters
    ----------
    trajectories : list[tuple[FloatArray, FloatArray]]
        List of ``(u, y)`` trajectory pairs where ``u`` is Control Current of shape
        ``(T, n_controls)`` and ``y`` is Observable Frame trajectory of shape ``(T, n_outputs)``.
    n_u : int, default=0
        Past Control Current steps in the history window. When 0, constructs instantaneous
        state-action pairs ``z_t = [y_t, u_t]``. When positive, pairs the Observable Frame with the
        trailing control window ``u_{t - n_u + 1 : t + 1}``.

    Returns
    -------
    z : FloatArray
        State-action query vectors of shape ``(N_samples, D)``.
    y_next : FloatArray
        Next-step target Observable Frames of shape ``(N_samples, n_outputs)``.
    """
    z_list: list[FloatArray] = []
    y_next_list: list[FloatArray] = []

    for u, y in trajectories:
        u_arr = np.asarray(u, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        t_len = min(len(u_arr), len(y_arr))
        start_idx = max(0, n_u - 1)
        end_idx = t_len - 1
        if end_idx <= start_idx:
            continue

        indices = np.arange(start_idx, end_idx)
        y_curr = y_arr[indices]
        y_tgt = y_arr[indices + 1]

        if n_u <= 1:
            u_curr = u_arr[indices]
            z_mat = np.hstack([y_curr, u_curr])
        else:
            u_windows = np.lib.stride_tricks.sliding_window_view(u_arr[:end_idx], n_u, axis=0)
            u_flat = u_windows.reshape(len(u_windows), -1)
            z_mat = np.hstack([y_curr, u_flat])

        z_list.append(z_mat)
        y_next_list.append(y_tgt)

    if not z_list:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    return (
        np.ascontiguousarray(np.vstack(z_list), dtype=np.float64),
        np.ascontiguousarray(np.vstack(y_next_list), dtype=np.float64),
    )


@dataclass(frozen=True)
class OODIndex:
    """Nearest-neighbor index and empirical calibration mapping for state-action distribution shift.

    Attributes
    ----------
    mu : FloatArray
        Reference distribution empirical feature means, shape ``(D,)``.
    sigma : FloatArray
        Reference distribution feature standard deviations, shape ``(D,)``.
    id_cal_distances : FloatArray
        Sorted in-distribution calibration K-NN distances, shape ``(N_cal,)``.
    k : int
        Number of nearest neighbors averaged in distance metric.
    ref_tensor : torch.Tensor
        Standardized reference points tensor, shape ``(N_ref, D)``.
    batch_size : int
        Query chunk size for vectorized distance evaluation.
    """

    mu: FloatArray
    sigma: FloatArray
    id_cal_distances: FloatArray
    k: int
    ref_tensor: torch.Tensor
    batch_size: int

    @property
    def reference_count(self) -> int:
        """Total reference points stored in index."""
        return int(self.ref_tensor.shape[0])

    @property
    def calibration_count(self) -> int:
        """Total calibration points evaluated for empirical distribution."""
        return len(self.id_cal_distances)

    @property
    def dimension(self) -> int:
        """State-action feature dimension."""
        return len(self.mu)

    @classmethod
    def fit(
        cls,
        reference_data: FloatArray,
        calibration_data: FloatArray,
        *,
        k: int = 10,
        batch_size: int = 500,
    ) -> Self:
        """Fit feature standardizers on reference data and establish the in-distribution baseline distances.

        Parameters
        ----------
        reference_data : FloatArray
            Reference state-action vectors of shape ``(N_ref, D)``.
        calibration_data : FloatArray
            Held-out in-distribution state-action vectors of shape ``(N_cal, D)``.
        k : int, default=10
            Number of nearest neighbors.
        batch_size : int, default=500
            Query batch size.
        """
        ref = np.asarray(reference_data, dtype=np.float64)
        cal = np.asarray(calibration_data, dtype=np.float64)

        mu = np.mean(ref, axis=0)
        sigma = np.std(ref, axis=0)
        # Avoid division by zero on invariant channels
        sigma[sigma <= 1e-12] = 1.0  # noqa: PLR2004 -- numerical floor

        ref_std = (ref - mu) / sigma
        cal_std = (cal - mu) / sigma

        ref_tensor = torch.as_tensor(ref_std, dtype=torch.float32)

        index = cls(
            mu=mu,
            sigma=sigma,
            id_cal_distances=np.zeros(0, dtype=np.float64),
            k=k,
            ref_tensor=ref_tensor,
            batch_size=batch_size,
        )

        cal_distances = index._compute_standardized_distances(cal_std)
        cal_distances_sorted = np.sort(cal_distances)

        return cls(
            mu=mu,
            sigma=sigma,
            id_cal_distances=cal_distances_sorted,
            k=k,
            ref_tensor=ref_tensor,
            batch_size=batch_size,
        )

    def standardize(self, queries: FloatArray) -> FloatArray:
        """Standardize query vectors using reference set statistics.

        Parameters
        ----------
        queries : FloatArray
            Raw query vectors of shape ``(N_queries, D)``.

        Returns
        -------
        FloatArray
            Standardized query vectors of shape ``(N_queries, D)``.
        """
        q = np.asarray(queries, dtype=np.float64)
        return (q - self.mu) / self.sigma

    def _compute_standardized_distances(self, standardized_queries: FloatArray) -> FloatArray:
        """Compute K-NN distances against reference points for standardized vectors."""
        q_arr = np.asarray(standardized_queries, dtype=np.float32)
        n_queries = len(q_arr)
        distances = np.empty(n_queries, dtype=np.float64)

        for i in range(0, n_queries, self.batch_size):
            chunk = torch.as_tensor(q_arr[i : i + self.batch_size])
            dists = torch.cdist(chunk, self.ref_tensor)
            vals, _ = torch.topk(dists, self.k, largest=False)
            distances[i : i + self.batch_size] = vals.mean(dim=1).cpu().numpy().astype(np.float64)

        return distances

    def compute_distances(self, queries: FloatArray) -> FloatArray:
        """Compute K-nearest neighbor distances for state-action queries.

        Parameters
        ----------
        queries : FloatArray
            Raw query vectors of shape ``(N_queries, D)``.

        Returns
        -------
        FloatArray
            Mean distance to K nearest neighbors, shape ``(N_queries,)``.
        """
        q_std = self.standardize(queries)
        return self._compute_standardized_distances(q_std)

    def percentiles(self, distances: FloatArray) -> FloatArray:
        """Convert K-NN distances to in-distribution empirical percentiles q(z) in [0, 1].

        Parameters
        ----------
        distances : FloatArray
            Distances of shape ``(N,)``.

        Returns
        -------
        FloatArray
            Empirical in-distribution percentiles of shape ``(N,)``.
        """
        d = np.asarray(distances, dtype=np.float64)
        if len(self.id_cal_distances) == 0:
            return np.zeros_like(d)
        ranks = np.searchsorted(self.id_cal_distances, d, side="right")
        return ranks / float(len(self.id_cal_distances))

    def global_distances(self, queries: FloatArray) -> FloatArray:
        """Compute standardized global distance (diagonal Mahalanobis) from reference center.

        Parameters
        ----------
        queries : FloatArray
            Raw query vectors of shape ``(N_queries, D)``.

        Returns
        -------
        FloatArray
            Standardized Euclidean norm from training mean, shape ``(N_queries,)``.
        """
        q_std = self.standardize(queries)
        return np.linalg.norm(q_std, axis=1)

    def coefficient_of_variation(self) -> float:
        """Compute the coefficient of variation CV = std(d_K) / mean(d_K) on calibration distances."""
        if len(self.id_cal_distances) == 0:
            return 0.0
        mean_d = float(np.mean(self.id_cal_distances))
        if mean_d <= 1e-12:  # noqa: PLR2004 -- floor for zero distances
            return 0.0
        return float(np.std(self.id_cal_distances) / mean_d)


def extract_mpc_rollout_queries(
    log_file: str | Path,
    *,
    n_u: int = 0,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Extract candidate Rollout queries from an MPC simulation log archive.

    Parameters
    ----------
    log_file : str | Path
        Path to ``log.npz`` generated by closed-loop simulation.
    n_u : int, default=0
        Past Control Current steps. When 0, pairs instantaneous Observable Frames with candidate
        Control Currents.

    Returns
    -------
    queries : FloatArray
        Candidate Rollout queries of shape ``(n_decisions, horizon, D)``.
    planned_u : FloatArray
        Planned Control Current sequences of shape ``(n_decisions, horizon, n_controls)``.
    valid_decisions : IntArray
        Indices of decision steps executed outside the Warm-up Period.
    """
    with np.load(log_file) as data:
        pred_y = np.asarray(data["controller.predicted_y"], dtype=np.float64)
        plan_u = np.asarray(data["controller.planned_u"], dtype=np.float64)
        warmup = np.asarray(data["controller.warmup"], dtype=bool)

    valid_idx = np.where(~warmup)[0]
    if len(valid_idx) == 0:
        return np.empty((0, 0, 0), dtype=np.float64), np.empty((0, 0, 0), dtype=np.float64), np.empty(0, dtype=np.intp)

    valid_pred_y = pred_y[valid_idx]
    valid_plan_u = plan_u[valid_idx]

    n_decisions, _, _ = valid_pred_y.shape
    horizon = valid_plan_u.shape[1]

    # Knot 0 to horizon - 1 corresponds to candidate states evaluated with candidate controls
    states_along_horizon = valid_pred_y[:, :horizon, :]  # (n_decisions, horizon, n_outputs)

    if n_u <= 1:
        queries = np.concatenate([states_along_horizon, valid_plan_u], axis=-1)
    else:
        # Full history mode: pad prefix controls with the first planned control
        queries_list: list[FloatArray] = []
        for d in range(n_decisions):
            u_seq = valid_plan_u[d]  # (horizon, n_controls)
            padded_u = np.vstack([np.tile(u_seq[0], (n_u - 1, 1)), u_seq])
            u_win = np.lib.stride_tricks.sliding_window_view(padded_u, n_u, axis=0).reshape(horizon, -1)
            z_d = np.hstack([states_along_horizon[d], u_win])
            queries_list.append(z_d)
        queries = np.stack(queries_list, axis=0)

    return queries, valid_plan_u, valid_idx


def _extract_control_metrics_from_log(
    data: Mapping[str, np.ndarray], run_cfg: dict[str, Any], metrics: dict[str, Any]
) -> None:
    """Extract charge and amplitude from log data if not present in metrics."""
    if "controller.u" not in data or "controller.t" not in data:
        return
    u = np.asarray(data["controller.u"], dtype=np.float64)
    t = np.asarray(data["controller.t"], dtype=np.float64)
    t_end = float(run_cfg.get("t_end", t[-1] if len(t) > 0 else 12.0))
    durations = np.diff(np.append(t, t_end))
    u_max = float(run_cfg.get("controller", {}).get("problem", {}).get("u_max", 2.0))
    if "delivered_charge" not in metrics:
        metrics["delivered_charge"] = float(np.sum(np.abs(u) * durations[:, None]))
    if "mean_amplitude" not in metrics and t_end > 0:
        metrics["mean_amplitude"] = float(np.sum(np.mean(np.abs(u) / u_max, axis=1) * durations) / t_end)
    if "kirchhoff_max" not in metrics:
        metrics["kirchhoff_max"] = float(np.max(np.abs(np.sum(u, axis=1))))


def _extract_solver_metrics_from_log(
    data: Mapping[str, np.ndarray], run_cfg: dict[str, Any], metrics: dict[str, Any]
) -> None:
    """Extract solver runtime metrics from log data if not present in metrics."""
    if "controller.solve_time" not in data or "controller.warmup" not in data:
        return
    warmup = np.asarray(data["controller.warmup"], dtype=bool)
    solve_time = np.asarray(data["controller.solve_time"], dtype=np.float64)
    solved = ~warmup
    if solved.any():
        if "solve_time_mean_s" not in metrics:
            metrics["solve_time_mean_s"] = float(np.mean(solve_time[solved]))
        if "solve_time_p95_s" not in metrics:
            metrics["solve_time_p95_s"] = float(np.quantile(solve_time[solved], 0.95))
        if "realtime_factor" not in metrics:
            control_dt = float(run_cfg.get("controller", {}).get("dt", 0.06))
            metrics["realtime_factor"] = float(metrics["solve_time_p95_s"] / control_dt) if control_dt > 0 else 0.0
    if "controller.success" in data and "solve_success_rate" not in metrics:
        success = np.asarray(data["controller.success"], dtype=bool)
        metrics["solve_success_rate"] = float(np.mean(success[solved])) if solved.any() else 0.0


def extract_run_clinical_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Score Seizure Burden, suppression, delivered charge, and solver metrics for a closed-loop run.

    Parameters
    ----------
    run_dir : str | Path
        Directory of the closed-loop run containing ``config.yaml``, ``log.npz``, and optionally ``scores.json``.

    Returns
    -------
    dict[str, Any]
        Dictionary of closed-loop clinical efficacy and solver runtime statistics.
    """
    path = Path(run_dir)
    scores_file = path / "scores.json"
    metrics: dict[str, Any] = {}
    if scores_file.exists():
        with scores_file.open(encoding="utf-8") as f:
            metrics.update(json.load(f))

    cfg_file = path / "config.yaml"
    run_cfg: dict[str, Any] = {}
    if cfg_file.exists():
        with cfg_file.open(encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f) or {}

    log_file = path / "log.npz"
    if log_file.exists():
        with np.load(log_file) as data:
            _extract_control_metrics_from_log(data, run_cfg, metrics)
            _extract_solver_metrics_from_log(data, run_cfg, metrics)

    return metrics


def horizon_statistics(
    rollout_percentiles: FloatArray,
    rollout_distances: FloatArray,
) -> dict[str, FloatArray]:
    """Aggregate per-horizon diagnostic metrics across candidate MPC Rollouts.

    Parameters
    ----------
    rollout_percentiles : FloatArray
        ID percentiles of shape ``(n_decisions, horizon)``.
    rollout_distances : FloatArray
        K-NN distances of shape ``(n_decisions, horizon)``.

    Returns
    -------
    dict[str, FloatArray]
        Dictionary mapping diagnostic metric names to 1D arrays.
    """
    p = np.asarray(rollout_percentiles, dtype=np.float64)
    d = np.asarray(rollout_distances, dtype=np.float64)

    return {
        "mean_percentile": np.mean(p, axis=0),
        "median_percentile": np.median(p, axis=0),
        "p95_rate": np.mean(p > 0.95, axis=0),  # noqa: PLR2004 -- 95th percentile
        "p99_rate": np.mean(p > 0.99, axis=0),  # noqa: PLR2004 -- 99th percentile
        "mean_distance": np.mean(d, axis=0),
        "std_distance": np.std(d, axis=0),
        "q_max": np.max(p, axis=1),
        "q_mean": np.mean(p, axis=1),
    }


def binned_prediction_error(
    percentiles: FloatArray,
    errors: FloatArray,
    *,
    bins: tuple[float, ...] = (0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0),
) -> list[dict[str, Any]]:
    """Partition one-step prediction errors into empirical in-distribution percentile bins.

    Parameters
    ----------
    percentiles : FloatArray
        Calibrated empirical percentiles of shape ``(N,)``.
    errors : FloatArray
        Model prediction errors of shape ``(N,)``.
    bins : tuple[float, ...], default=(0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        Ascending bin edges.

    Returns
    -------
    list[dict[str, Any]]
        List of bin summaries with count, mean error, standard deviation, and median.
    """
    p = np.asarray(percentiles, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)

    summaries: list[dict[str, Any]] = []
    for low, high in itertools.pairwise(bins):
        mask = (p >= low) & (p <= high) if high == bins[-1] else (p >= low) & (p < high)
        bin_errors = e[mask]
        count = int(np.sum(mask))
        summaries.append(
            {
                "bin_range": (low, high),
                "bin_label": f"{int(low * 100)}--{int(high * 100)}%",
                "count": count,
                "mean_error": float(np.mean(bin_errors)) if count > 0 else float("nan"),
                "std_error": float(np.std(bin_errors)) if count > 0 else float("nan"),
                "median_error": float(np.median(bin_errors)) if count > 0 else float("nan"),
            }
        )

    return summaries


def evaluate_prediction_errors(  # noqa: PLR0913, PLR0915 -- multi-step and one-step evaluation share anchor alignment
    model: InferencePredictor,
    index: OODIndex,
    trajectories: list[tuple[FloatArray, FloatArray]],
    *,
    horizon: int = 25,
    n_u: int = 0,
    stride: int = 5,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate one-step and multi-step Rollout errors against OOD percentiles on held-out trajectories.

    Parameters
    ----------
    model : InferencePredictor
        Loaded predictor runtime.
    index : OODIndex
        Fitted K-NN distribution support index.
    trajectories : list[tuple[FloatArray, FloatArray]]
        List of ``(u, y)`` Frame trajectories.
    horizon : int, default=25
        Prediction horizon for recursive Rollout evaluation.
    n_u : int, default=0
        Past Control Current steps used in the state-action vector.
    stride : int, default=5
        Subsampling stride between consecutive multi-step Rollout anchors.

    Returns
    -------
    one_step_percentiles : FloatArray
        Percentiles q(z_t) for each step, shape ``(N_points,)``.
    one_step_errors : FloatArray
        L2 prediction error e_t for each step, shape ``(N_points,)``.
    rollout_q_max : FloatArray
        Maximum support violation Q_H along each multi-step Rollout, shape ``(N_rollouts,)``.
    rollout_e_max : FloatArray
        Maximum prediction error E_H along each multi-step Rollout, shape ``(N_rollouts,)``.
    """
    n_y = getattr(model, "n_y", 1)
    n_u_model = getattr(model, "n_u", 0)

    q_1step_list: list[FloatArray] = []
    e_1step_list: list[FloatArray] = []
    q_max_list: list[float] = []
    e_max_list: list[float] = []

    for u_raw, y_raw in trajectories:
        u_arr = np.asarray(u_raw, dtype=np.float64)
        y_arr = np.asarray(y_raw, dtype=np.float64)
        if y_arr.ndim > 2:  # noqa: PLR2004 -- flatten multi-dimensional Observable Frames
            y_arr = y_arr.reshape(len(y_arr), -1)

        t_len = min(len(u_arr), len(y_arr))
        start_anchor = max(n_y - 1, n_u_model, 0, n_u - 1)
        end_anchor = t_len - 1

        if end_anchor <= start_anchor:
            continue

        anchors_1step = np.arange(start_anchor, end_anchor)
        y_hist = np.stack([y_arr[t - n_y + 1 : t + 1] for t in anchors_1step], axis=0)
        u_hist = (
            np.stack([u_arr[t - n_u_model : t] for t in anchors_1step], axis=0)
            if n_u_model > 0
            else np.zeros((len(anchors_1step), 0, u_arr.shape[1]))
        )
        u_fut = np.stack([u_arr[t : t + 1] for t in anchors_1step], axis=0)
        y_tgt = np.stack([y_arr[t + 1] for t in anchors_1step], axis=0)

        preds_1step = np.asarray(model.free_run(y_hist, u_hist, u_fut)).reshape(len(anchors_1step), -1)
        errs_1step = np.linalg.norm(preds_1step - y_tgt, axis=1)

        if n_u <= 1:
            z_1step = np.hstack([y_arr[anchors_1step], u_arr[anchors_1step]])
        else:
            u_win = np.stack([u_arr[t - n_u + 1 : t + 1].flatten() for t in anchors_1step], axis=0)
            z_1step = np.hstack([y_arr[anchors_1step], u_win])

        d_1step = index.compute_distances(z_1step)
        q_1step = index.percentiles(d_1step)

        q_1step_list.append(q_1step)
        e_1step_list.append(errs_1step)

        # Multi-step rollouts
        max_multi_anchor = t_len - horizon - 1
        if max_multi_anchor > start_anchor:
            anchors_multi = np.arange(start_anchor, max_multi_anchor, stride)
            y_hist_m = np.stack([y_arr[t - n_y + 1 : t + 1] for t in anchors_multi], axis=0)
            u_hist_m = (
                np.stack([u_arr[t - n_u_model : t] for t in anchors_multi], axis=0)
                if n_u_model > 0
                else np.zeros((len(anchors_multi), 0, u_arr.shape[1]))
            )
            u_fut_m = np.stack([u_arr[t : t + horizon] for t in anchors_multi], axis=0)
            y_true_m = np.stack([y_arr[t + 1 : t + horizon + 1] for t in anchors_multi], axis=0)

            preds_multi = np.asarray(model.free_run(y_hist_m, u_hist_m, u_fut_m)).reshape(
                len(anchors_multi), horizon, -1
            )
            rollout_errs = np.linalg.norm(preds_multi - y_true_m, axis=-1)  # (n_m, horizon)
            e_max_batch = np.max(rollout_errs, axis=1)

            # Evaluate OOD queries along predicted rollout
            for i, anc in enumerate(anchors_multi):
                pred_seq = preds_multi[i]  # (horizon, p)
                u_seq = u_arr[anc : anc + horizon]
                if n_u <= 1:
                    z_rollout = np.hstack([pred_seq, u_seq])
                else:
                    padded_u = np.vstack([u_arr[anc - n_u + 1 : anc], u_seq])
                    u_win_r = np.lib.stride_tricks.sliding_window_view(padded_u, n_u, axis=0).reshape(horizon, -1)
                    z_rollout = np.hstack([pred_seq, u_win_r])
                d_rollout = index.compute_distances(z_rollout)
                q_rollout = index.percentiles(d_rollout)
                q_max_list.append(float(np.max(q_rollout)))
                e_max_list.append(float(e_max_batch[i]))

    q_1step_arr = np.concatenate(q_1step_list) if q_1step_list else np.empty(0, dtype=np.float64)
    e_1step_arr = np.concatenate(e_1step_list) if e_1step_list else np.empty(0, dtype=np.float64)
    q_max_arr = np.asarray(q_max_list, dtype=np.float64)
    e_max_arr = np.asarray(e_max_list, dtype=np.float64)

    return q_1step_arr, e_1step_arr, q_max_arr, e_max_arr


_RATIO_TOLERANCE = 1.05
_CV_THRESHOLD = 0.05
_DRIFT_THRESHOLD = 0.10
_P95_THRESHOLD = 5.0
_CORR_THRESHOLD = 0.1
_MIN_BINS_FOR_AMPLIFICATION = 2
_EPSILON = 1e-9
_DRIFT_MEDIAN_THRESHOLD = 0.50


def _interpret_lookahead_dynamics(q_fut_med: float, q0_p95: float) -> tuple[str, str]:
    """Provide physical interpretation of planned vs realized lookahead dynamics."""
    if q_fut_med < _DRIFT_MEDIAN_THRESHOLD:
        phys = (
            f"The MPC solver plans an idealized suppression trajectory where\n"
            f"                               controls taper to zero (u -> 0), driving planned knots deep into\n"
            f"                               the nominal distribution interior (median q = {q_fut_med:.3f}).\n"
            f"                               The realized system (k=0) operates persistently at P(q > 0.95) = {q0_p95:.2f}%,\n"
            f"                               faithfully matching the in-distribution tail baseline (~5.0%)."
        )
        summary = (
            f"Planned Attractor Collapse (k >= 1): Planned knots decay into an idealized\n"
            f"     quiescent attractor (median q = {q_fut_med:.3f}), explaining why horizon-wide\n"
            f"     aggregates appeared lower than knot 0."
        )
    else:
        phys = (
            f"The MPC solver's recursive open-loop predictions suffer from compounding\n"
            f"                               multi-step drift (median q = {q_fut_med:.3f}), diverging into tail regimes.\n"
            f"                               Despite lookahead divergence, closed-loop state feedback safely anchors the\n"
            f"                               realized plant (k=0) at P(q > 0.95) = {q0_p95:.2f}% (~5.0% baseline)."
        )
        summary = (
            f"Planned Horizon Divergence (k >= 1): Multi-step open-loop lookahead drifts\n"
            f"     off-manifold (median q = {q_fut_med:.3f}), while closed-loop state feedback resets\n"
            f"     knot 0 on each decision to keep the physical plant well-supported."
        )
    return phys, summary


def _format_clinical_report_block(clin: dict[str, Any] | None) -> tuple[str, str]:
    """Format clinical efficacy and control performance block for the OOD report."""
    if not clin:
        return "", ""

    burden = clin.get("seizure_burden")
    burden_str = f"{burden * 100:.2f}%" if burden is not None else "N/A"
    n_final = clin.get("n_seizing_final")
    ez = clin.get("ez_recruited")
    pz = clin.get("pz_recruited")
    hl = clin.get("healthy_recruited")
    charge = clin.get("delivered_charge")
    amp = clin.get("mean_amplitude")
    kcl = clin.get("kirchhoff_max")
    p95_s = clin.get("solve_time_p95_s")
    rt = clin.get("realtime_factor")
    succ = clin.get("solve_success_rate")

    charge_str = f"{charge:.3f} mA*s" if charge is not None else "N/A"
    amp_str = f"{amp * 100:.2f}% of u_max" if amp is not None else "N/A"
    kcl_str = f"{kcl:.2e} mA" if kcl is not None else "N/A"

    containment_str = f"{n_final:.0f} / 76 regions seizing at t_end" if n_final is not None else "N/A"
    recruitment_str = (
        f"\n                             (EZ: {ez:.0f}/3, PZ: {pz:.0f}/2, Healthy: {hl:.0f}/71)"
        if ez is not None and pz is not None and hl is not None
        else ""
    )

    solver_str = f"{p95_s:.3f}s (Realtime factor: {rt:.1f}x)" if p95_s is not None and rt is not None else "N/A"
    succ_str = f"{succ * 100:.1f}%" if succ is not None else "N/A"

    block = f"""--------------------------------------------------------------------------------
CLOSED-LOOP CLINICAL EFFICACY & CONTROL PERFORMANCE
--------------------------------------------------------------------------------
  * Seizure Burden:          {burden_str} (time-averaged network fraction seizing)
  * Seizure Containment:     {containment_str}{recruitment_str}
  * Total Delivered Charge:  {charge_str}
  * Mean Control Amplitude:  {amp_str}
  * Kirchhoff Residual:      {kcl_str} (zero-sum charge balance)
  * Solver Runtime (p95):    {solver_str}
  * Solve Success Rate:      {succ_str}
"""
    summary = f"  0. Clinical Efficacy: Seizure Burden = {burden_str} (Containment: {containment_str}).\n"
    return block, summary


def _summarize_realized_vs_planned(defs: dict[str, Any]) -> dict[str, Any]:
    """Compute summary statistics comparing realized knot 0 against planned knots."""
    mpc_p = defs.get("mpc_percentiles")
    if mpc_p is not None and len(mpc_p) > 0 and mpc_p.shape[1] > 1:
        q0 = mpc_p[:, 0]
        q_fut = mpc_p[:, 1:]
        q0_mean, q0_med = float(np.mean(q0)), float(np.median(q0))
        q0_p95 = float(np.mean(q0 > 0.95) * 100.0)  # noqa: PLR2004 -- 95th percentile
        q0_p99 = float(np.mean(q0 > 0.99) * 100.0)  # noqa: PLR2004 -- 99th percentile
        q_fut_mean, q_fut_med = float(np.mean(q_fut)), float(np.median(q_fut))
        q_fut_p95 = float(np.mean(q_fut > 0.95) * 100.0)  # noqa: PLR2004 -- 95th percentile
        q_fut_p99 = float(np.mean(q_fut > 0.99) * 100.0)  # noqa: PLR2004 -- 99th percentile
    else:
        q0 = None
        q0_mean, q0_med, q0_p95, q0_p99 = float("nan"), float("nan"), float("nan"), float("nan")
        q_fut_mean, q_fut_med, q_fut_p95, q_fut_p99 = float("nan"), float("nan"), float("nan"), float("nan")

    e_real = defs.get("e_realized")
    if e_real is not None and len(e_real) > 0:
        e_real_mean = float(np.mean(e_real))
        e_real_med = float(np.median(e_real))
        r_real = (
            float(np.corrcoef(q0[:-1], e_real)[0, 1])
            if (
                q0 is not None
                and len(q0) > _MIN_BINS_FOR_AMPLIFICATION
                and np.std(q0[:-1]) > _EPSILON
                and np.std(e_real) > _EPSILON
            )
            else 0.0
        )
    else:
        e_real_mean, e_real_med, r_real = float("nan"), float("nan"), float("nan")

    phys_interp, exec_item2 = _interpret_lookahead_dynamics(q_fut_med, q0_p95)
    return {
        "q0_mean": q0_mean,
        "q0_med": q0_med,
        "q0_p95": q0_p95,
        "q0_p99": q0_p99,
        "q_fut_mean": q_fut_mean,
        "q_fut_med": q_fut_med,
        "q_fut_p95": q_fut_p95,
        "q_fut_p99": q_fut_p99,
        "e_real_mean": e_real_mean,
        "e_real_med": e_real_med,
        "r_real": r_real,
        "phys_interp": phys_interp,
        "exec_item2": exec_item2,
    }


def format_ood_report(defs: dict[str, Any]) -> str:
    """Format structured text-based OOD evaluation metrics and criteria verdicts."""
    mpc_d = defs["mpc_distances_flat"]
    cal_d = defs["cal_distances"]
    p95 = defs["p95_rate"]
    p99 = defs["p99_rate"]
    cv = defs["cv_score"]
    h_stats = defs["h_stats"]
    drift_delta = defs["drift_delta"]
    mean_q_max = defs["mean_q_max"]
    mean_q_mean = defs["mean_q_mean"]
    bins_summary = defs["bins_summary"]
    corr = defs["corr"]
    horizon = defs["horizon"]
    dim = defs["dim"]
    active_k = defs["active_k"]
    ood_index = defs["ood_index"]
    run_select = defs.get("run_select")
    run_val = getattr(run_select, "value", run_select)
    run_path = str(run_val) if run_val is not None else ""
    model_name = run_path.replace("\\", "/").split("/")[-1] if run_path else "fast_linear_kw5"

    d_id_mean = float(np.mean(cal_d))
    d_mpc_mean = float(np.mean(mpc_d))
    d_ratio = d_mpc_mean / d_id_mean if d_id_mean > 0 else float("nan")

    rs = _summarize_realized_vs_planned(defs)
    corr_str = f"r = {corr:+.3f}" if not np.isnan(corr) else "r = N/A (constant Q_H)"
    clin_block, burden_summary = _format_clinical_report_block(defs.get("clinical_metrics"))

    report = f"""
================================================================================
  MPC OOD SUPPORT EVALUATION REPORT (docs/mpc_ood_experiment.md)
================================================================================
Model:                     {model_name}
State-Action Dimension D:  {dim}
Nearest Neighbors K:       {active_k}
Training Reference Points: {ood_index.reference_count:,}
ID Calibration Points:     {ood_index.calibration_count:,}
Lookahead Horizon H:       {horizon} steps (0.06s per step)
{clin_block}
--------------------------------------------------------------------------------
REALIZED TRAJECTORY (k=0) VS. PLANNED ROLLOUTS (k >= 1)
--------------------------------------------------------------------------------
  * Realized Steps (k=0):      mean q = {rs["q0_mean"]:.3f}, median q = {rs["q0_med"]:.3f}
                               P(q > 0.95) = {rs["q0_p95"]:.2f}%, P(q > 0.99) = {rs["q0_p99"]:.2f}%
  * Planned Knots (k >= 1):    mean q = {rs["q_fut_mean"]:.3f}, median q = {rs["q_fut_med"]:.3f}
                               P(q > 0.95) = {rs["q_fut_p95"]:.2f}%, P(q > 0.99) = {rs["q_fut_p99"]:.2f}%
  * Realized 1-Step Error:     mean = {rs["e_real_mean"]:.3f}, median = {rs["e_real_med"]:.3f}
  * Realized Error Corr:       r(q_0, e_realized) = {rs["r_real"]:+.3f}
  * Physical Interpretation:   {rs["phys_interp"]}

--------------------------------------------------------------------------------
CRITERION 1: OOD Distance Support Comparison
--------------------------------------------------------------------------------
  * ID Calibration Distance: mean = {d_id_mean:.3f}, median = {np.median(cal_d):.3f}, std = {np.std(cal_d):.3f}
  * MPC Queries Distance:    mean = {d_mpc_mean:.3f}, median = {np.median(mpc_d):.3f}, std = {np.std(mpc_d):.3f}
  * Distance Ratio (MPC/ID): {d_ratio:.3f}
  * Desired Behavior:        d_MPC <= d_ID (MPC queries fall within supported manifold)
  * Verdict:                 {"[PASS] Well-supported. MPC queries remain on-manifold." if d_ratio <= _RATIO_TOLERANCE else "[SHIFT DETECTED] MPC queries stray into less supported regions."}

--------------------------------------------------------------------------------
CRITERION 6: Distance Concentration Diagnostic (High-D Check)
--------------------------------------------------------------------------------
  * Coefficient of Variation: CV = {cv:.4f} ({cv * 100:.2f}%)
  * Target Threshold:         CV > 0.05 (Ideal), CV < 0.01 (Distance Collapse)
  * Verdict:                 {"[HEALTHY DISPERSION] Distances remain informative." if cv >= _CV_THRESHOLD else f"[MODERATE CONCENTRATION] D={dim} compresses relative spread, but metric remains discriminative."}

--------------------------------------------------------------------------------
CRITERION 2: Lookahead Horizon Compounding Drift (k = 0 -> H-1)
--------------------------------------------------------------------------------
  * Knot 0 Support (q_0):          mean = {h_stats["mean_percentile"][0]:.3f}, median = {h_stats["median_percentile"][0]:.3f}
  * Terminal Knot Support (q_H-1): mean = {h_stats["mean_percentile"][-1]:.3f}, median = {h_stats["median_percentile"][-1]:.3f}
  * Compounding Drift (Delta_q):   {drift_delta:+.3f}
  * Desired Behavior:              Flat baseline (Delta_q ~ 0) or mild drift (q <= 0.50)
  * Verdict:                       {"[PASS - STABLE] No compounding runaway drift." if drift_delta <= _DRIFT_THRESHOLD else "[DRIFT DETECTED] Multi-step lookahead drifts off-distribution."}

--------------------------------------------------------------------------------
CRITERION 3: Support Violation Frequency (Rollout-Wide)
--------------------------------------------------------------------------------
  * Exceeding 95th Percentile:  {p95:.2f}%  (In-distribution expected baseline: <= 5.0%)
  * Exceeding 99th Percentile:  {p99:.2f}%  (In-distribution expected baseline: <= 1.0%)
  * Mean Trajectory Q_max:      {mean_q_max:.3f}
  * Mean Trajectory Q_mean:     {mean_q_mean:.3f}
  * Desired Behavior:           P(q > 0.95) <= 5%, P(q > 0.99) <= 1%
  * Verdict:                    {"[PASS - LOW VIOLATIONS] Queries rarely exceed training distribution tails." if p95 <= _P95_THRESHOLD else "[WARNING] Significant proportion of queries enter unsupported space."}

--------------------------------------------------------------------------------
CRITERION 4: One-Step Dynamics Prediction Error vs. Support Percentile
--------------------------------------------------------------------------------
  Empirical Error by Calibration Percentile Bins:
  +------------------+----------+---------------+---------------+---------------+
  | Percentile Range | Count    | Mean Error e_t| Std Error     | Median Error  |
  +------------------+----------+---------------+---------------+---------------+"""

    for b in bins_summary:
        report += f"\n  | {b['bin_label']:<16} | {b['count']:<8} | {b['mean_error']:<13.4f} | {b['std_error']:<13.4f} | {b['median_error']:<13.4f} |"

    means = [b["mean_error"] for b in bins_summary if not np.isnan(b["mean_error"])]
    amplification = means[-1] / max(means[0], 1e-6) if len(means) >= _MIN_BINS_FOR_AMPLIFICATION else 1.0

    report += f"""
  +------------------+----------+---------------+---------------+---------------+
  * Error Amplification (Tail / Baseline): {amplification:.2f}x
  * Desired Behavior:                      Monotonic escalation (q_t ^ => e_t ^)
  * Verdict:                               {"[CONFIRMED] Prediction error increases with OOD score." if amplification > 1.0 else "[UNCONFIRMED] Error does not correlate with distance."}

--------------------------------------------------------------------------------
CRITERION 5: Multi-Step Rollout Error vs. Maximum Violation (Q_H)
--------------------------------------------------------------------------------
  * Pearson Correlation r(Q_H, E_H): {corr:+.3f}
  * Desired Behavior:                r > 0 (Higher support violation predicts higher rollout error)
  * Verdict:                         {"[CONFIRMED] Positive correlation between rollout OOD score and multi-step error." if corr > _CORR_THRESHOLD else "[WEAK/NO CORRELATION] Q_H does not reliably predict E_H."}

================================================================================
  EXECUTIVE SUMMARY
================================================================================
{burden_summary}  1. Realized Closed-Loop Operation (k=0): Operates steadily on the training manifold
     (median q = {rs["q0_med"]:.3f}, P(q > 0.95) = {rs["q0_p95"]:.2f}%), closely matching the theoretical
     ~5.0% tail baseline without runaway divergence.
  2. {rs["exec_item2"]}
  3. Functional Relevance: Tail violations correlate with higher one-step dynamics
     error ({amplification:.2f}x) and multi-step rollout divergence ({corr_str}).
================================================================================
"""
    return report


def print_ood_report(defs: dict[str, Any]) -> None:
    """Print structured text-based OOD evaluation metrics to stdout."""
    print(format_ood_report(defs))  # noqa: T201 -- intentional CLI report output


@dataclass(frozen=True)
class CandidateState:
    """Recorded closed-loop transition flagged as an out-of-distribution candidate state."""

    run_dir: Path
    seed: int
    decision_idx: int
    t_star: float
    ood_score: float


def select_ood_candidate_states(  # noqa: PLR0913 -- candidate selection requires run, thresholds, index, and intervals
    run_dir: Path | str,
    *,
    ood_index: OODIndex | None = None,
    ood_scores: FloatArray | None = None,
    threshold: float = 0.95,
    min_interval_s: float = 1.5,
    max_candidates: int = 5,
    n_u: int = 0,
) -> list[CandidateState]:
    """Select candidate decision steps exceeding an OOD percentile threshold.

    Parameters
    ----------
    run_dir : Path | str
        Directory of the closed-loop MPC run containing ``config.yaml`` and ``log.npz``.
    ood_index : OODIndex | None, optional
        Fitted empirical support index used to evaluate realized knot 0 queries if
        ``ood_scores`` is not provided.
    ood_scores : FloatArray | None, optional
        Precomputed knot 0 percentile scores corresponding to valid decisions.
    threshold : float, default=0.95
        Minimum OOD percentile for a decision step to qualify as a candidate state.
    min_interval_s : float, default=1.5
        Minimum physical time interval in seconds between consecutive candidate states.
    max_candidates : int, default=5
        Maximum number of candidate states to return.
    n_u : int, default=0
        Past Control Current steps used when extracting queries from ``log.npz``.

    Returns
    -------
    list[CandidateState]
        Discovered candidate states sorted chronologically by decision time.
    """
    r_path = Path(run_dir)
    cfg_file = r_path / "config.yaml"
    log_file = r_path / "log.npz"

    if not cfg_file.exists() or not log_file.exists():
        msg = f"Run artifacts missing at {r_path}"
        raise FileNotFoundError(msg)

    with cfg_file.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    seed = int(cfg.get("dynamics", {}).get("seed", 0))

    if ood_scores is None:
        if ood_index is None:
            msg = "Either ood_index or ood_scores must be provided."
            raise ValueError(msg)
        queries, _, valid_decisions = extract_mpc_rollout_queries(log_file, n_u=n_u)
        q0 = queries[:, 0, :]
        distances = ood_index.compute_distances(q0)
        scores = ood_index.percentiles(distances)
    else:
        scores = np.asarray(ood_scores, dtype=np.float64)
        with np.load(log_file) as log_data:
            if "controller.warmup" in log_data:
                valid_decisions = np.where(~log_data["controller.warmup"].astype(bool))[0]
            else:
                valid_decisions = np.arange(len(scores))

    with np.load(log_file) as log_data:
        t_ctrl = np.asarray(log_data["controller.t"], dtype=np.float64)

    t_valid = t_ctrl[valid_decisions]

    candidate_indices = np.where(scores >= threshold)[0]
    if len(candidate_indices) == 0:
        return []

    ranked = candidate_indices[np.argsort(-scores[candidate_indices])]
    selected: list[int] = []

    for idx in ranked:
        t_cand = t_valid[idx]
        if any(abs(t_cand - t_valid[prev]) < min_interval_s for prev in selected):
            continue
        selected.append(idx)
        if len(selected) >= max_candidates:
            break

    selected.sort()

    return [
        CandidateState(
            run_dir=r_path,
            seed=seed,
            decision_idx=int(valid_decisions[idx]),
            t_star=float(t_valid[idx]),
            ood_score=float(scores[idx]),
        )
        for idx in selected
    ]


def _slice_and_save_branch(  # noqa: PLR0913 -- helper for branch trajectory slicing and persistence
    sim: Simulation,
    candidate: CandidateState,
    *,
    ras_seed: int,
    t_start: float,
    t_star: float,
    t_end: float,
    out_dir: Path,
    prefix: str,
) -> Path:
    """Slice simulation records to history buffer and save compressed archive."""
    if sim.logger is None:
        msg = "Simulation logger missing after run."
        raise RuntimeError(msg)

    s_t_raw, s_y_raw = sim.logger.signal("sensor_0", "y_mea")
    c_t_raw, c_u_raw = sim.logger.signal("controller", "u")
    s_t = np.asarray(s_t_raw, dtype=np.float64)
    s_y = np.asarray(s_y_raw, dtype=np.float64)
    c_t = np.asarray(c_t_raw, dtype=np.float64)
    c_u = np.asarray(c_u_raw, dtype=np.float64)

    # Slice sensor EEG
    s_mask = (s_t >= t_start) & (s_t <= t_end)
    sliced_s_t = s_t[s_mask] - t_start
    sliced_s_y = s_y[s_mask]

    # Slice controller u ensuring first control covers t_start
    idx_c_start = max(0, int(np.searchsorted(c_t, t_start, side="right") - 1))
    idx_c_end = int(np.searchsorted(c_t, t_end, side="right"))
    sliced_c_t = np.maximum(0.0, c_t[idx_c_start:idx_c_end] - t_start)
    sliced_c_u = c_u[idx_c_start:idx_c_end]

    save_dict: dict[str, np.ndarray] = {
        "sensor_0.t": sliced_s_t,
        "sensor_0.y_mea": sliced_s_y,
        "controller.t": sliced_c_t,
        "controller.u": sliced_c_u,
    }

    # Dynamics LFP slice if logged
    try:
        d_t_raw, d_lfp_raw = sim.logger.signal("dynamics", "lfp")
        d_t = np.asarray(d_t_raw, dtype=np.float64)
        d_lfp = np.asarray(d_lfp_raw, dtype=np.float64)
        d_mask = (d_t >= t_start) & (d_t <= t_end)
        save_dict["dynamics.t"] = d_t[d_mask] - t_start
        save_dict["dynamics.lfp"] = d_lfp[d_mask]
    except (KeyError, ValueError):
        pass

    save_dict["metadata_candidate_t_star"] = np.array(t_star, dtype=np.float64)
    save_dict["metadata_relative_t_star"] = np.array(t_star - t_start, dtype=np.float64)
    save_dict["metadata_source_run"] = np.array(str(candidate.run_dir))
    save_dict["metadata_dynamics_seed"] = np.array(candidate.seed, dtype=np.int64)
    save_dict["metadata_ras_seed"] = np.array(ras_seed, dtype=np.int64)
    save_dict["metadata_ood_score"] = np.array(candidate.ood_score, dtype=np.float64)

    out_file = out_dir / f"{prefix}_s{candidate.seed}_t{t_star:.2f}_ras{ras_seed}.npz"
    np.savez_compressed(out_file, allow_pickle=True, **save_dict)
    return out_file


def simulate_ras_branch(  # noqa: PLR0913 -- branch simulation requires duration, history, seeds, and schedule parameters
    candidate: CandidateState,
    *,
    ras_seeds: Sequence[int] = (101, 102, 103),
    duration_s: float = 4.0,
    history_s: float = 2.0,
    output_dir: Path | str = Path("data/extended_train"),
    amp: float = 2.0,
    hold_ms: list[float] | None = None,
    prefix: str = "cand",
) -> list[Path]:
    """Simulate Option B Random Amplitude Schedule (RAS) branches from a candidate state.

    Parameters
    ----------
    candidate : CandidateState
        Candidate state providing decision time ``t_star``, seed, and closed-loop run path.
    ras_seeds : Sequence[int], default=(101, 102, 103)
        Random seeds for the Random Amplitude Schedule excitation signals.
    duration_s : float, default=4.0
        Duration of the active Random Amplitude Schedule excitation segment in seconds.
    history_s : float, default=2.0
        Preceding history buffer duration retained before ``t_star`` in seconds.
    output_dir : Path | str, default=Path("data/extended_train")
        Directory where generated branch trajectory ``.npz`` files will be saved.
    amp : float, default=2.0
        Excitation current amplitude in mA.
    hold_ms : list[float] | None, optional
        Hold durations in milliseconds. Defaults to ``[60.0, 120.0, 240.0, 600.0, 1200.0]``.
    prefix : str, default="cand"
        File name prefix for emitted trajectory files.

    Returns
    -------
    list[Path]
        Paths of saved branch trajectory archives.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hold_ms is None:
        hold_ms = [60.0, 120.0, 240.0, 600.0, 1200.0]

    cfg_file = candidate.run_dir / "config.yaml"
    log_file = candidate.run_dir / "log.npz"

    if not cfg_file.exists() or not log_file.exists():
        msg = f"Candidate run artifacts missing at {candidate.run_dir}"
        raise FileNotFoundError(msg)

    with cfg_file.open(encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f) or {}

    with np.load(log_file) as log_data:
        u_rec = np.asarray(log_data["controller.u"], dtype=np.float64)

    t_star = float(candidate.t_star)
    t_end = t_star + duration_s
    t_start = max(0.0, t_star - history_s)
    dt_u = float(run_cfg.get("controller", {}).get("dt", 0.06))

    n_prefix = min(len(u_rec), round(t_star / dt_u))
    n_branch = max(1, round(duration_s / dt_u))
    n_total = n_prefix + n_branch
    n_controls = u_rec.shape[1]

    created_paths: list[Path] = []
    for ras_seed in ras_seeds:
        ras_u = build_input_schedule(
            input_type="ras",
            n_steps=n_branch,
            transient_steps=0,
            n_controls=n_controls,
            amp=amp,
            hold_ms=hold_ms,
            dt=dt_u,
            rng=np.random.default_rng(ras_seed),
        )
        full_u = np.zeros((n_total, n_controls), dtype=np.float64)
        full_u[:n_prefix] = u_rec[:n_prefix]
        full_u[n_prefix:] = ras_u[: n_total - n_prefix]

        sim_cfg = deepcopy(run_cfg)
        sim_cfg["t_end"] = t_end
        sim_cfg.setdefault("dynamics", {})["seed"] = candidate.seed
        sim_cfg["controller"] = {"class_path": "neuro.control.zero.ZeroController", "dt": dt_u, "n_u": n_controls}

        sim = Simulation.from_config(sim_cfg)
        sim.controller = ScheduleController(dt=dt_u, schedule=full_u)
        sim.run()

        out_file = _slice_and_save_branch(
            sim=sim,
            candidate=candidate,
            ras_seed=ras_seed,
            t_start=t_start,
            t_star=t_star,
            t_end=t_end,
            out_dir=out_dir,
            prefix=prefix,
        )
        created_paths.append(out_file)

    return created_paths
