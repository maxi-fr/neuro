# Diagnostic probe measuring empirical signal properties and predictive information of the 0 Hz STFT bin.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal.windows import hann

if TYPE_CHECKING:
    from neuro.types import FloatArray


def fit_ridge_predict(x_train: FloatArray, y_train: FloatArray, x_test: FloatArray, alpha: float = 10.0) -> FloatArray:
    """Fit ridge regression with an intercept and return predictions on test set."""
    # Add column of ones for intercept
    x_tr_aug = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_te_aug = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    dim = x_tr_aug.shape[1]
    reg = alpha * np.eye(dim)
    reg[0, 0] = 0.0  # Do not regularize intercept
    beta = np.linalg.solve(x_tr_aug.T @ x_tr_aug + reg, x_tr_aug.T @ y_train)
    return np.asarray(x_te_aug @ beta, dtype=np.float64)


def calc_r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Calculate coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_frames(
    y: FloatArray,
    *,
    n_segment: int = 50,
    n_hop: int = 5,
    detrend: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Compute raw power and log-power STFT frames with Hann window."""
    n_samples, _n_channels = y.shape
    n_frames = (n_samples - n_segment) // n_hop + 1
    w_hann = hann(n_segment, sym=False)
    segments = np.stack([y[m * n_hop : m * n_hop + n_segment] for m in range(n_frames)], axis=0)
    if detrend:
        segments = segments - segments.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(segments * w_hann[None, :, None], axis=1)
    power = np.abs(spectrum) ** 2  # (n_frames, n_bins, n_channels)
    power = np.moveaxis(power, 2, 1)  # (n_frames, n_channels, n_bins)
    log_power = np.log(power + 1e-8)
    return power, log_power


def run_probe(  # noqa: PLR0915 -- diagnostic probe performs sequential dataset analysis
    data_dir: Path,
    n_files: int = 50,
    n_segment: int = 50,
    n_hop: int = 5,
    fs: float = 50.0,
) -> dict[str, float]:
    """Run empirical diagnostic probe on dataset trajectories."""
    files = sorted(data_dir.glob("sim_*.npz"))[:n_files]
    if not files:
        msg = f"No simulation files found in {data_dir}"
        raise FileNotFoundError(msg)

    # 1. Compare detrended vs non-detrended DC power on first file
    sample = np.load(files[0])
    y_sample = sample["sensor_0.y_mea"][::200]
    p_nodetrend, _ = compute_frames(y_sample, n_segment=n_segment, n_hop=n_hop, detrend=False)
    p_detrend, _ = compute_frames(y_sample, n_segment=n_segment, n_hop=n_hop, detrend=True)

    dc_p_nodetrend = float(np.median(p_nodetrend[:, :, 0]))
    dc_p_detrend = float(np.median(p_detrend[:, :, 0]))
    ac_p_median = float(np.median(p_nodetrend[:, :, 1:]))

    # 2. Extract multi-file dataset for predictive regressions
    all_p0: list[FloatArray] = []
    all_p_ac: list[FloatArray] = []
    all_seizure: list[FloatArray] = []
    all_u: list[FloatArray] = []

    # Seizure band: 3 to 12 Hz inclusive -> bin indices 3 to 12 (at n_segment=50, fs=50, 1 Hz/bin)
    seiz_lo, seiz_hi = 3, 13

    for path in files:
        data = np.load(path)
        y = data["sensor_0.y_mea"][::200]  # (601, 62)
        u = data["controller.u"][::200]  # (601, 3)
        _, log_p = compute_frames(y, n_segment=n_segment, n_hop=n_hop, detrend=False)
        n_frames = log_p.shape[0]

        p0 = log_p[:, :, 0]  # (n_frames, 62)
        p_ac = log_p[:, :, 1:].reshape(n_frames, -1)  # (n_frames, 62 * 25)
        p_seiz = log_p[:, :, seiz_lo:seiz_hi].mean(axis=(1, 2))  # (n_frames,)
        u_aligned = u[n_segment - 1 :: n_hop][:n_frames]  # (n_frames, 3)

        all_p0.append(p0)
        all_p_ac.append(p_ac)
        all_seizure.append(p_seiz)
        all_u.append(u_aligned)

    n_train = int(len(files) * 0.8)

    # Correlation with stimulation
    p0_all_concat = np.concatenate(all_p0)
    u_all_concat = np.concatenate(all_u)
    corr_u_p0 = [float(np.corrcoef(u_all_concat[:, j], p0_all_concat.mean(axis=1))[0, 1]) for j in range(3)]

    # 3. Test predictive incremental information across lookahead horizons
    horizons = [1, 5, 10, 15]  # 0.1s, 0.5s, 1.0s, 1.5s
    results: dict[str, float] = {
        "dc_p_nodetrend": dc_p_nodetrend,
        "dc_p_detrend": dc_p_detrend,
        "ac_p_median": ac_p_median,
        "corr_u0_p0": corr_u_p0[0],
        "corr_u1_p0": corr_u_p0[1],
        "corr_u2_p0": corr_u_p0[2],
    }

    for h in horizons:
        # Build train and test sets
        x_base_tr, x_aug_tr, y_fut_tr = [], [], []
        x_base_te, x_aug_te, y_fut_te = [], [], []

        for i, (p0_arr, ac_arr, seiz_arr, u_arr) in enumerate(zip(all_p0, all_p_ac, all_seizure, all_u, strict=True)):
            if len(seiz_arr) <= h:
                continue
            x_base = np.hstack([ac_arr[:-h], u_arr[:-h]])
            x_aug = np.hstack([p0_arr[:-h], ac_arr[:-h], u_arr[:-h]])
            y_target = seiz_arr[h:]

            if i < n_train:
                x_base_tr.append(x_base)
                x_aug_tr.append(x_aug)
                y_fut_tr.append(y_target)
            else:
                x_base_te.append(x_base)
                x_aug_te.append(x_aug)
                y_fut_te.append(y_target)

        xb_tr = np.vstack(x_base_tr)
        xa_tr = np.vstack(x_aug_tr)
        yf_tr = np.concatenate(y_fut_tr)

        xb_te = np.vstack(x_base_te)
        xa_te = np.vstack(x_aug_te)
        yf_te = np.concatenate(y_fut_te)

        # Fit Ridge regression
        pred_base = fit_ridge_predict(xb_tr, yf_tr, xb_te, alpha=10.0)
        pred_aug = fit_ridge_predict(xa_tr, yf_tr, xa_te, alpha=10.0)

        r2_base = calc_r2(yf_te, pred_base)
        r2_aug = calc_r2(yf_te, pred_aug)
        mse_base = float(np.mean((yf_te - pred_base) ** 2))
        mse_aug = float(np.mean((yf_te - pred_aug) ** 2))

        # Predictability of 0 Hz itself
        p0_fut_tr = np.concatenate([arr[h:] for i, arr in enumerate(all_p0) if i < n_train])
        p0_fut_te = np.concatenate([arr[h:] for i, arr in enumerate(all_p0) if i >= n_train])
        p0_pred_te = fit_ridge_predict(xa_tr, p0_fut_tr, xa_te, alpha=10.0)
        r2_p0 = calc_r2(p0_fut_te, p0_pred_te)

        lookahead_s = h * (n_hop / fs)
        results[f"r2_base_{lookahead_s:0.1f}s"] = r2_base
        results[f"r2_aug_{lookahead_s:0.1f}s"] = r2_aug
        results[f"mse_base_{lookahead_s:0.1f}s"] = mse_base
        results[f"mse_aug_{lookahead_s:0.1f}s"] = mse_aug
        results[f"r2_p0_fut_{lookahead_s:0.1f}s"] = r2_p0

    return results


if __name__ == "__main__":
    data_directory = Path("data/experiment_excited_long/train")
    metrics = run_probe(data_directory, n_files=50)

    print("=== Empirical DC (0 Hz) Signal & Detrending Properties ===")
    print(f"Median power at 0 Hz (detrend=False): {metrics['dc_p_nodetrend']:.2e} mV^2")
    print(f"Median power at 0 Hz (detrend=True):  {metrics['dc_p_detrend']:.2e} mV^2")
    print(f"Median power across AC bins (1-25 Hz): {metrics['ac_p_median']:.2e} mV^2")
    print(f"Ratio 0 Hz / AC power: {metrics['dc_p_nodetrend'] / metrics['ac_p_median']:.1f}x")

    print("\n=== Cross-Correlation with Stimulation Current u ===")
    print(f"Corr(u[0], mean(0 Hz)): {metrics['corr_u0_p0']:+.4f}")
    print(f"Corr(u[1], mean(0 Hz)): {metrics['corr_u1_p0']:+.4f}")
    print(f"Corr(u[2], mean(0 Hz)): {metrics['corr_u2_p0']:+.4f}")

    print("\n=== Out-of-Sample Predictive Value for Seizure Band (3-12 Hz) Power ===")
    print(f"{'Horizon':<10} | {'Base R^2 (1-25Hz)':<18} | {'Aug R^2 (+0Hz)':<16} | {'Base MSE':<10} | {'Aug MSE':<10} | {'0Hz Self-R^2':<12}")
    print("-" * 88)
    for h_s in [0.1, 0.5, 1.0, 1.5]:
        r2_b = metrics[f"r2_base_{h_s:0.1f}s"]
        r2_a = metrics[f"r2_aug_{h_s:0.1f}s"]
        mse_b = metrics[f"mse_base_{h_s:0.1f}s"]
        mse_a = metrics[f"mse_aug_{h_s:0.1f}s"]
        r2_p = metrics[f"r2_p0_fut_{h_s:0.1f}s"]
        print(f"{h_s:0.1f}s{'':<6} | {r2_b:<18.4f} | {r2_a:<16.4f} | {mse_b:<10.4f} | {mse_a:<10.4f} | {r2_p:<12.4f}")
