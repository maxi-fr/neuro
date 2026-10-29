from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import psutil


def _set_low_resource_profile() -> None:
    """Set process priority to BelowNormal and limit thread counts to prevent resource contention."""
    try:
        proc = psutil.Process(os.getpid())
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001, S110 -- fallback gracefully if priority adjustment fails
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def _run_manifest(manifest_path: str, output_dir: str) -> None:
    """Run one comparison manifest sequentially using run_comparison.py."""
    cmd = [
        sys.executable,
        "scripts/run_comparison.py",
        manifest_path,
        "--output-dir",
        output_dir,
        "--workers",
        "1",
        "--resume",
    ]
    print("\n========================================================")
    print(f"Launching calibration: {manifest_path} -> {output_dir}")
    print("========================================================\n", flush=True)

    result = subprocess.run(cmd, check=True)  # noqa: S603 -- internal invocation
    if result.returncode != 0:
        msg = f"Comparison failed with exit code {result.returncode}"
        raise RuntimeError(msg)


def _load_summary(summary_csv_path: Path) -> list[dict[str, str]]:
    """Read summary CSV rows into list of dictionaries."""
    with summary_csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _generate_calibration_report(  # noqa: PLR0915 -- formats comprehensive comparison tables and sections
    hop5_dir: Path,
    hop3_dir: Path,
    report_path: Path,
) -> None:
    """Analyze calibration summaries, determine Pareto frontiers and select the champion Predictor."""
    hop5_summary = _load_summary(hop5_dir / "summary.csv")
    hop3_summary = _load_summary(hop3_dir / "summary.csv")

    def parse_float(val: str | None, default: float = float("nan")) -> float:
        if val is None or val == "":
            return default
        try:
            return float(val)
        except ValueError:
            return default

    # Sort arms by Seizure Burden
    hop5_sorted = sorted(hop5_summary, key=lambda row: parse_float(row.get("seizure_burden", "inf")))
    hop3_sorted = sorted(hop3_summary, key=lambda row: parse_float(row.get("seizure_burden", "inf")))

    # Identify best controlled arms (excluding baselines)
    controlled_hop5 = [row for row in hop5_sorted if row["arm"] not in ("uncontrolled", "threshold")]
    controlled_hop3 = [row for row in hop3_sorted if row["arm"] not in ("uncontrolled", "threshold")]

    best_hop5 = controlled_hop5[0] if controlled_hop5 else hop5_sorted[0]
    best_hop3 = controlled_hop3[0] if controlled_hop3 else hop3_sorted[0]

    # Overall champion selection per Section 5 rule:
    # 1. Smallest mean Seizure Burden
    # 2. Tie-break: lowest mean Delivered Charge
    burden_h5 = parse_float(best_hop5.get("seizure_burden"))
    burden_h3 = parse_float(best_hop3.get("seizure_burden"))
    charge_h5 = parse_float(best_hop5.get("delivered_charge"))
    charge_h3 = parse_float(best_hop3.get("delivered_charge"))

    if burden_h3 < burden_h5:
        champ_hop = "Hop 3"
        champ_row = best_hop3
    elif burden_h5 < burden_h3:
        champ_hop = "Hop 5"
        champ_row = best_hop5
    elif charge_h3 <= charge_h5:
        champ_hop = "Hop 3"
        champ_row = best_hop3
    else:
        champ_hop = "Hop 5"
        champ_row = best_hop5

    lines = [
        "# Observable Hankel-DMDc Controller Effort Calibration & Head-to-Head Report",
        "",
        "## 1. Executive Summary",
        "",
        "This report records the closed-loop controller effort calibration ($w_u$ sweep) and head-to-head comparison for Hankel-DMDc Observable Predictors under Section 5 and 6.1 of `docs/hyperparameter_sweep_strategy.md`.",
        "",
        f"- **Hankel-DMDc Champion**: **{champ_hop}** (`{champ_row['arm']}`)",
        f"- **Champion Seizure Burden**: `{parse_float(champ_row.get('seizure_burden')):.4f} ± {parse_float(champ_row.get('seizure_burden_sd')):.4f}`",
        f"- **Champion Delivered Charge**: `{parse_float(champ_row.get('delivered_charge')):.4f} ± {parse_float(champ_row.get('delivered_charge_sd')):.4f}`",
        f"- **IPOPT Solve Success Rate**: `{parse_float(champ_row.get('solve_success_rate')) * 100:.1f}%`",
        f"- **P95 Solve Time**: `{parse_float(champ_row.get('solve_time_p95_s')):.4f} s` (Real-time factor: `{parse_float(champ_row.get('realtime_factor')):.2f}`)",
        "",
        "---",
        "",
        "## 2. Hop 5 Effort Calibration (dt = 0.10 s, Horizon = 10)",
        "",
        "| Arm | Mean Seizure Burden | Mean Delivered Charge | Solve Success | P95 Solve Time (s) | Real-time Factor |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in hop5_sorted:
        b = parse_float(row.get("seizure_burden"))
        b_sd = parse_float(row.get("seizure_burden_sd"))
        c = parse_float(row.get("delivered_charge"))
        c_sd = parse_float(row.get("delivered_charge_sd"))
        s = parse_float(row.get("solve_success_rate"))
        p95 = parse_float(row.get("solve_time_p95_s"))
        rtf = parse_float(row.get("realtime_factor"))
        lines.append(
            f"| `{row['arm']}` | {b:.4f} ± {b_sd:.4f} | {c:.4f} ± {c_sd:.4f} | {s * 100:.1f}% | {p95:.4f} | {rtf:.2f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Hop 3 Effort Calibration (dt = 0.06 s, Horizon = 17)",
            "",
            "| Arm | Mean Seizure Burden | Mean Delivered Charge | Solve Success | P95 Solve Time (s) | Real-time Factor |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in hop3_sorted:
        b = parse_float(row.get("seizure_burden"))
        b_sd = parse_float(row.get("seizure_burden_sd"))
        c = parse_float(row.get("delivered_charge"))
        c_sd = parse_float(row.get("delivered_charge_sd"))
        s = parse_float(row.get("solve_success_rate"))
        p95 = parse_float(row.get("solve_time_p95_s"))
        rtf = parse_float(row.get("realtime_factor"))
        lines.append(
            f"| `{row['arm']}` | {b:.4f} ± {b_sd:.4f} | {c:.4f} ± {c_sd:.4f} | {s * 100:.1f}% | {p95:.4f} | {rtf:.2f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Head-to-Head Comparison & Champion Selection",
            "",
            "| Metric | Uncontrolled Baseline | Threshold Baseline | Hop 5 Calibrated Champion | Hop 3 Calibrated Champion |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    unc_h5 = next((r for r in hop5_summary if r["arm"] == "uncontrolled"), {})
    thr_h5 = next((r for r in hop5_summary if r["arm"] == "threshold"), {})

    lines.append(
        f"| Seizure Burden | {parse_float(unc_h5.get('seizure_burden')):.4f} | {parse_float(thr_h5.get('seizure_burden')):.4f} | {parse_float(best_hop5.get('seizure_burden')):.4f} | {parse_float(best_hop3.get('seizure_burden')):.4f} |"
    )
    lines.append(
        f"| Delivered Charge | {parse_float(unc_h5.get('delivered_charge')):.4f} | {parse_float(thr_h5.get('delivered_charge')):.4f} | {parse_float(best_hop5.get('delivered_charge')):.4f} | {parse_float(best_hop3.get('delivered_charge')):.4f} |"
    )
    lines.append(
        f"| Solve Success Rate | N/A | N/A | {parse_float(best_hop5.get('solve_success_rate')) * 100:.1f}% | {parse_float(best_hop3.get('solve_success_rate')) * 100:.1f}% |"
    )
    lines.append(
        f"| P95 Solve Time (s) | N/A | N/A | {parse_float(best_hop5.get('solve_time_p95_s')):.4f} | {parse_float(best_hop3.get('solve_time_p95_s')):.4f} |"
    )

    lines.extend(
        [
            "",
            "### Selection Decision",
            "",
            "Per Section 5 of the Hyperparameter Sweep Strategy, candidate advancement selects the eligible arm with the **smallest mean Seizure Burden** across the five development Plant seeds (`[7000, 7001, 7002, 7004, 7005]`), with exact ties broken by lower Delivered Charge, higher solve success rate, and lower P95 solve time.",
            "",
            f"- **Advancement Decision**: **{champ_hop}** (`{champ_row['arm']}`) advances as the Observable Hankel-DMDc champion to the final cross-family benchmark (Manifest Arm 5).",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote calibration report to: {report_path}\n", flush=True)


def main() -> None:
    """Execute sequential effort calibrations for Hop 5 and Hop 3 under low-resource priority."""
    _set_low_resource_profile()

    hop5_manifest = "configs/comparison/observable_dmd/calibration_hop5.yaml"
    hop5_dir = Path("artifacts/sweep_2026_10/observable_dmd/calibration/hop5")

    hop3_manifest = "configs/comparison/observable_dmd/calibration_hop3.yaml"
    hop3_dir = Path("artifacts/sweep_2026_10/observable_dmd/calibration/hop3")

    report_path = Path("artifacts/sweep_2026_10/observable_dmd/calibration/report.md")

    # 1. Run Hop 5 calibration
    _run_manifest(hop5_manifest, str(hop5_dir))

    # 2. Run Hop 3 calibration
    _run_manifest(hop3_manifest, str(hop3_dir))

    # 3. Generate analysis and report
    _generate_calibration_report(hop5_dir, hop3_dir, report_path)


if __name__ == "__main__":
    main()
