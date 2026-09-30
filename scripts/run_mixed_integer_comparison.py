from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from simulate.config import load_config

from neuro.mixed_integer_comparison import SEEDS, controller_arms, score_seed, target_summary
from neuro.validation import validate_simulation_config


def main() -> None:
    """Run every arm on each specified seed and persist each completed cell."""
    parser = argparse.ArgumentParser(description="Run the paired CasADi MPC Plant comparison.")
    parser.add_argument("base", type=Path, help="Resolved observable predictor simulation YAML.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True, help="Maximum enabled steps per Control Horizon.")
    parser.add_argument("--weight", type=float, required=True, help="Enabled-step Cost weight.")
    parser.add_argument("--arms", nargs="*", default=None, help="Only these arms; useful for a timed first seed.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Only these paired Plant seeds.")
    args = parser.parse_args()
    if args.cap < 0 or args.weight <= 0:
        msg = "--cap must be nonnegative and --weight must be positive."
        raise ValueError(msg)
    arms = controller_arms(load_config(args.base), cap=args.cap, weight=args.weight)
    if args.arms is not None:
        arms = {name: arms[name] for name in args.arms}
    seeds = SEEDS if args.seeds is None else tuple(args.seeds)
    if not set(seeds) <= set(SEEDS):
        msg = f"Seeds must be selected from {SEEDS}."
        raise ValueError(msg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for arm, template in arms.items():
            config = template | {"dynamics": template["dynamics"] | {"seed": seed}}
            validate_simulation_config(config)
            cell_dir = args.output_dir / f"{arm}_s{seed}"
            cell_dir.mkdir(exist_ok=True)
            config_path = cell_dir / "config.yaml"
            if config_path.exists() and yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
                msg = f"Cannot resume {cell_dir}: resolved config changed."
                raise ValueError(msg)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result_path = cell_dir / "result.json"
            if result_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                row = {"arm": arm, **score_seed(config)}
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(
                f"{arm} seed {seed}: {row['n_seizing_final']} final seizing regions; status {row['solver_failure_status']}",
                flush=True,
            )
            _write_report(args.output_dir, rows)


def _write_report(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Flush completed cells and target status after every run."""
    report = {
        "current_zero_threshold_ma": 1e-6,
        "charge_units": "mA s = mC, summed over electrodes",
        "energy_proxy_units": "mA^2 s summed over electrodes; physical energy needs resistance",
        "activation_scope": "horizon-local; no run-wide duty-cycle cap",
        "optimality_claim": "solver-successful feasible result; no global optimality claim",
        "realtime_claim": "none",
        "rows": rows,
        "target": target_summary(rows),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
