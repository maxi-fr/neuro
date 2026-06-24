"""Script to run system identification on simulated Jansen-Rit data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from neuro.sysid import SysIDSolver

MAX_PRINT_ELEMENTS = 4
NUM_CHANNELS_PLOT = 4


def _plot_results(y_data: np.ndarray, y_hat: np.ndarray, n_steps: int, dt: float) -> None:
    _fig, axes = plt.subplots(NUM_CHANNELS_PLOT, 1, figsize=(10, 8), sharex=True)
    t_axis = np.arange(n_steps) * dt
    for i in range(NUM_CHANNELS_PLOT):
        axes[i].plot(t_axis, y_data[:, i], label="True Data", color="black", linewidth=1.5)
        axes[i].plot(t_axis, y_hat[:, i], label="Identified", color="red", linestyle="--", linewidth=1.5)
        axes[i].set_ylabel(f"Ch {i}")
        if i == 0:
            axes[i].legend()
            axes[i].set_title(f"EEG Output Comparison (First {NUM_CHANNELS_PLOT} Channels)")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("identified_eeg_comparison.png", dpi=300)
    print("   Plot saved to identified_eeg_comparison.png")


def _load_data(data_file: str, n_steps: int, downsample: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n1. Loading data from {data_file}...")
    with np.load(data_file) as data:
        max_idx = n_steps * downsample
        y_data = data["universal_y_mea"][:max_idx:downsample]
        u_data = data["universal_u"][:max_idx:downsample]

        print(f"   Loaded y_data shape: {y_data.shape}")
        print(f"   Loaded u_data shape: {u_data.shape}")
    return u_data, y_data


def main() -> None:
    """Run SysID on simulated data based on a YAML configuration file."""
    parser = argparse.ArgumentParser(description="Run SysID on simulated data.")
    parser.add_argument("--config", type=str, default="configs/sysid/sysid_config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with config_path.open() as f:
        config = yaml.safe_load(f)

    dt = config.get("dt", 1e-3)
    downsample = config.get("downsample", 1)
    n_steps = config.get("n_steps", 200)
    free_params = config.get("free_params", ["A"])
    data_file = config.get("data_file")

    if not data_file or not Path(data_file).exists():
        print(f"Data file not found or not specified in config: {data_file}")
        return

    u_data, y_data = _load_data(data_file, n_steps, downsample)

    print("\n2. Configuring SysIDSolver...")
    print(f"   Free parameters: {free_params}")
    solver = SysIDSolver.from_config(config)

    print("\n3. Solving...")
    res = solver.solve(u_data, y_data)

    print("\n===============================")
    print(f"Optimization finished. Final Cost: {res.cost:.4f}")
    print("===============================\n")

    out_params = {}
    for p in free_params:
        print(f"--- Parameter '{p}' ---")
        rec_val = np.asarray(res.free_params[p])
        out_params[p] = rec_val.tolist()

        if rec_val.size <= MAX_PRINT_ELEMENTS:
            print("Recovered:")
            print(np.round(rec_val, 4))
        else:
            print(f"Recovered norm: {np.linalg.norm(rec_val):.4f}")
        print()

    print("\n4. Saving and Plotting...")

    # Save parameters
    with Path("identified_params.yaml").open("w") as f:
        yaml.dump(out_params, f)
    print("   Identified parameters saved to identified_params.yaml")

    # Plotting
    y_hat = res.trajectory  # (n_steps, n_channels)
    _plot_results(y_data, y_hat, n_steps, dt)


if __name__ == "__main__":
    main()
