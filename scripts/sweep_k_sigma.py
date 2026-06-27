import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro.connectome import load_connectome
from neuro.jansen_rit import JansenRitParams, lfp, simulate_network


def main() -> None:
    """Run a parameter sweep over K and sigma to find the seizure threshold."""
    conn = load_connectome()
    ez = ("lHC", "lPHC", "lAMYG")
    pz = ("lTCI", "lTCV")

    a_gains = np.full(len(conn.region_labels), 3.25)
    for _nm in ez:
        a_gains[conn.region_index[_nm]] = 3.6
    for _nm in pz:
        a_gains[conn.region_index[_nm]] = 3.4

    # We sweep K_eff directly (where K_eff = 0.54 / divisor)
    k_vals = [0.59, 0.50, 0.51, 0.52, 0.53, 0.54]
    sigma_vals = [350, 400, 450, 500, 550]
    seeds = [42, 7, 13, 99, 123, 59, 60, 62, 62, 63]

    print(f"{'K':>6} | {'Sigma':>6} | {'N_seizing (min-median-max)':>26} | {'Std Dev':>7} | {'N_seeds':>7}")
    print("-" * 60)

    # To store median seizing nodes for the heatmap
    heatmap_data = np.zeros((len(k_vals), len(sigma_vals)))

    for i, K in enumerate(k_vals):
        for j, sigma in enumerate(sigma_vals):
            run_seeds = seeds if sigma > 0 else [42]

            seizing_counts = []

            for seed in run_seeds:
                hr_params = JansenRitParams.from_config(
                    {
                        "connectome": conn,
                        "dt": 1e-4,
                        "params": {
                            "A": a_gains,
                            "sigma": sigma,
                            "K": K,
                            "initial_bounds": [
                                [-1.0, 1.0],
                                [-500.0, 500.0],
                                [-50.0, 50.0],
                                [-6.0, 6.0],
                                [-20.0, 20.0],
                                [-500.0, 500.0],
                            ],
                        },
                    }
                )

                hr_t, hr_x = simulate_network(
                    params=hr_params,
                    duration=5.0,
                    dt=1e-4,
                    seed=seed,
                )

                hr_y = lfp(hr_x)

                thr = 5.0
                ptp = np.ptp(hr_y[:, hr_t >= 1.0], axis=1)
                n_seizing = int(np.sum(ptp > thr))
                seizing_counts.append(n_seizing)

            c_min = int(np.min(seizing_counts))
            c_med = int(np.median(seizing_counts))
            c_max = int(np.max(seizing_counts))
            c_std = float(np.std(seizing_counts))

            stat_str = f"{c_min:6d} - {c_med:5d} - {c_max:5d}"
            print(f"{K:6.3f} | {sigma:6.1f} | {stat_str:>26} | {c_std:7.2f} | {len(run_seeds):7d}", flush=True)
            heatmap_data[i, j] = c_med

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Median Seizing Nodes")

    plt.xticks(ticks=np.arange(len(sigma_vals)), labels=[f"{s:.0f}" for s in sigma_vals])
    plt.yticks(ticks=np.arange(len(k_vals)), labels=[f"{k:.3f}" for k in k_vals])

    plt.xlabel("Sigma (Noise Level)")
    plt.ylabel("K_eff (Coupling)")
    plt.title("Median Seizing Nodes vs Coupling and Noise")

    out_path = Path("results/sweep_heatmap.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nHeatmap saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
