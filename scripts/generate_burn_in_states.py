import argparse
from pathlib import Path

import numpy as np

from neuro.connectome import load_connectome
from neuro.jansen_rit import JansenRitParams, sigmoid_jit, simulate_network


def generate_burn_in_states(  # noqa: D103
    output_file: str = "burn_in_states.npz",
    duration: float = 60.0,
    dt: float = 1e-4,
    n_samples: int = 10,
    seed: int = 69,
) -> None:
    print("Loading connectome...")
    connectome = load_connectome()

    # 1. Setup parameters with no seizure nodes (A = A_bg = 3.25 for all nodes)
    # The default JRParams has A=3.25 if we just don't pass an array
    p = JansenRitParams(
        w_weights=connectome.weights,
        delay_steps=np.round(connectome.tract_lengths / connectome.speed / (dt * 1000)).astype(np.int64),
        A=3.25,
        sigma=0.001,
    )

    print(f"Running burn-in simulation for {duration} seconds...")
    _t, x_traj = simulate_network(params=p, duration=duration, dt=dt, seed=seed)

    n_steps = x_traj.shape[2] - 1
    max_history_len = int(np.max(p.delay_steps)) + 1

    print(f"Simulation done. n_steps = {n_steps}, max_history_len = {max_history_len}")

    # Exclude the first 10 seconds to avoid initial transient
    min_idx = max(int(10.0 / dt), max_history_len)

    rng = np.random.default_rng(seed + 1)
    # Pick n_samples random indices
    sample_indices = rng.choice(np.arange(min_idx, n_steps), size=n_samples, replace=False)

    x0_list = []
    history0_list = []

    y_traj = x_traj[1] - x_traj[2]  # Shape: (N, T)
    # Pre-calculate S(y_traj) to save time, or calculate on the fly

    for T in sample_indices:
        x0 = x_traj[:, :, T]
        # history slice: from T - max_history_len to T - 1
        y_slice = y_traj[:, T - max_history_len : T]

        # apply sigmoid: sigmoid_jit takes (y, e0, v0, r)
        # Note: y_slice is (N, max_history_len). history0 needs to be (max_history_len, N)
        s_y_slice = sigmoid_jit(y_slice.T, p.e0, p.v0, p.r)

        x0_list.append(x0)
        history0_list.append(s_y_slice)

    x0_arr = np.stack(x0_list)  # (n_samples, 6, N)
    history0_arr = np.stack(history0_list)  # (n_samples, max_history_len, N)

    np.savez_compressed(output_file, x0=x0_arr, history0=history0_arr, indices=sample_indices)
    print(f"Saved {n_samples} samples to {output_file}")
    print(f"x0 shape: {x0_arr.shape}, history0 shape: {history0_arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate burn-in states for initialization")
    parser.add_argument("--output", type=str, default="data/burn_in_states.npz")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    # Ensure directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_burn_in_states(output_file=args.output, duration=args.duration, n_samples=args.n_samples)
