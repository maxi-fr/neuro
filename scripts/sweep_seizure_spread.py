"""Grid-search the global coupling ``K`` and the noise ``sigma`` for a slow seizure spread.

Both knobs push the same way -- they are the two sources of the drive that carries the
seizure out of the EZ -- so the pair that recruits the left hemisphere over ~10 s instead of
instantly lies on a ridge, not at a point. This script maps the ridge: it runs the EZ/PZ
regime over a ``K x sigma`` grid with several noise seeds each, records the per-region
recruitment envelope, and ranks the cells with :meth:`neuro.seizure.SpreadSummary.score`.

The full amplitude envelope is saved, so ``notebooks/seizure_spread_search.py`` can replot
the search and re-threshold it without re-simulating::

    uv run python scripts/sweep_seizure_spread.py --seeds 5 --duration 20
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, resting_state
from neuro.seizure import (
    DT,
    SPEED,
    SPREAD_HOP_S,
    SPREAD_WINDOW_S,
    SpreadProfile,
    build_seizure_a_gains,
    spread_profile,
    spread_summary,
)
from neuro.types import FloatArray

_DEFAULT_K = (0.45, 0.50, 0.5357, 0.60, 0.70, 0.85)
_DEFAULT_SIGMA = (100.0, 200.0, 300.0, 400.0, 500.0)


def _run_cell(args: tuple[float, float, int, float, bool]) -> FloatArray:
    """Simulate one (K, sigma, seed) cell and return its ``(n_nodes, n_windows)`` envelope."""
    k, sigma, seed, duration, from_rest = args
    conn = replace(_connectome(), K=k)
    params = JansenRitParams(A=build_seizure_a_gains(conn), sigma=sigma)
    initial_state = resting_state(conn, DT) if from_rest else None
    dyn = JansenRitDynamics(dt=DT, params=params, conn=conn, seed=seed, initial_state=initial_state)
    return spread_profile(dyn, duration).ptp


_CONNECTOME: Connectome | None = None


def _connectome() -> Connectome:
    """Return the process-local connectome, loading it from TVB on first use.

    Each worker process pays the TVB load once instead of once per grid cell, and the
    connectome is not pickled across the process boundary.
    """
    global _CONNECTOME  # noqa: PLW0603
    if _CONNECTOME is None:
        _CONNECTOME = Connectome.from_config({"speed": SPEED})
    return _CONNECTOME


def main() -> None:
    """Run the K x sigma grid and save the envelopes plus the ranked summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, nargs="+", default=list(_DEFAULT_K), help="Global coupling values.")
    parser.add_argument("--sigma", type=float, nargs="+", default=list(_DEFAULT_SIGMA), help="Noise std values.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of noise seeds per grid cell.")
    parser.add_argument("--seed0", type=int, default=69, help="First noise seed; the rest follow consecutively.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulated seconds per run.")
    # Each worker holds its own TVB dataset (~0.5 GB), so the grid is memory- rather than
    # core-bound; raise --workers only if there is headroom for it.
    parser.add_argument("--workers", type=int, default=4, help="Worker processes.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for the NPZ.")
    parser.add_argument(
        "--from-zero",
        action="store_true",
        help="Start from all-zeros (the configs' default) instead of the healthy resting state.",
    )
    args = parser.parse_args()

    k_vals = np.asarray(args.k, dtype=np.float64)
    sigma_vals = np.asarray(args.sigma, dtype=np.float64)
    seeds = np.arange(args.seed0, args.seed0 + args.seeds)

    conn = _connectome()
    from_rest = not args.from_zero
    jobs = [
        (float(k), float(s), int(seed), args.duration, from_rest) for k in k_vals for s in sigma_vals for seed in seeds
    ]
    print(
        f"{len(jobs)} runs of {args.duration:.0f} s over {len(k_vals)} K x {len(sigma_vals)} sigma x "
        f"{len(seeds)} seeds, starting from {'the healthy resting state' if from_rest else 'all-zeros'}"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        envelopes = list(pool.map(_run_cell, jobs, chunksize=1))

    ptp = np.stack(envelopes).reshape(len(k_vals), len(sigma_vals), len(seeds), *envelopes[0].shape)
    # spread_profile centres its first window half a window into the run, then hops.
    times = SPREAD_WINDOW_S / 2.0 + np.arange(ptp.shape[-1]) * SPREAD_HOP_S

    out_dir = args.out or Path(f"artifacts/seizure_spread_{datetime.now(UTC).astimezone():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "sweep.npz",
        k_vals=k_vals,
        sigma_vals=sigma_vals,
        seeds=seeds,
        duration=args.duration,
        from_rest=from_rest,
        times=times,
        ptp=ptp,
        region_labels=conn.region_labels,
        hemispheres=conn.hemispheres,
    )

    _report(ptp, times, k_vals, sigma_vals, conn, args.duration)
    print(f"\nSaved {ptp.nbytes / 1e6:.0f} MB of envelopes to {(out_dir / 'sweep.npz').resolve()}")


def _report(  # noqa: PLR0913
    ptp: FloatArray,
    times: FloatArray,
    k_vals: FloatArray,
    sigma_vals: FloatArray,
    conn: Connectome,
    duration: float,
) -> None:
    """Print the seed-averaged summary of every grid cell, best score first."""
    rows = []
    for i, k in enumerate(k_vals):
        for j, sigma in enumerate(sigma_vals):
            summaries = [spread_summary(SpreadProfile.from_ptp(times, ptp[i, j, s]), conn) for s in range(ptp.shape[2])]
            # A region that never seizes has a NaN onset; averaging it in as `duration`
            # keeps the column readable as "how late, at worst" instead of collapsing to NaN.
            mean = {
                key: float(np.mean([np.nan_to_num(asdict(s)[key], nan=duration) for s in summaries]))
                for key in asdict(summaries[0])
            }
            score = float(np.mean([s.score(duration) for s in summaries]))
            rows.append((score, k, sigma, mean))

    header = f"{'score':>7} {'K':>7} {'sigma':>7} {'t_ez':>7} {'t_pz':>7} {'t_left/2':>9} {'left':>6} {'right':>6}"
    print(f"\n{header}\n{'-' * len(header)}")
    for score, k, sigma, m in sorted(rows, key=lambda r: r[0]):
        print(
            f"{score:7.3f} {k:7.4f} {sigma:7.0f} {m['t_ez']:7.2f} {m['t_pz']:7.2f} "
            f"{m['t_left_half']:9.2f} {m['frac_left']:6.2f} {m['frac_right']:6.2f}"
        )


if __name__ == "__main__":
    main()
