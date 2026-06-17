"""Collect whole-brain plant simulation data for reduced-model fitting.

Runs the 76-region Jansen-Rit plant in the seizure regime (EZ ``lHC``/``lPHC``/``lAMYG``
at ``A=3.6``, PZ ``lTCI``/``lTCV`` at ``A=3.4``, rest ``A=3.25`` -- identical to
``scripts/run_channel_selection.py``) as an ensemble of short trials with varied seeds, and
saves raw data so either a channel-space (oscillator == EEG channel) or a region-space
(oscillator == region) reduced model can be fit offline for MPC.

Two dataset families are written to separate subdirectories:

* ``autonomous`` -- no tES stimulation (``u = 0``): the free seizure dynamics.
* ``excited`` -- a persistently-exciting tES current (random-amplitude steps, a random
  binary signal, or a multisine) to identify the input->output response.

Each trial ``.npz`` stores the regional output ``y_full (76, T)``, the 62-channel EEG at the
full ``10 kHz`` rate plus a decimated copy, the per-electrode tES input ``u``, and the
initial/final ``(6, 76)`` states (for exact restart). The full ``(6, 76)`` state is not kept
at full rate (~146 MB/trial); pass ``--save-full-state`` for a decimated copy. A shared
``meta.npz`` (regime, lead field, gamma, structural weights) and a ``manifest.json`` (git
commit, seeds, CLI args) tie the trials together.

Input/output time alignment: ``u[:, k]`` is the tES current applied during the step from
``t[k]`` to ``t[k + 1]`` (zero-order hold), i.e. pair as ``y[k + 1] = f(y[k], u[k])``.

Usage
-----
    uv run python scripts/run_plant_dataset.py
    uv run python scripts/run_plant_dataset.py --which excited --n-trials 10 --input-type prbs
    uv run python scripts/run_plant_dataset.py --which both --n-trials 2 --trial-duration 1.5 \
        --output-dir simulations/plant_dataset_smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import decimate

from neuro.connectome import Connectome, compute_gamma, load_connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, output
from utils.processing import steady_window

FloatArray = npt.NDArray[np.float64]

_DT = 1e-4
_K = 0.75
_SPEED = 50.0
_N_STATES = 6
_A_HEALTHY = 3.25
_A_EZ = 3.6
_A_PZ = 3.4
_EZ_REGIONS = ("lHC", "lPHC", "lAMYG")
_PZ_REGIONS = ("lTCI", "lTCV")

# Decouple the tES-input RNG stream from the plant-noise seed so the two families can share
# the same noise realisation (isolating the effect of stimulation) while drawing distinct inputs.
_INPUT_SEED_OFFSET = 100_000
_NO_INPUT_SEED = -1
_MULTISINE_F_MIN_HZ = 1
_MULTISINE_F_MAX_HZ = 15
_EPS = 1e-12

OUTPUT_DIR = Path("simulations/plant_dataset")


def _git_commit() -> str:
    """Return the current git commit hash, or a placeholder when not in a repository."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)  # noqa: S607
            .decode("ascii")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "not_a_git_repository"
    else:
        return commit


def _build_a_gains(connectome: Connectome) -> FloatArray:
    """Build the per-region excitatory-gain vector for the EZ/PZ seizure regime."""
    a_gains = np.full(len(connectome.region_labels), _A_HEALTHY)
    a_gains[[connectome.region_index[name] for name in _EZ_REGIONS]] = _A_EZ
    a_gains[[connectome.region_index[name] for name in _PZ_REGIONS]] = _A_PZ
    return a_gains


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of ``n`` (for staged, high-quality decimation)."""
    factors: list[int] = []
    remaining = n
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return factors


def _decimate(signal: FloatArray, q: int) -> FloatArray:
    """Anti-alias and downsample ``signal`` by integer factor ``q`` along the last axis."""
    out = signal
    for factor in _prime_factors(q):
        out = decimate(out, factor, ftype="iir", zero_phase=True, axis=-1)
    return out


def _multisine(n_samples: int, n_elec: int, amp: float, dt: float, rng: np.random.Generator) -> FloatArray:
    """Build a random-phase multisine of peak amplitude ``amp``, one column per electrode."""
    t = np.arange(n_samples) * dt
    freqs = np.arange(_MULTISINE_F_MIN_HZ, _MULTISINE_F_MAX_HZ + 1)
    out = np.zeros((n_samples, n_elec))
    for elec in range(n_elec):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
        sig = np.sin(2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
        out[:, elec] = amp * sig / max(np.abs(sig).max(), _EPS)
    return out


def _build_input_schedule(  # noqa: PLR0913
    *,
    input_type: str,
    n_steps: int,
    transient_steps: int,
    n_elec: int,
    amp: float,
    hold_ms: float,
    dt: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Build the per-step tES schedule ``(n_steps, n_elec)``; zero during the leading transient."""
    u = np.zeros((n_steps, n_elec))
    active = n_steps - transient_steps
    if active <= 0:
        return u

    if input_type in ("ras", "prbs"):
        hold = max(1, round(hold_ms / (dt * 1000.0)))
        n_blocks = (active + hold - 1) // hold
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_elec))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_elec))
        seq = np.repeat(block_vals, hold, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_elec, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    u[transient_steps:] = seq
    return u


def _simulate_trial(  # noqa: PLR0913
    *,
    a_gains: FloatArray,
    connectome: Connectome,
    seed: int,
    u_sched: FloatArray,
    n_steps: int,
    dt: float,
) -> FloatArray:
    """Integrate one trial and return the state trajectory ``(6, N, n_steps + 1)``."""
    dyn = JansenRitDynamics(dt=dt, params=JansenRitParams(A=a_gains), connectome=connectome, K=_K, seed=seed)
    n_nodes = dyn.x.shape[1]
    x_traj = np.zeros((_N_STATES, n_nodes, n_steps + 1))
    x_traj[:, :, 0] = dyn.x
    for k in range(n_steps):
        dyn.evaluate(k * dt, u_sched[k])
        x_traj[:, :, k + 1] = dyn.x
    return x_traj


def _save_trial(  # noqa: PLR0913
    *,
    path: Path,
    x_traj: FloatArray,
    u_sched: FloatArray,
    connectome: Connectome,
    dt: float,
    transient_ms: float,
    eeg_fs: float,
    save_full_state: bool,
    state_fs: float,
    seed: int,
    input_seed: int,
    family: str,
) -> dict[str, Any]:
    """Derive EEG/output, drop the transient, decimate, and write one trial ``.npz``."""
    n_total = x_traj.shape[2]
    n_steps = n_total - 1
    n_elec = u_sched.shape[1]
    dt_ms = dt * 1000.0

    y_full_all = output(x_traj)
    eeg_all = connectome.gain @ y_full_all
    t_all = np.arange(n_total) * dt
    # Align u on the output time grid (ZOH); the final column has no following step.
    u_all = np.zeros((n_elec, n_total))
    u_all[:, :n_steps] = u_sched.T
    u_all[:, n_steps] = u_sched[n_steps - 1]

    y_full = steady_window(y_full_all, dt_ms, transient_ms)
    eeg = steady_window(eeg_all, dt_ms, transient_ms)
    u = steady_window(u_all, dt_ms, transient_ms)
    t = steady_window(t_all, dt_ms, transient_ms)
    n_drop = n_total - y_full.shape[1]

    q = max(1, round(1.0 / (eeg_fs * dt)))
    eeg_ds = _decimate(eeg, q)
    t_ds = t[0] + np.arange(eeg_ds.shape[1]) * q * dt

    arrays: dict[str, Any] = {
        "t": t,
        "y_full": y_full,
        "eeg": eeg,
        "t_ds": t_ds,
        "eeg_ds": eeg_ds,
        "eeg_fs": 1.0 / (q * dt),
        "u": u,
        "x_init": x_traj[:, :, n_drop],
        "x_final": x_traj[:, :, -1],
        "seed": seed,
        "input_seed": input_seed,
        "family": family,
    }
    if save_full_state:
        q_state = max(1, round(1.0 / (state_fs * dt)))
        arrays["state_ds"] = _decimate(x_traj[:, :, n_drop:], q_state)
        arrays["t_state_ds"] = t[0] + np.arange(arrays["state_ds"].shape[2]) * q_state * dt

    np.savez_compressed(path, **arrays)
    return {
        "file": path.name,
        "seed": int(seed),
        "input_seed": int(input_seed),
        "n_samples": int(t.shape[0]),
        "n_samples_ds": int(t_ds.shape[0]),
    }


def _run_family(  # noqa: PLR0913
    *,
    family: str,
    fam_dir: Path,
    connectome: Connectome,
    a_gains: FloatArray,
    n_steps: int,
    transient_ms: float,
    n_trials: int,
    seed_base: int,
    input_type: str,
    stim_amp: float,
    hold_ms: float,
    eeg_fs: float,
    save_full_state: bool,
    state_fs: float,
) -> list[dict[str, Any]]:
    """Simulate and save every trial of one dataset family, returning per-trial records."""
    fam_dir.mkdir(parents=True, exist_ok=True)
    transient_steps = round(transient_ms / (_DT * 1000.0))
    n_elec = 1 if connectome.gamma is None else connectome.gamma.shape[0]

    records: list[dict[str, Any]] = []
    for idx in range(n_trials):
        seed = seed_base + idx
        if family == "excited":
            input_seed = seed_base + idx + _INPUT_SEED_OFFSET
            u_sched = _build_input_schedule(
                input_type=input_type,
                n_steps=n_steps,
                transient_steps=transient_steps,
                n_elec=n_elec,
                amp=stim_amp,
                hold_ms=hold_ms,
                dt=_DT,
                rng=np.random.default_rng(input_seed),
            )
        else:
            input_seed = _NO_INPUT_SEED
            u_sched = np.zeros((n_steps, n_elec))

        print(f"  [{family}] trial {idx + 1}/{n_trials}  seed={seed}")
        x_traj = _simulate_trial(
            a_gains=a_gains, connectome=connectome, seed=seed, u_sched=u_sched, n_steps=n_steps, dt=_DT
        )
        rec = _save_trial(
            path=fam_dir / f"trial_{family}_{idx:03d}.npz",
            x_traj=x_traj,
            u_sched=u_sched,
            connectome=connectome,
            dt=_DT,
            transient_ms=transient_ms,
            eeg_fs=eeg_fs,
            save_full_state=save_full_state,
            state_fs=state_fs,
            seed=seed,
            input_seed=input_seed,
            family=family,
        )
        records.append(rec)
    return records


def _save_meta(  # noqa: PLR0913
    *,
    path: Path,
    connectome: Connectome,
    a_gains: FloatArray,
    gamma: FloatArray,
    stim_electrodes: list[str],
    gamma_sigma: float,
    sigma: float,
) -> None:
    """Write the shared ``meta.npz`` (regime, lead field, gamma, structural weights)."""
    np.savez_compressed(
        path,
        a_gains=a_gains,
        region_labels=connectome.region_labels,
        channel_labels=connectome.channel_labels,
        ez_region_indices=np.array([connectome.region_index[name] for name in _EZ_REGIONS]),
        ez_region_labels=np.array(_EZ_REGIONS),
        pz_region_indices=np.array([connectome.region_index[name] for name in _PZ_REGIONS]),
        pz_region_labels=np.array(_PZ_REGIONS),
        gain=connectome.gain,
        gamma=gamma,
        stim_electrode_labels=np.array(stim_electrodes),
        weights=connectome.weights,
        K=_K,
        sigma=sigma,
        dt=_DT,
        speed=_SPEED,
        gamma_sigma=gamma_sigma,
    )


def _write_manifest(*, path: Path, families: dict[str, Any], cli_args: dict[str, Any]) -> None:
    """Write ``manifest.json`` tying the trials together with run provenance."""
    manifest = {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "dt": _DT,
        "K": _K,
        "regime": {
            "healthy_A": _A_HEALTHY,
            "ez_A": _A_EZ,
            "ez_regions": list(_EZ_REGIONS),
            "pz_A": _A_PZ,
            "pz_regions": list(_PZ_REGIONS),
        },
        "cli_args": cli_args,
        "families": families,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4)


def main(  # noqa: PLR0913
    *,
    which: str = "both",
    n_trials: int = 20,
    trial_duration: float = 5.0,
    transient_ms: float = 1000.0,
    seed_base: int = 42,
    input_type: str = "ras",
    stim_amp: float = 50.0,
    hold_ms: float = 50.0,
    stim_electrodes: list[str] | None = None,
    gamma_sigma: float = 20.0,
    eeg_fs: float = 250.0,
    save_full_state: bool = False,
    state_fs: float = 1000.0,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Collect the autonomous and/or excited seizure datasets and write meta + manifest."""
    electrodes = stim_electrodes if stim_electrodes is not None else ["CP5", "T7"]
    output_dir.mkdir(parents=True, exist_ok=True)

    connectome = load_connectome(speed=_SPEED)
    a_gains = _build_a_gains(connectome)
    gamma = np.atleast_2d(compute_gamma(connectome.centres, electrodes, gamma_sigma))
    connectome_stim = replace(connectome, gamma=gamma)
    n_steps = round(trial_duration / _DT)

    families = ["autonomous", "excited"] if which == "both" else [which]
    print(f"Collecting {families} ({n_trials} trials each, {trial_duration} s, dt={_DT}, K={_K})")

    family_records: dict[str, Any] = {}
    for family in families:
        family_records[family] = _run_family(
            family=family,
            fam_dir=output_dir / family,
            connectome=connectome_stim if family == "excited" else connectome,
            a_gains=a_gains,
            n_steps=n_steps,
            transient_ms=transient_ms,
            n_trials=n_trials,
            seed_base=seed_base,
            input_type=input_type,
            stim_amp=stim_amp,
            hold_ms=hold_ms,
            eeg_fs=eeg_fs,
            save_full_state=save_full_state,
            state_fs=state_fs,
        )

    _save_meta(
        path=output_dir / "meta.npz",
        connectome=connectome,
        a_gains=a_gains,
        gamma=gamma,
        stim_electrodes=electrodes,
        gamma_sigma=gamma_sigma,
        sigma=float(JansenRitParams(A=a_gains).sigma),
    )
    cli_args = {
        "which": which,
        "n_trials": n_trials,
        "trial_duration": trial_duration,
        "transient_ms": transient_ms,
        "seed_base": seed_base,
        "input_type": input_type,
        "stim_amp": stim_amp,
        "hold_ms": hold_ms,
        "stim_electrodes": electrodes,
        "gamma_sigma": gamma_sigma,
        "eeg_fs": eeg_fs,
        "save_full_state": save_full_state,
        "state_fs": state_fs,
        "output_dir": str(output_dir),
    }
    _write_manifest(path=output_dir / "manifest.json", families=family_records, cli_args=cli_args)
    print(f"\nWrote {output_dir} (meta.npz, manifest.json, {', '.join(families)})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plant-dataset collector."""
    parser = argparse.ArgumentParser(description="Collect whole-brain plant simulation data for reduced-model fitting.")
    parser.add_argument("--which", choices=("autonomous", "excited", "both"), default="both", help="Families to run.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials per family.")
    parser.add_argument("--trial-duration", type=float, default=5.0, help="Per-trial duration in seconds.")
    parser.add_argument(
        "--transient-ms", type=float, default=1000.0, help="Leading transient to discard (and skip stim)."
    )
    parser.add_argument("--seed-base", type=int, default=42, help="Plant seed for trial 0; incremented per trial.")
    parser.add_argument(
        "--input-type", choices=("ras", "prbs", "multisine"), default="ras", help="Excitation signal (excited family)."
    )
    parser.add_argument(
        "--stim-amp", type=float, default=50.0, help="tES amplitude; tune so EEG responds without saturating."
    )
    parser.add_argument("--hold-ms", type=float, default=50.0, help="Hold time per level for ras/prbs inputs.")
    parser.add_argument("--stim-electrodes", type=str, default="CP5,T7", help="Comma-separated tES electrode labels.")
    parser.add_argument("--gamma-sigma", type=float, default=20.0, help="tES spatial spread in mm.")
    parser.add_argument("--eeg-fs", type=float, default=250.0, help="Downsampled EEG rate in Hz (decimated copy).")
    parser.add_argument(
        "--save-full-state", action="store_true", help="Also save a decimated (6, 76) state trajectory."
    )
    parser.add_argument("--state-fs", type=float, default=1000.0, help="Rate for the decimated full state in Hz.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    main(
        which=cli.which,
        n_trials=cli.n_trials,
        trial_duration=cli.trial_duration,
        transient_ms=cli.transient_ms,
        seed_base=cli.seed_base,
        input_type=cli.input_type,
        stim_amp=cli.stim_amp,
        hold_ms=cli.hold_ms,
        stim_electrodes=[e.strip() for e in cli.stim_electrodes.split(",") if e.strip()],
        gamma_sigma=cli.gamma_sigma,
        eeg_fs=cli.eeg_fs,
        save_full_state=cli.save_full_state,
        state_fs=cli.state_fs,
        output_dir=cli.output_dir,
    )
