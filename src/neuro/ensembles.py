from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import decimate
from simulate.config import load_config as load_sim_config

from neuro.connectome import Connectome
from neuro.eeg import build_eeg_gain, focal_channels
from neuro.jansen_rit import JansenRitDynamics, lfp, simulate_network
from neuro.metrics import (
    DEFAULT_HOP_S,
    METRICS,
    Ensemble,
    baseline_grid,
    baseline_r2,
    score_store,
    state_store,
)
from neuro.seizure import A_HEALTHY, SPREAD_WINDOW_S, build_seizure_a_gains, spread_profile_from_lfp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neuro.types import FloatArray

PLANT_CONFIG = Path("configs/simulation/nonlinear_full_mpc.yaml")
GOOD_COMMAND = (2.0, 0.0, -2.0)

PlantKind = Literal["healthy", "seizure"]


@dataclass(frozen=True)
class Arm:
    """One open-loop stimulation condition, held for the whole rollout.

    Attributes
    ----------
    name
        Output key.
    current
        Per-electrode current in mA, one entry per control electrode. Must sum to zero.
    """

    name: str
    current: tuple[float, ...]


@dataclass(frozen=True)
class Branch:
    """One operating point to branch ensembles from.

    Attributes
    ----------
    name
        Output key; one ``.npy`` per branch and arm is named after it.
    plant
        Which ``A`` vector the parent runs under.
    t_branch
        Seconds of parent run before the snapshot is taken.
    arms
        Stimulation conditions allocated to this branch.
    """

    name: str
    plant: PlantKind
    t_branch: float
    arms: tuple[Arm, ...]


ARM_ZERO = Arm("zero", (0.0, 0.0, 0.0))
ARM_D1 = Arm("d1", GOOD_COMMAND)
# 0.0 - c rather than -c: negating the null electrode would render as -0.0.
ARM_NEG_D1 = Arm("-d1", tuple(0.0 - c for c in GOOD_COMMAND))

ARMS: tuple[Arm, ...] = (ARM_ZERO, ARM_D1, ARM_NEG_D1)

BRANCHES: tuple[Branch, ...] = (
    Branch("healthy", "healthy", 4.0, arms=(ARM_ZERO,)),
    Branch("pre_onset", "seizure", 1.0, arms=(ARM_ZERO, ARM_D1, ARM_NEG_D1)),
    Branch("ez_ignited", "seizure", 3.0, arms=(ARM_ZERO, ARM_D1, ARM_NEG_D1)),
    Branch("mid_spread", "seizure", 7.0, arms=(ARM_ZERO, ARM_D1)),
    Branch("saturated", "seizure", 14.0, arms=(ARM_ZERO,)),
)


def all_arms(branches: tuple[Branch, ...]) -> tuple[Arm, ...]:
    """Collect the unique arms across ``branches``, preserving first-seen order."""
    seen: dict[str, Arm] = {}
    for branch in branches:
        seen.update({arm.name: arm for arm in branch.arms if arm.name not in seen})
    return tuple(seen.values())


@dataclass(frozen=True)
class EnsembleConfig:
    """What to generate: how many parents and children, over which branches and arms.

    Attributes
    ----------
    n_parents
        Distinct plant seeds per plant kind. They supply the ``Var_across_parents`` denominator
        of the R2 score and make the healthy/saturated separability statistic defined at all,
        so this is not a knob to drop to 1.
    n_children
        Noise realisations branched from each parent snapshot.
    rollout_s
        Length of each child rollout in seconds.
    branches
        Operating points to branch from, each carrying its own stimulation arms.
    parent_seed_base, child_seed_base
        Roots of the two seed sequences; child seeds are shared across arms by construction.
    region_fs
        Sampling rate the region LFP is decimated to before storage, in Hz.
    focus_region
        Region whose hardest-loading channels form the focal channel set.
    n_focal_channels
        Size of that focal channel set.
    chunk_s
        Granularity the parent run is stepped in; only bounds peak memory.
    """

    n_parents: int = 16
    n_children: int = 8
    rollout_s: float = 3.0
    branches: tuple[Branch, ...] = BRANCHES
    parent_seed_base: int = 69
    child_seed_base: int = 1_000_000
    region_fs: float = 1000.0
    focus_region: str = "lTCI"
    n_focal_channels: int = 4
    chunk_s: float = 1.0

    def __post_init__(self) -> None:
        """Reject a branch too early to carry the pre-branch history the seizure state needs."""
        for branch in self.branches:
            if branch.t_branch < SPREAD_WINDOW_S:
                msg = (
                    f"Branch '{branch.name}' branches at {branch.t_branch} s, shorter than the "
                    f"{SPREAD_WINDOW_S} s of pre-branch history each region rollout carries."
                )
                raise ValueError(msg)

    def child_seed(self, parent: int, child: int) -> int:
        """Seed of one child, shared by every arm so the arms are common-random-number coupled."""
        return self.child_seed_base + parent * self.n_children + child

    def parent_seed(self, plant: PlantKind, parent: int) -> int:
        """Seed of one parent run; the two plant kinds get disjoint blocks."""
        return self.parent_seed_base + parent + (0 if plant == "healthy" else 1000)

    def branches_of(self, plant: PlantKind) -> tuple[Branch, ...]:
        """Return the branches taken from a parent of this plant kind, ordered by ``t_branch``."""
        return tuple(sorted((b for b in self.branches if b.plant == plant), key=lambda b: b.t_branch))

    @property
    def n_rollouts(self) -> int:
        """Rollouts stored per branch and arm, indexed ``parent * n_children + child``."""
        return self.n_parents * self.n_children


@dataclass
class PlantPair:
    """The healthy and seizure plants, built once and branched from thereafter.

    They are constructed from a single dynamics block and differ **only** in ``params.A``, so
    the healthy/seizure contrast carries no other confound. Both carry the resting state the
    config asks for, which is deterministic, so every parent can be a reseeded copy.
    """

    healthy: JansenRitDynamics
    seizure: JansenRitDynamics
    connectome: Connectome
    eeg_gain: FloatArray
    channel_labels: list[str]

    def of(self, plant: PlantKind) -> JansenRitDynamics:
        """Return the plant of the requested kind."""
        return self.healthy if plant == "healthy" else self.seizure


def build_plants(config_path: Path = PLANT_CONFIG) -> PlantPair:
    """Build the healthy and seizure plants from one config's ``dynamics`` block.

    Raises
    ------
    ValueError
        If the config's ``A`` vector is not the EZ/PZ gain vector :mod:`neuro.seizure` defines,
        which would mean the seizure plant is not the one the rest of the repo reasons about.
    """
    # class_path selects the plant for the simulate orchestrator; from_config forbids it as a key.
    dynamics_cfg = {k: v for k, v in load_sim_config(config_path)["dynamics"].items() if k != "class_path"}
    conn = Connectome.from_config(dynamics_cfg["connectome"])

    expected = build_seizure_a_gains(conn)
    if not np.allclose(np.asarray(dynamics_cfg["params"]["A"], dtype=np.float64), expected):
        msg = f"{config_path} does not carry the EZ/PZ A vector of neuro.seizure.build_seizure_a_gains"
        raise ValueError(msg)

    seizure = JansenRitDynamics.from_config(dynamics_cfg)
    healthy = deepcopy(seizure)
    healthy.net_params = replace(seizure.net_params, A=np.full(len(expected), A_HEALTHY))
    healthy.params_tuple = healthy.net_params.to_numba_tuple(len(expected))

    gain, channel_labels = build_eeg_gain()
    return PlantPair(
        healthy=healthy,
        seizure=seizure,
        connectome=conn,
        eeg_gain=np.asarray(gain[:, : len(conn.region_labels)], dtype=np.float64),
        channel_labels=[str(label) for label in channel_labels],
    )


def reseed(dyn: JansenRitDynamics, seed: int) -> JansenRitDynamics:
    """Copy ``dyn`` whole and give the copy a fresh noise stream.

    ``deepcopy`` rather than the ``initial_state=`` constructor argument on purpose. The plant
    state is ``x`` **plus** the delay ``history`` ring buffer **plus** the step counter carried
    in ``next_update_time`` -- ``initial_state=`` sets only ``x`` and refills ``history`` with a
    constant sigmoid of it, which is right from rest and silently wrong mid-seizure.
    """
    child = deepcopy(dyn)
    child.rng = np.random.default_rng(seed)
    return child


def _elapsed_steps(dyn: JansenRitDynamics) -> int:
    """Return steps the plant has already taken, so a continuation can resume on its own clock."""
    return round(dyn.next_update_time / dyn.dt)


def run_segment(dyn: JansenRitDynamics, n_steps: int, current: tuple[float, ...]) -> FloatArray:
    """Step ``dyn`` on from wherever it is, holding ``current``, and return the region LFP.

    Resumes at the plant's own clock, so the delay history and the noise stream continue rather
    than restart. The pre-existing state is dropped from the output, so the returned samples
    cover ``(t0, t0 + n_steps * dt]`` and consecutive segments concatenate without a seam.

    Returns
    -------
    FloatArray
        Region LFP ``y = x2 - x3`` in mV, shape ``(n_nodes, n_steps)``.
    """
    t0 = _elapsed_steps(dyn) * dyn.dt
    _, x_traj = simulate_network(
        dyn=dyn,
        duration=n_steps * dyn.dt,
        u_hat_tES=np.asarray(current, dtype=np.float64),
        stim_window=(t0, t0 + n_steps * dyn.dt),
        t0=t0,
    )
    return lfp(x_traj)[:, 1:]


@dataclass
class Parent:
    """The state of one parent run at its branch points and its recruitment ground truth.

    Attributes
    ----------
    index
        Parent index within its plant kind (0 to ``n_parents - 1``).
    plant
        Which plant kind generated this parent.
    seed
        Random seed the parent was initialised with.
    snapshots
        Deep copies of the plant taken at each branch's ``t_branch``, keyed by branch name.
    pre_regions
        Full-rate pre-branch region LFPs leading up to each snapshot, keyed by branch name. Kept
        undecimated so :func:`run_child` can decimate across the branch instant in one pass.
    n_seizing_final
        Regions seizing in the last spread window -- the basin label. The plant is bimodal
        across seeds, so pooling parents without this hides a mixture rather than averaging one.
    onsets
        Per-region recruitment time in seconds, ``NaN`` where a region never seizes.
    """

    index: int
    plant: PlantKind
    seed: int
    snapshots: dict[str, JansenRitDynamics]
    pre_regions: dict[str, FloatArray]
    n_seizing_final: int
    onsets: FloatArray


def _to_region_fs(y: FloatArray, dt: float, region_fs: float) -> npt.NDArray[np.float32]:
    """Decimate a full-rate region LFP to ``region_fs`` and narrow it to the stored float32.

    Zero-phase, which is legitimate here: the causal constraint applies to the control path, not
    to an artefact read back offline.
    """
    factor = round(1.0 / (dt * region_fs))
    return np.asarray(decimate(y, factor, axis=1) if factor > 1 else y, dtype=np.float32)


def _trailing(chunks: list[FloatArray], n: int) -> FloatArray:
    """Join the last ``n`` columns of ``chunks``, touching only the chunks that cover them.

    Equivalent to ``np.concatenate(chunks, axis=1)[:, -n:]``, but without copying the history
    ahead of the window -- 85 MB per branch by the 14 s one, which would defeat ``chunk_s``.
    """
    tail: list[FloatArray] = []
    need = n
    for chunk in reversed(chunks):
        taken = chunk if need >= chunk.shape[1] else chunk[:, chunk.shape[1] - need :]
        tail.append(taken)
        need -= taken.shape[1]
        if need == 0:
            break
    return np.concatenate(tail[::-1], axis=1)


def run_parent(plants: PlantPair, cfg: EnsembleConfig, plant: PlantKind, index: int) -> Parent:
    """Run one unstimulated parent to its last branch point, snapshotting on the way."""
    dyn = reseed(plants.of(plant), cfg.parent_seed(plant, index))
    zero = (0.0,) * dyn.n_controls

    n_chunk = round(cfg.chunk_s / dyn.dt)
    snapshots: dict[str, JansenRitDynamics] = {}
    pre_regions: dict[str, FloatArray] = {}
    y_chunks: list[FloatArray] = []
    n_pre = round(SPREAD_WINDOW_S / dyn.dt)

    for branch in cfg.branches_of(plant):
        target = round(branch.t_branch / dyn.dt)
        while _elapsed_steps(dyn) < target:
            y_chunks.append(run_segment(dyn, min(n_chunk, target - _elapsed_steps(dyn)), zero))
        snapshots[branch.name] = deepcopy(dyn)
        pre_regions[branch.name] = _trailing(y_chunks, n_pre)

    profile = spread_profile_from_lfp(np.concatenate(y_chunks, axis=1), dyn.dt)
    return Parent(
        index=index,
        plant=plant,
        seed=cfg.parent_seed(plant, index),
        snapshots=snapshots,
        pre_regions=pre_regions,
        n_seizing_final=int(profile.n_seizing()[-1]),
        onsets=profile.onsets,
    )


def run_child(  # noqa: PLR0913, PLR0917
    snapshot: JansenRitDynamics,
    plants: PlantPair,
    cfg: EnsembleConfig,
    seed: int,
    arm: Arm,
    pre_region: FloatArray,
) -> tuple[FloatArray, npt.NDArray[np.float32]]:
    """Branch one noise realisation off ``snapshot`` and roll it out under ``arm``.

    Parameters
    ----------
    snapshot
        The parent plant frozen at its branch point; it is copied, not advanced.
    plants
        The plant pair, for its EEG leadfield.
    cfg
        Rollout length and region sampling rate.
    seed
        Noise seed of this child, shared across arms.
    arm
        The stimulation condition held for the whole rollout.
    pre_region
        The parent's full-rate region LFP over the window ending at the branch, prepended to the
        rollout so the causal seizure-state window can evaluate from lookahead 0.

    Returns
    -------
    eeg
        Scalp EEG at the plant's own rate, shape ``(n_channels, n_samples)``.
    region
        ``pre_region`` joined to the rollout's region LFP and decimated to ``cfg.region_fs`` in a
        single pass, shape ``(n_nodes, n_samples_region)``. Decimating once rather than per
        segment keeps an anti-alias edge transient off the branch instant, which is exactly where
        the lookahead-0 seizure state is read.
    """
    dyn = reseed(snapshot, seed)
    y = run_segment(dyn, round(cfg.rollout_s / dyn.dt), arm.current)

    region = _to_region_fs(np.concatenate([pre_region, y], axis=1), dyn.dt, cfg.region_fs)
    return plants.eeg_gain @ y, region


def _open_store(path: Path, shape: tuple[int, ...], dtype: type[np.floating]) -> np.memmap:
    """Open a ``.npy`` file as a writable memmap, so a 1.9 GB array is never resident."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@dataclass
class _Stores:
    """The open output memmaps, keyed by ``(branch, arm)``."""

    eeg: dict[tuple[str, str], np.memmap] = field(default_factory=dict)
    region: dict[tuple[str, str], np.memmap] = field(default_factory=dict)

    def flush(self) -> None:
        """Push every store to disk and drop the handles."""
        for store in (*self.eeg.values(), *self.region.values()):
            store.flush()
        self.eeg.clear()
        self.region.clear()


def eeg_path(out_dir: Path, branch: str, arm: str) -> Path:
    """Path of the full-rate scalp EEG store for one branch and arm."""
    return out_dir / "eeg" / f"{branch}_{arm}.npy"


def region_path(out_dir: Path, branch: str, arm: str) -> Path:
    """Path of the decimated region LFP store for one branch and arm."""
    return out_dir / "region" / f"{branch}_{arm}.npy"


def _build_stores(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> _Stores:
    """Pre-allocate one store per branch and arm, in both spaces."""
    dt = plants.seizure.dt
    n_eeg = round(cfg.rollout_s / dt)
    n_region = round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs)
    n_channels = plants.eeg_gain.shape[0]
    n_nodes = plants.eeg_gain.shape[1]

    stores = _Stores()
    for branch in cfg.branches:
        for arm in branch.arms:
            key = (branch.name, arm.name)
            stores.eeg[key] = _open_store(eeg_path(out_dir, *key), (cfg.n_rollouts, n_channels, n_eeg), np.float64)
            stores.region[key] = _open_store(
                region_path(out_dir, *key), (cfg.n_rollouts, n_nodes, n_region), np.float32
            )
    return stores


def generate(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> dict[str, Any]:
    """Generate the whole ensemble into ``out_dir`` and return the manifest.

    Yields nothing and prints nothing; use :func:`generate_iter` to follow progress.
    """
    manifest: dict[str, Any] = {}
    for update in generate_iter(out_dir, cfg, plants):
        manifest = update
    return manifest


def generate_iter(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> Iterator[dict[str, Any]]:
    """Generate the ensemble, yielding the manifest-so-far after each parent completes."""
    stores = _build_stores(out_dir, cfg, plants)
    parents: list[Parent] = []

    for plant in ("healthy", "seizure"):
        branches = cfg.branches_of(plant)
        if not branches:
            continue
        for index in range(cfg.n_parents):
            parent = run_parent(plants, cfg, plant, index)
            for branch in branches:
                pre_region = parent.pre_regions[branch.name]
                for child in range(cfg.n_children):
                    seed = cfg.child_seed(index, child)
                    row = index * cfg.n_children + child
                    for arm in branch.arms:
                        eeg, region = run_child(parent.snapshots[branch.name], plants, cfg, seed, arm, pre_region)
                        stores.eeg[branch.name, arm.name][row] = eeg
                        stores.region[branch.name, arm.name][row] = region
            parent.snapshots.clear()
            parent.pre_regions.clear()
            parents.append(parent)
            yield _manifest(out_dir, cfg, plants, parents)

    stores.flush()
    yield _manifest(out_dir, cfg, plants, parents)


def _manifest(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair, parents: list[Parent]) -> dict[str, Any]:
    """Assemble and write the run manifest, and return it."""
    focus = plants.connectome.region_index[cfg.focus_region]
    focal = focal_channels(plants.eeg_gain, focus, cfg.n_focal_channels)

    manifest: dict[str, Any] = {
        "dt": plants.seizure.dt,
        "fs": 1.0 / plants.seizure.dt,
        "region_fs": cfg.region_fs,
        "rollout_s": cfg.rollout_s,
        # Provenance only -- the read side takes the convention from SPREAD_WINDOW_S.
        "pre_branch_s": SPREAD_WINDOW_S,
        "n_parents": cfg.n_parents,
        "n_children": cfg.n_children,
        "rollout_index": "parent * n_children + child",
        "branches": [
            {
                "name": b.name,
                "plant": b.plant,
                "t_branch": b.t_branch,
                "arms": [a.name for a in b.arms],
            }
            for b in cfg.branches
        ],
        "arms": [{"name": a.name, "current": list(a.current)} for a in all_arms(cfg.branches)],
        "electrode_labels": [str(label) for label in plants.seizure.stim.control_labels],
        "channel_labels": plants.channel_labels,
        "region_labels": [str(label) for label in plants.connectome.region_labels],
        "channel_sets": {
            "all62": list(range(plants.eeg_gain.shape[0])),
            cfg.focus_region: [int(i) for i in focal],
        },
        "parents": [
            {
                "index": p.index,
                "plant": p.plant,
                "seed": p.seed,
                "n_seizing_final": p.n_seizing_final,
                "onsets": [None if not np.isfinite(t) else float(t) for t in p.onsets],
            }
            for p in parents
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(out_dir: Path) -> dict[str, Any]:
    """Read back the manifest of a generated ensemble."""
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


SCORES_FILE = "scores.npz"
CACHE_DIR = "cache_1khz"

RAW = "raw"
CUTOFFS_HZ: tuple[float | None, ...] = (None, 500.0, 100.0, 45.0, 20.0, 10.0)
SWEPT_METRICS: tuple[str, ...] = ("block_ptp", "line_length", "eeg_ms", "fc_strength")


def cutoff_key(cutoff: float | None) -> str:
    """Archive key of one bandwidth-axis point; ``None`` is the unfiltered signal."""
    return RAW if cutoff is None else f"{cutoff:g}"


def cached_scalp(
    out_dir: Path,
    branch: str,
    arm: str,
    manifest: dict[str, Any],
    *,
    target_fs: float = 1000.0,
) -> FloatArray:
    """Build and return a decimated, float32 view of one scalp store, memory-mapped on repeat use.

    19 GB of full-rate EEG will not sit in a notebook kernel, and no plot needs it: every metric
    here tops out at 12 Hz of interest, so 1 kHz is ~40x oversampled. Decimation is zero-phase --
    the causal constraint that forces ``sosfilt`` in the control path is about the controller,
    not about a figure drawn afterwards.

    Scoring never reads this: :func:`score_ensemble_dir` goes to the full-rate store, because
    block PTP and line length both depend on the sampling rate.
    """
    cache = out_dir / CACHE_DIR / f"{branch}_{arm}.npy"
    if not cache.exists():
        source = np.load(eeg_path(out_dir, branch, arm), mmap_mode="r")
        factor = round(manifest["fs"] / target_fs)
        store = _open_store(cache, (*source.shape[:2], source.shape[2] // factor), np.float32)
        for row in range(source.shape[0]):
            store[row] = decimate(np.asarray(source[row], dtype=np.float64), factor, axis=1)[:, : store.shape[2]]
        store.flush()
    return np.load(cache, mmap_mode="r")


@dataclass(frozen=True)
class ScoreArchive:
    """Every metric series read off a generated ensemble, keyed and small enough to hold.

    Scoring the stores is a ~40 min sweep at full scale, dominated by the Welch transform the
    spectral metrics need; the resulting series are a few MB. So it is done once and cached,
    and everything downstream -- including the choice of ``h_eval`` -- is then instant.

    Attributes
    ----------
    times
        Lookahead grid per metric, in seconds since the branch. Metrics with different windows
        have different grids, which is why this is not one shared axis.
    series
        ``(branch, arm, metric, channel_set, cutoff) -> (n_rollouts, n_times)``, or
        ``(n_rollouts, n_channels, n_times)`` at the raw cutoff, where ``cutoff`` is a
        :func:`cutoff_key`. Scalp only: region space is read for the seizure state alone.
    baselines
        ``(branch, baseline, cutoff) -> (n_channels, n_times)``: per-channel R2 of the
        raw-signal rungs the metrics are ranked against. Cached as scores rather than series --
        the series would be ~500 MB against the archive's few, and would buy only a baseline
        ``d_ctrl`` that nothing needs.
    baseline_times
        Lookahead grid the baselines are read on. Finer than the metric grid at short lookahead,
        which is where waveform predictability lives and dies.
    states
        ``(branch, arm) -> (n_rollouts, n_times)``: the fraction of regions seizing, from
        :func:`~neuro.metrics.seizure_state`. The ground truth every metric is scored against,
        and -- read on the ``d1`` arm against the ``zero`` one -- the only measurement of whether
        the probe moves the seizure rather than merely the observable.
    state_times
        Lookahead grid the seizure state is read on.
    n_replicates
        Rollouts per state, carried so an :class:`~neuro.metrics.Ensemble` can be rebuilt.
    manifest
        The generating run's manifest.
    """

    times: dict[str, FloatArray]
    series: dict[tuple[str, str, str, str, str], FloatArray]
    baselines: dict[tuple[str, str, str], FloatArray]
    baseline_times: FloatArray
    states: dict[tuple[str, str], FloatArray]
    state_times: FloatArray
    n_replicates: int
    manifest: dict[str, Any]

    def ensemble(  # noqa: PLR0913
        self,
        branch: str,
        arm: str,
        metric: str,
        channel_set: str,
        cutoff: str = RAW,
        *,
        pool: bool = True,
    ) -> Ensemble:
        """Rebuild one scored ensemble, ready for the score functions.

        Only the raw cutoff is stored per channel -- the bandwidth sweep is a pooled axis, and 6
        cutoffs of per-channel series would be gigabytes for a question the sweep does not ask.
        So ``pool=False`` is meaningful only there; elsewhere the series is already pooled and
        the flag is a no-op.
        """
        values = np.asarray(self.series[branch, arm, metric, channel_set, cutoff], dtype=np.float64)
        if pool and values.ndim == 3:  # noqa: PLR2004
            values = values.mean(axis=1)
        return Ensemble(times=self.times[metric], values=values, n_replicates=self.n_replicates)

    def state(self, branch: str, arm: str) -> Ensemble:
        """Rebuild the seizure-state ensemble of one branch and arm."""
        return Ensemble(
            times=self.state_times,
            values=np.asarray(self.states[branch, arm], dtype=np.float64),
            n_replicates=self.n_replicates,
        )

    def baseline(self, branch: str, name: str, channel_set: str, cutoff: str = RAW) -> FloatArray:
        """Average one baseline's per-channel R2 over a channel set, giving a curve over `h`.

        Averaging the R2 rather than the signal is the whole reason the baselines are stored per
        channel: a channel-mean of a signed waveform cancels.
        """
        channels = self.manifest["channel_sets"][channel_set]
        return np.asarray(self.baselines[branch, name, cutoff][channels].mean(axis=0), dtype=np.float64)

    @property
    def channel_sets(self) -> list[str]:
        """Scalp channel-set names, in manifest order."""
        return list(self.manifest["channel_sets"])

    @property
    def cutoffs(self) -> list[str]:
        """Bandwidth-axis keys present in the archive, unfiltered first then widest to narrowest."""
        keys = {key[-1] for key in self.series}
        return [RAW, *sorted(keys - {RAW}, key=float, reverse=True)]


def score_ensemble_dir(
    out_dir: Path,
    *,
    hop_s: float = DEFAULT_HOP_S,
    cutoffs: tuple[float | None, ...] = CUTOFFS_HZ,
    rebuild: bool = False,
) -> ScoreArchive:
    """Score every stored rollout on every metric and bandwidth, caching the result beside the stores.

    Reads the stores memory-mapped, one rollout at a time, so the 19 GB artefact is never
    resident. Each cutoff costs one traversal of the scalp store for the metrics and, on the
    reference arm, one more for the baselines; within a traversal every metric and channel set is
    read off the same rollout rather than the store being re-read per metric.

    Only :data:`SWEPT_METRICS` are re-scored under a filter. ``band_power`` and
    ``spectral_centroid`` stay at the unfiltered signal because the low-pass redefines them
    rather than denoising them -- it mutilates a 3-12 Hz integral, and it moves a centroid down
    mechanically -- so sweeping them would not be one metric measured at several bandwidths.

    The region store is read once per branch and arm for the seizure state alone; no metric is
    scored in region space, that having been the superseded observability axis's reference.
    """
    manifest = load_manifest(out_dir)
    cache = out_dir / SCORES_FILE
    if cache.exists() and not rebuild:
        return _load_scores(cache, manifest)

    channel_sets = {name: np.asarray(idx, dtype=np.intp) for name, idx in manifest["channel_sets"].items()}
    n_replicates = int(manifest["n_children"])
    fs = float(manifest["fs"])

    swept = {name: METRICS[name] for name in SWEPT_METRICS}
    grid = baseline_grid(float(manifest["rollout_s"]), hop_s=hop_s)

    times: dict[str, FloatArray] = {}
    series: dict[tuple[str, str, str, str, str], FloatArray] = {}
    baselines: dict[tuple[str, str, str], FloatArray] = {}
    states: dict[tuple[str, str], FloatArray] = {}
    state_times = np.empty(0, dtype=np.float64)

    for branch_info in manifest["branches"]:
        branch = branch_info["name"]
        branch_arms = branch_info["arms"]
        for arm in branch_arms:
            scalp = np.load(eeg_path(out_dir, branch, arm), mmap_mode="r")
            for cutoff in cutoffs:
                key = cutoff_key(cutoff)
                metric_times, scored = score_store(
                    scalp,
                    fs,
                    metrics=METRICS if cutoff is None else swept,
                    channel_sets=channel_sets,
                    n_replicates=n_replicates,
                    hop_s=hop_s,
                    cutoff_hz=cutoff,
                    pool=cutoff is not None,
                )
                times.update(metric_times)
                for (name, set_name), ens in scored.items():
                    series[branch, arm, name, set_name, key] = ens.values
                if arm == branch_arms[0]:
                    scored_baselines = baseline_r2(scalp, fs, times=grid, n_replicates=n_replicates, cutoff_hz=cutoff)
                    baselines.update({(branch, name, key): r2 for name, r2 in scored_baselines.items()})

            region = np.load(region_path(out_dir, branch, arm), mmap_mode="r")
            state_times, states[branch, arm] = state_store(region, float(manifest["region_fs"]), hop_s=hop_s)

    archive = ScoreArchive(
        times=times,
        series=series,
        baselines=baselines,
        baseline_times=grid,
        states=states,
        state_times=state_times,
        n_replicates=n_replicates,
        manifest=manifest,
    )
    _save_scores(cache, archive)
    return archive


def _save_scores(cache: Path, archive: ScoreArchive) -> None:
    """Flatten a score archive into one compressed npz beside the stores.

    Series are narrowed to float32 on the way out. They are scores derived from a float32 region
    store and a chaotic plant, so the seventh significant digit is not information; keeping the
    per-channel raw-cutoff series at float64 would double a ~100 MB artefact for none.
    """
    payload: dict[str, Any] = {f"series|{'|'.join(key)}": v.astype(np.float32) for key, v in archive.series.items()}
    payload.update({f"baseline|{'|'.join(key)}": v for key, v in archive.baselines.items()})
    payload.update({f"times|{name}": grid for name, grid in archive.times.items()})
    payload.update({f"state|{'|'.join(key)}": v.astype(np.float32) for key, v in archive.states.items()})
    payload["baseline_times"] = archive.baseline_times
    payload["state_times"] = archive.state_times
    np.savez_compressed(cache, **payload)


def _load_scores(cache: Path, manifest: dict[str, Any]) -> ScoreArchive:
    """Read a cached score archive back into its keyed form."""
    times: dict[str, FloatArray] = {}
    series: dict[tuple[str, str, str, str, str], FloatArray] = {}
    baselines: dict[tuple[str, str, str], FloatArray] = {}
    states: dict[tuple[str, str], FloatArray] = {}
    baseline_times = np.empty(0, dtype=np.float64)
    state_times = np.empty(0, dtype=np.float64)

    with np.load(cache) as data:
        for key in data.files:
            kind, _, rest = key.partition("|")
            if kind == "baseline_times":
                baseline_times = data[key]
            elif kind == "state_times":
                state_times = data[key]
            elif kind == "times":
                times[rest] = data[key]
            elif kind == "baseline":
                branch, name, cutoff = rest.split("|")
                baselines[branch, name, cutoff] = data[key]
            elif kind == "state":
                branch, arm = rest.split("|")
                states[branch, arm] = data[key]
            else:
                branch, arm, metric, channel_set, cutoff = rest.split("|")
                series[branch, arm, metric, channel_set, cutoff] = data[key]

    return ScoreArchive(
        times=times,
        series=series,
        baselines=baselines,
        baseline_times=baseline_times,
        states=states,
        state_times=state_times,
        n_replicates=manifest["n_children"],
        manifest=manifest,
    )
