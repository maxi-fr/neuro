from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.signal import decimate
from simulate.config import load_config as load_sim_config

from experiments.observable_predictability.archive import (
    EVALUATION_SCORES_FILE,
    SCHEMA_VERSION,
    ObservableEvaluationBundle,
    ScoreCacheMetadata,
    compute_manifest_hashes,
    load_evaluation_bundle,
    save_evaluation_bundle,
)
from experiments.observable_predictability.features import (
    CONNECTIVITY_CANDIDATE_WINDOWS_S,
    CONNECTIVITY_VALID_PAIR_THRESHOLD,
    CONTROL_HORIZONS_S,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FRAME_HOP_S,
    DEFAULT_PSD_FLOOR,
    DEFAULT_WAVEFORM_FS,
    HEALTHY_SPECTRAL_QUANTILE,
    MIN_CONNECTIVITY_SEGMENTS,
    REPRESENTATIONS,
    ConnectivityExtraction,
    ConnectivitySupportSelection,
    HealthyConnectivityReference,
    HealthySpectralReference,
    HealthyWaveformReference,
    baseline_stft_geometry,
    compute_connectivity_penalty,
    compute_seizure_state,
    compute_spectral_penalty,
    compute_waveform_penalty,
    condition_waveform_pair,
    extract_connectivity_frames,
    extract_spectral_frames,
    select_connectivity_support_window,
)
from experiments.observable_predictability.metrics import (
    SEIZURE_BAND_HZ,
    RolloutIdentity,
    validate_identities,
)
from experiments.observable_predictability.scoring import (
    ReplayMoments,
    compute_potential_predictability,
    compute_replay_moments,
    score_zero_current_seizure_relevance,
)
from neuro.config import StftGeometry, resolve_simulation_config
from neuro.plant.connectome import Connectome
from neuro.plant.eeg import build_eeg_leadfield, focal_channels
from neuro.plant.jansen_rit import JansenRitDynamics, lfp, simulate_network
from neuro.plant.seizure import A_HEALTHY, SPREAD_WINDOW_S, build_seizure_a_gains, spread_profile_from_lfp

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from neuro.types import BoolArray, Float32Array, FloatArray, IntArray

PLANT_CONFIG = Path("configs/simulation/uncontrolled.yaml")
DEFAULT_PRE_BRANCH_S = 3.5

PlantKind = Literal["healthy", "seizure"]
SplitName = Literal["reference", "dev", "eval"]
SPLITS: tuple[SplitName, ...] = ("reference", "dev", "eval")

BASE_PARAMS: dict[str, Any] = {
    "A": A_HEALTHY,
    "B": 22.0,
    "a": 100.0,
    "b": 50.0,
    "C1": 135.0,
    "C2": 108.0,
    "C3": 33.75,
    "C4": 33.75,
    "e0": 2.5,
    "v0": 6.0,
    "r": 0.56,
    "mean_input": 90.0,
    "sigma": 280.0,
}

SEIZURE_PARAMS: dict[str, Any] = {
    "dt": 1e-4,
    "initial_state": "rest",
    "log": "lfp",
    "regime": "seizure",
    "connectome": {"speed": 50.0, "K": 0.60},
    "stimulation": {
        "model": "roast_3d",
        "field_projection_path": "data/roast_field_projection_3d.npz",
        "electrodes": ["TP9", "CP5", "Ex8"],
    },
    "params": dict(BASE_PARAMS),
}


def default_plant_params() -> dict[str, Any]:
    """Return a fresh copy of the repo-local seizure Plant configuration."""
    return deepcopy(SEIZURE_PARAMS)


@dataclass(frozen=True)
class Arm:
    """One open-loop Control Current condition held across a Rollout."""

    name: str
    current: tuple[float, ...]


@dataclass(frozen=True)
class Branch:
    """One operating point along a parent trajectory to branch ensembles from."""

    name: str
    plant: PlantKind
    t_branch: float
    arms: tuple[Arm, ...]


ARM_ZERO = Arm("zero", (0.0, 0.0, 0.0))
BRANCHES: tuple[Branch, ...] = (
    Branch("healthy", "healthy", 4.0, arms=(ARM_ZERO,)),
    Branch("pre_onset", "seizure", 1.0, arms=(ARM_ZERO,)),
    Branch("ez_ignited", "seizure", 3.0, arms=(ARM_ZERO,)),
    Branch("mid_spread", "seizure", 7.0, arms=(ARM_ZERO,)),
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
    """Configuration for healthy calibration and seizure evaluation replays."""

    n_parents: int = 16
    n_children: int = 8
    rollout_s: float = 3.0
    pre_branch_s: float = DEFAULT_PRE_BRANCH_S
    branches: tuple[Branch, ...] = BRANCHES
    parent_seed_base: int = 69
    child_seed_base: int = 1_000_000
    reference_seeds: tuple[int, ...] = (101, 102)
    dev_seeds: tuple[int, ...] = (201, 202)
    eval_seeds: tuple[int, ...] | None = None
    region_fs: float = 1000.0
    target_region: str = "lTCI"
    n_focal_channels: int = 4
    chunk_s: float = 1.0
    store_conditioned_eeg: bool = False

    def __post_init__(self) -> None:
        """Validate branch timing, history support, and the seeds of the active splits."""
        if self.eval_seeds is not None:
            object.__setattr__(self, "n_parents", len(self.eval_seeds))
        if self.pre_branch_s < SPREAD_WINDOW_S:
            msg = f"pre_branch_s ({self.pre_branch_s} s) is shorter than the {SPREAD_WINDOW_S} s spread window."
            raise ValueError(msg)
        for branch in self.branches:
            if branch.t_branch < SPREAD_WINDOW_S:
                msg = (
                    f"Branch '{branch.name}' branches at {branch.t_branch} s, shorter than the "
                    f"{SPREAD_WINDOW_S} s of pre-branch history each region rollout carries."
                )
                raise ValueError(msg)

        ref_set = set(self.reference_seeds)
        dev_set = set(self.dev_seeds)
        eval_seizure = set(self.split_parent_seeds("eval", "seizure"))
        if (
            len(ref_set) != len(self.reference_seeds)
            or len(dev_set) != len(self.dev_seeds)
            or len(eval_seizure) != self.n_parents
        ):
            msg = "Duplicate parent seed within reference_seeds, dev_seeds, or eval_seeds."
            raise ValueError(msg)
        eval_all = eval_seizure
        if (ref_set & dev_set) or (ref_set & eval_all) or (dev_set & eval_all):
            msg = "reference_seeds, dev_seeds, and eval_seeds must be pairwise disjoint."
            raise ValueError(msg)

    def branch_history_s(self, branch: Branch) -> float:
        """Return pre-branch history duration in seconds for ``branch``, capped at ``branch.t_branch``."""
        return min(self.pre_branch_s, branch.t_branch)

    def child_seed(self, parent: int, child: int) -> int:
        """Return the replicate noise seed shared across arms of one parent snapshot."""
        return self.child_seed_base + parent * self.n_children + child

    def replicate_seed(self, split: SplitName, parent: int, replicate: int) -> int:
        """Return the replicate noise seed for ``split``, shared across paired arms."""
        if split == "eval":
            return self.child_seed(parent, replicate)
        if split == "dev":
            return self.child_seed_base + 500_000 + parent * self.n_children + replicate
        return self.child_seed_base + 900_000 + parent

    def parent_seed(self, plant: PlantKind, parent: int) -> int:
        """Return the evaluation parent seed for ``plant`` and parent index ``parent``."""
        if self.eval_seeds is not None:
            return int(self.eval_seeds[parent])
        return self.parent_seed_base + parent + (0 if plant == "healthy" else 1000)

    def split_parent_seeds(self, split: SplitName, plant: PlantKind) -> tuple[int, ...]:
        """Return the ordered parent seeds for an active split and Plant."""
        if split == "reference":
            return tuple(int(s) for s in self.reference_seeds) if plant == "healthy" else ()
        if split == "dev":
            return self.dev_seeds if plant == "healthy" else ()
        if plant == "healthy":
            return ()
        return tuple(self.parent_seed(plant, i) for i in range(self.n_parents))

    def split_n_replicates(self, split: SplitName) -> int:
        """Return the replicate count per parent snapshot for ``split``."""
        return 1 if split == "reference" else self.n_children

    def branches_of(self, plant: PlantKind) -> tuple[Branch, ...]:
        """Return the branches taken from a parent of ``plant``, ordered by ``t_branch``."""
        return tuple(sorted((b for b in self.branches if b.plant == plant), key=lambda b: b.t_branch))

    def reference_branches(self) -> tuple[Branch, ...]:
        """Return the healthy zero-current branches generated for the reference split."""
        if not self.reference_seeds:
            return ()
        healthy = self.branches_of("healthy")
        if healthy:
            return tuple(replace(b, arms=(ARM_ZERO,)) for b in healthy)
        t_ref = max(SPREAD_WINDOW_S, min(b.t_branch for b in self.branches))
        return (Branch("healthy", "healthy", t_ref, arms=(ARM_ZERO,)),)

    def branches_for_split(self, split: SplitName, plant: PlantKind) -> tuple[Branch, ...]:
        """Return the branches active for ``split`` and ``plant``."""
        if split == "reference":
            return self.reference_branches() if plant == "healthy" else ()
        if not self.split_parent_seeds(split, plant):
            return ()
        return self.branches_of(plant)

    def expected_identities(self) -> tuple[RolloutIdentity, ...]:
        """Build and validate the complete ordered table of expected Rollout identities."""
        identities: list[RolloutIdentity] = []
        for split in SPLITS:
            n_rep = self.split_n_replicates(split)
            for plant in ("healthy", "seizure"):
                branches = self.branches_for_split(split, plant)
                if not branches:
                    continue
                seeds = self.split_parent_seeds(split, plant)
                for p_idx, p_seed in enumerate(seeds):
                    for branch in branches:
                        for rep in range(n_rep):
                            rep_seed = self.replicate_seed(split, p_idx, rep)
                            identities.extend(
                                RolloutIdentity(
                                    plant=plant,
                                    parent_seed=p_seed,
                                    branch=branch.name,
                                    replicate=rep,
                                    arm=arm.name,
                                    split=split,
                                    replicate_seed=rep_seed,
                                )
                                for arm in branch.arms
                            )
        return validate_identities(identities)

    @property
    def n_replicates(self) -> int:
        """Noise replicates per parent snapshot on development and evaluation splits."""
        return self.n_children

    @property
    def n_rollouts(self) -> int:
        """Evaluation Rollouts stored per branch and arm, indexed ``parent * n_children + child``."""
        return self.n_parents * self.n_children


@dataclass
class PlantPair:
    """Paired healthy and seizure Plants sharing connectome, resting state, and Leadfield."""

    healthy: JansenRitDynamics
    seizure: JansenRitDynamics
    connectome: Connectome
    leadfield: FloatArray
    channel_labels: list[str]

    def of(self, plant: PlantKind) -> JansenRitDynamics:
        """Return the Plant of the requested kind."""
        return self.healthy if plant == "healthy" else self.seizure


def build_plants(config_path: Path | dict[str, Any] | None = None) -> PlantPair:
    """Build paired healthy and seizure Plants from repo-local parameters or a config file."""
    if config_path is None:
        dynamics_cfg = default_plant_params()
    elif isinstance(config_path, dict):
        raw_cfg = resolve_simulation_config(config_path) if "dynamics" in config_path else config_path
        source = raw_cfg.get("dynamics", raw_cfg)
        dynamics_cfg = {k: v for k, v in source.items() if k != "class_path"}
    else:
        raw_cfg = load_sim_config(config_path)
        if isinstance(raw_cfg, dict):
            raw_cfg = resolve_simulation_config(raw_cfg)
        dynamics_cfg = {k: v for k, v in raw_cfg["dynamics"].items() if k != "class_path"}

    conn = Connectome.from_config(dynamics_cfg["connectome"])
    expected = build_seizure_a_gains(conn)
    is_seizure = dynamics_cfg.get("regime") == "seizure"
    has_matching_a = (
        "params" in dynamics_cfg
        and "A" in dynamics_cfg["params"]
        and not isinstance(dynamics_cfg["params"]["A"], str)
        and np.allclose(np.asarray(dynamics_cfg["params"]["A"], dtype=np.float64), expected)
    )
    if not (is_seizure or has_matching_a):
        msg = f"{config_path} does not carry the EZ/PZ A vector of neuro.plant.seizure.build_seizure_a_gains"
        raise ValueError(msg)

    seizure = JansenRitDynamics.from_config(dynamics_cfg)
    healthy = deepcopy(seizure)
    healthy.net_params = replace(seizure.net_params, A=np.full(len(expected), A_HEALTHY))
    healthy.params_tuple = healthy.net_params.to_numba_tuple(len(expected))

    leadfield, channel_labels = build_eeg_leadfield()
    return PlantPair(
        healthy=healthy,
        seizure=seizure,
        connectome=conn,
        leadfield=np.asarray(leadfield[:, : len(conn.region_labels)], dtype=np.float64),
        channel_labels=[str(label) for label in channel_labels],
    )


def reseed(dyn: JansenRitDynamics, seed: int) -> JansenRitDynamics:
    """Copy ``dyn`` with its full delay history and clock intact, replacing only its RNG stream."""
    child = deepcopy(dyn)
    child.rng = np.random.default_rng(seed)
    return child


def _elapsed_steps(dyn: JansenRitDynamics) -> int:
    """Return steps the Plant has already taken on its monotonic clock."""
    return round(dyn.next_update_time / dyn.dt)


def run_segment(dyn: JansenRitDynamics, n_steps: int, current: tuple[float, ...]) -> FloatArray:
    """Step ``dyn`` forward ``n_steps`` holding ``current`` and return regional LFP.

    Returns
    -------
    FloatArray
        Regional LFP ``y = x2 - x3`` in mV, shape ``(n_nodes, n_steps)``.
    """
    t0 = _elapsed_steps(dyn) * dyn.dt
    _, x_traj = simulate_network(
        dyn=dyn,
        duration=n_steps * dyn.dt,
        control_current=np.asarray(current, dtype=np.float64),
        stim_window=(t0, t0 + n_steps * dyn.dt),
        t0=t0,
    )
    return lfp(x_traj)[:, 1:]


@dataclass
class Parent:
    """State of one parent trajectory at its branch points along with pre-branch histories."""

    index: int
    plant: PlantKind
    seed: int
    snapshots: dict[str, JansenRitDynamics]
    pre_regions: dict[str, FloatArray]
    n_seizing_final: int
    onsets: FloatArray
    history_eeg: dict[str, FloatArray] = field(default_factory=dict)
    history_regions: dict[str, FloatArray] = field(default_factory=dict)
    split: SplitName = "eval"


def _to_region_fs(y: FloatArray, dt: float, region_fs: float) -> Float32Array:
    """Decimate a full-rate regional LFP array to ``region_fs`` in float32."""
    factor = round(1.0 / (dt * region_fs))
    return np.asarray(decimate(y, factor, axis=1) if factor > 1 else y, dtype=np.float32)


def _trailing(chunks: list[FloatArray], n: int) -> FloatArray:
    """Join the trailing ``n`` columns from ``chunks`` without concatenating earlier chunks."""
    tail: list[FloatArray] = []
    need = n
    for chunk in reversed(chunks):
        taken = chunk if need >= chunk.shape[1] else chunk[:, chunk.shape[1] - need :]
        tail.append(taken)
        need -= taken.shape[1]
        if need == 0:
            break
    return np.concatenate(tail[::-1], axis=1)


def run_parent(
    plants: PlantPair,
    cfg: EnsembleConfig,
    plant: PlantKind,
    index: int,
    *,
    split: SplitName = "eval",
) -> Parent:
    """Run one unstimulated parent to its last branch point, capturing snapshots and histories."""
    actual_seed = cfg.split_parent_seeds(split, plant)[index]
    active_branches = cfg.branches_for_split(split, plant)
    dyn = reseed(plants.of(plant), actual_seed)
    zero = (0.0,) * dyn.n_controls

    n_chunk = round(cfg.chunk_s / dyn.dt)
    snapshots: dict[str, JansenRitDynamics] = {}
    pre_regions: dict[str, FloatArray] = {}
    history_eeg: dict[str, FloatArray] = {}
    history_regions: dict[str, FloatArray] = {}
    y_chunks: list[FloatArray] = []
    n_pre = round(SPREAD_WINDOW_S / dyn.dt)

    for branch in active_branches:
        target = round(branch.t_branch / dyn.dt)
        while _elapsed_steps(dyn) < target:
            y_chunks.append(run_segment(dyn, min(n_chunk, target - _elapsed_steps(dyn)), zero))
        snapshots[branch.name] = deepcopy(dyn)
        pre_regions[branch.name] = _trailing(y_chunks, n_pre)
        n_hist = round(cfg.branch_history_s(branch) / dyn.dt)
        hist_lfp = _trailing(y_chunks, n_hist)
        history_regions[branch.name] = hist_lfp
        history_eeg[branch.name] = plants.leadfield @ hist_lfp

    profile = spread_profile_from_lfp(np.concatenate(y_chunks, axis=1), dyn.dt)
    return Parent(
        index=index,
        plant=plant,
        seed=actual_seed,
        snapshots=snapshots,
        pre_regions=pre_regions,
        n_seizing_final=int(profile.n_seizing()[-1]),
        onsets=profile.onsets,
        history_eeg=history_eeg,
        history_regions=history_regions,
        split=split,
    )


def run_child(  # noqa: PLR0913 -- child rollout requires snapshot, plants, config, seed, arm, and history
    snapshot: JansenRitDynamics,
    plants: PlantPair,
    cfg: EnsembleConfig,
    seed: int,
    arm: Arm,
    *,
    pre_region: FloatArray,
) -> tuple[FloatArray, Float32Array]:
    """Branch one noise realization off ``snapshot`` and roll it out under ``arm``.

    Parameters
    ----------
    snapshot
        Parent Plant frozen at its branch point; copied, not mutated.
    plants
        Paired Plants carrying the EEG Leadfield.
    cfg
        Ensemble configuration specifying Rollout duration and regional sampling rate.
    seed
        Replicate noise seed shared across matched arms.
    arm
        Open-loop Control Current condition held across the Rollout.
    pre_region
        Full-rate regional LFP over the pre-branch spread window, shape ``(n_nodes, n_pre)``.

    Returns
    -------
    eeg
        Raw EEG at the Plant rate, shape ``(n_channels, n_samples)``.
    region
        ``pre_region`` joined to the Rollout LFP and decimated in one pass,
        shape ``(n_nodes, n_samples_region)``.
    """
    dyn = reseed(snapshot, seed)
    y = run_segment(dyn, round(cfg.rollout_s / dyn.dt), arm.current)

    region = _to_region_fs(np.concatenate([pre_region, y], axis=1), dyn.dt, cfg.region_fs)
    return plants.leadfield @ y, region


def _open_store(path: Path, shape: tuple[int, ...], dtype: type[np.floating]) -> np.memmap:
    """Open a ``.npy`` file as a writable memory-mapped array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _split_prefix(split: str) -> str:
    """Return the filename prefix for ``split`` (empty for ``'eval'`` to preserve paths)."""
    return "" if split == "eval" else f"{split}_"


def eeg_path(out_dir: Path, branch: str, arm: str, split: str = "eval") -> Path:
    """Return the path of the full-rate Raw EEG store for ``branch``, ``arm``, and ``split``."""
    return out_dir / "eeg" / f"{_split_prefix(split)}{branch}_{arm}.npy"


def region_path(out_dir: Path, branch: str, arm: str, split: str = "eval") -> Path:
    """Return the path of the decimated regional LFP store for ``branch``, ``arm``, and ``split``."""
    return out_dir / "region" / f"{_split_prefix(split)}{branch}_{arm}.npy"


def eeg_history_path(out_dir: Path, branch: str, split: str = "eval") -> Path:
    """Return the path of the per-parent pre-branch Raw EEG history store for ``branch`` and ``split``."""
    return out_dir / "history_eeg" / f"{split}_{branch}.npy"


def region_history_path(out_dir: Path, branch: str, split: str = "eval") -> Path:
    """Return the path of the per-parent pre-branch regional LFP history store for ``branch`` and ``split``."""
    return out_dir / "history_region" / f"{split}_{branch}.npy"


@dataclass
class _Stores:
    """Open output memory maps keyed by ``(split, branch, arm)`` and ``(split, branch)``."""

    eeg: dict[tuple[str, str, str], np.memmap] = field(default_factory=dict)
    region: dict[tuple[str, str, str], np.memmap] = field(default_factory=dict)
    eeg_history: dict[tuple[str, str], np.memmap] = field(default_factory=dict)
    region_history: dict[tuple[str, str], np.memmap] = field(default_factory=dict)

    def flush(self) -> None:
        """Flush all memory-mapped stores to disk and release their handles."""
        all_stores = (
            *self.eeg.values(),
            *self.region.values(),
            *self.eeg_history.values(),
            *self.region_history.values(),
        )
        for store in all_stores:
            store.flush()
        self.eeg.clear()
        self.region.clear()
        self.eeg_history.clear()
        self.region_history.clear()


def _build_stores(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> _Stores:
    """Pre-allocate continuation and pre-branch history stores across all configured splits."""
    dt = plants.seizure.dt
    eeg_factor = round(1 / (dt * DEFAULT_WAVEFORM_FS)) if cfg.store_conditioned_eeg else 1
    n_eeg = round(cfg.rollout_s / dt) // eeg_factor
    n_region = round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs)
    n_channels = plants.leadfield.shape[0]
    n_nodes = plants.leadfield.shape[1]

    stores = _Stores()
    for split in SPLITS:
        n_rep = cfg.split_n_replicates(split)
        for plant in ("healthy", "seizure"):
            branches = cfg.branches_for_split(split, plant)
            seeds = cfg.split_parent_seeds(split, plant)
            if not branches or not seeds:
                continue
            n_parents_split = len(seeds)
            n_rollouts_split = n_parents_split * n_rep
            for branch in branches:
                hist_s = cfg.branch_history_s(branch)
                n_hist_eeg = (round(hist_s / dt) + eeg_factor - 1) // eeg_factor
                n_hist_region = round(hist_s * cfg.region_fs)
                h_key = (split, branch.name)
                stores.eeg_history[h_key] = _open_store(
                    eeg_history_path(out_dir, branch.name, split),
                    (n_parents_split, n_channels, n_hist_eeg),
                    np.float64,
                )
                stores.region_history[h_key] = _open_store(
                    region_history_path(out_dir, branch.name, split),
                    (n_parents_split, n_nodes, n_hist_region),
                    np.float32,
                )
                for arm in branch.arms:
                    key = (split, branch.name, arm.name)
                    stores.eeg[key] = _open_store(
                        eeg_path(out_dir, branch.name, arm.name, split),
                        (n_rollouts_split, n_channels, n_eeg),
                        np.float64,
                    )
                    stores.region[key] = _open_store(
                        region_path(out_dir, branch.name, arm.name, split),
                        (n_rollouts_split, n_nodes, n_region),
                        np.float32,
                    )
    return stores


def generate(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> dict[str, Any]:
    """Generate the complete ensemble into ``out_dir``, atomically publish ``manifest.json``, and return it."""
    manifest: dict[str, Any] = {}
    for update in generate_iter(out_dir, cfg, plants):
        manifest = update
    return manifest


def _write_parent_branch(
    stores: _Stores, parent: Parent, branch: Branch, cfg: EnsembleConfig, plants: PlantPair
) -> None:
    """Write the branch history and all replay rows for one Parent snapshot."""
    split = parent.split
    index = parent.index
    dt = plants.seizure.dt
    n_rep = cfg.split_n_replicates(split)
    history_key = (split, branch.name)
    eeg_history = parent.history_eeg[branch.name]
    if not cfg.store_conditioned_eeg:
        stores.eeg_history[history_key][index] = eeg_history
    stores.region_history[history_key][index] = _to_region_fs(parent.history_regions[branch.name], dt, cfg.region_fs)
    for child in range(n_rep):
        rep_seed = cfg.replicate_seed(split, index, child)
        row = index * n_rep + child
        for arm in branch.arms:
            eeg, region = run_child(
                parent.snapshots[branch.name], plants, cfg, rep_seed, arm, pre_region=parent.pre_regions[branch.name]
            )
            if cfg.store_conditioned_eeg:
                history_50, eeg_50 = condition_waveform_pair(eeg.T, 1 / dt, history_raw=eeg_history.T)
                if child == 0 and arm == branch.arms[0]:
                    assert history_50 is not None  # noqa: S101 -- pre-branch history was supplied
                    stores.eeg_history[history_key][index] = history_50.T
                eeg = eeg_50.T
            stores.eeg[split, branch.name, arm.name][row] = eeg
            stores.region[split, branch.name, arm.name][row] = region


def generate_iter(out_dir: Path, cfg: EnsembleConfig, plants: PlantPair) -> Iterator[dict[str, Any]]:
    """Generate in a fresh output directory and publish its manifest after all stores flush."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Reserve the directory before opening any stores. A retry uses a fresh output directory.
    reservation = out_dir / ".generation"
    reservation.touch(exist_ok=False)
    if (out_dir / "manifest.json").exists():
        reservation.unlink()
        msg = f"Ensemble already exists in {out_dir}. Choose a fresh output directory."
        raise FileExistsError(msg)
    stores = _build_stores(out_dir, cfg, plants)
    eval_parents: list[Parent] = []
    split_parents: dict[str, list[Parent]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        for plant in ("healthy", "seizure"):
            branches = cfg.branches_for_split(split, plant)
            seeds = cfg.split_parent_seeds(split, plant)
            if not branches or not seeds:
                continue
            for index in range(len(seeds)):
                parent = run_parent(plants, cfg, plant, index, split=split)
                for branch in branches:
                    _write_parent_branch(stores, parent, branch, cfg, plants)
                parent.snapshots.clear()
                parent.pre_regions.clear()
                parent.history_eeg.clear()
                parent.history_regions.clear()
                split_parents[split].append(parent)
                if split == "eval":
                    eval_parents.append(parent)
                    yield _assemble_manifest(cfg, plants, eval_parents, split_parents)

    stores.flush()
    final_manifest = _write_manifest_atomic(out_dir, cfg, plants, eval_parents, split_parents)
    reservation.unlink()
    yield final_manifest


def _serialize_parent(p: Parent) -> dict[str, Any]:
    """Serialize one parent summary record for the manifest."""
    return {
        "index": p.index,
        "plant": p.plant,
        "seed": p.seed,
        "split": p.split,
        "n_seizing_final": p.n_seizing_final,
        "onsets": [None if not np.isfinite(t) else float(t) for t in p.onsets],
    }


def _assemble_manifest(
    cfg: EnsembleConfig,
    plants: PlantPair,
    eval_parents: list[Parent],
    split_parents: Mapping[str, list[Parent]],
) -> dict[str, Any]:
    """Assemble manifest metadata for the configured splits and evaluation branches."""
    target_idx = plants.connectome.region_index[cfg.target_region]
    focal = focal_channels(plants.leadfield, target_idx, cfg.n_focal_channels)
    identities = cfg.expected_identities()

    splits_meta: dict[str, Any] = {}
    for split in SPLITS:
        n_rep = cfg.split_n_replicates(split)
        split_branches = [
            {
                "name": b.name,
                "plant": b.plant,
                "t_branch": b.t_branch,
                "history_s": cfg.branch_history_s(b),
                "arms": [a.name for a in b.arms],
            }
            for plant in ("healthy", "seizure")
            for b in cfg.branches_for_split(split, plant)
        ]
        splits_meta[split] = {
            "n_replicates": n_rep,
            "seeds": {
                "healthy": list(cfg.split_parent_seeds(split, "healthy")),
                "seizure": list(cfg.split_parent_seeds(split, "seizure")),
            },
            "branches": split_branches,
            "parents": [_serialize_parent(p) for p in split_parents.get(split, [])],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dt": plants.seizure.dt,
        "fs": DEFAULT_WAVEFORM_FS if cfg.store_conditioned_eeg else 1.0 / plants.seizure.dt,
        **({"eeg_conditioned_cutoff_hz": DEFAULT_CUTOFF_HZ} if cfg.store_conditioned_eeg else {}),
        "region_fs": cfg.region_fs,
        "rollout_s": cfg.rollout_s,
        "pre_branch_s": SPREAD_WINDOW_S,
        "history_s": cfg.pre_branch_s,
        "n_parents": cfg.n_parents,
        "n_children": cfg.n_children,
        "rollout_index": "parent * n_children + child",
        "branches": [
            {
                "name": b.name,
                "plant": b.plant,
                "t_branch": b.t_branch,
                "history_s": cfg.branch_history_s(b),
                "arms": [a.name for a in b.arms],
            }
            for b in cfg.branches
            if cfg.split_parent_seeds("eval", b.plant)
        ],
        "arms": [{"name": a.name, "current": list(a.current)} for a in all_arms(cfg.branches)],
        "electrode_labels": [str(label) for label in plants.seizure.stim.control_labels],
        "channel_labels": plants.channel_labels,
        "region_labels": [str(label) for label in plants.connectome.region_labels],
        "channel_sets": {
            "all62": list(range(plants.leadfield.shape[0])),
            cfg.target_region: [int(i) for i in focal],
        },
        "parents": [_serialize_parent(p) for p in eval_parents],
        "splits": splits_meta,
        "identities": [ident.to_dict() for ident in identities],
    }


def _write_manifest_atomic(
    out_dir: Path,
    cfg: EnsembleConfig,
    plants: PlantPair,
    eval_parents: list[Parent],
    split_parents: Mapping[str, list[Parent]],
) -> dict[str, Any]:
    """Write ``manifest.json`` atomically via a temporary file after validating all stores."""
    manifest = _assemble_manifest(cfg, plants, eval_parents, split_parents)
    validate_archive(out_dir, manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "manifest.json.tmp"
    final_path = out_dir / "manifest.json"
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(final_path)
    return manifest


def _split_expected_identities(split: str, split_info: Mapping[str, Any]) -> list[RolloutIdentity]:
    """Build expected RolloutIdentity entries for a single split block."""
    n_rep = int(split_info["n_replicates"])
    seeds_by_plant = split_info["seeds"]
    branches_by_plant: dict[str, list[Mapping[str, Any]]] = {"healthy": [], "seizure": []}
    for b_info in split_info["branches"]:
        branches_by_plant[str(b_info["plant"])].append(b_info)

    entries: list[RolloutIdentity] = []
    for plant in ("healthy", "seizure"):
        p_branches = branches_by_plant[plant]
        p_seeds = [int(s) for s in seeds_by_plant.get(plant, [])]
        entries.extend(
            RolloutIdentity(
                plant=plant,
                parent_seed=p_seed,
                branch=str(b_info["name"]),
                replicate=rep,
                arm=str(arm_name),
                split=split,
            )
            for p_seed in p_seeds
            for b_info in p_branches
            for rep in range(n_rep)
            for arm_name in b_info["arms"]
        )
    return entries


def _expected_manifest_identities(manifest: Mapping[str, Any]) -> tuple[RolloutIdentity, ...]:
    """Reconstruct the expected identity sequence from the manifest's split and branch metadata."""
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        msg = "Manifest is missing a valid 'splits' mapping."
        raise ValueError(msg)

    expected: list[RolloutIdentity] = []
    for split in SPLITS:
        if split in splits:
            expected.extend(_split_expected_identities(split, splits[split]))
    if not expected:
        msg = "Manifest defines zero expected Rollout identities."
        raise ValueError(msg)
    return validate_identities(expected)


def _check_store_shape(path: Path, want_shape: tuple[int, ...], kind: str) -> None:
    """Verify ``path`` exists on disk and its ``.npy`` array matches ``want_shape``."""
    if not path.exists():
        msg = f"Missing {kind} store: {path}"
        raise FileNotFoundError(msg)
    arr = np.load(path, mmap_mode="r")
    if arr.shape != want_shape:
        msg = f"{kind} store {path} has shape {arr.shape}, expected {want_shape}."
        raise ValueError(msg)


def _validate_split_stores(
    out_dir: Path,
    split: str,
    split_info: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Validate parent counts and on-disk stores for a single split."""
    dt = float(manifest["dt"])
    region_fs = float(manifest["region_fs"])
    rollout_s = float(manifest["rollout_s"])
    n_channels = len(manifest["channel_labels"])
    n_regions = len(manifest["region_labels"])
    eeg_factor = round(1 / (dt * float(manifest["fs"])))
    n_eeg = round(rollout_s / dt) // eeg_factor
    n_region = round((SPREAD_WINDOW_S + rollout_s) * region_fs)

    n_rep = int(split_info["n_replicates"])
    seeds_by_plant = split_info["seeds"]
    expected_parents = sum(
        len(seeds_by_plant.get(plant, []))
        for plant in ("healthy", "seizure")
        if any(b["plant"] == plant for b in split_info["branches"])
    )
    if len(split_info.get("parents", [])) != expected_parents:
        msg = (
            f"Partial manifest for split {split!r}: recorded {len(split_info.get('parents', []))} "
            f"parents, expected {expected_parents}."
        )
        raise ValueError(msg)

    default_hist_s = float(manifest.get("history_s", SPREAD_WINDOW_S))
    for b_info in split_info["branches"]:
        b_name = str(b_info["name"])
        plant = str(b_info["plant"])
        n_parents_split = len(seeds_by_plant.get(plant, []))
        n_rollouts_split = n_parents_split * n_rep
        hist_s = float(b_info.get("history_s", min(default_hist_s, float(b_info["t_branch"]))))

        _check_store_shape(
            eeg_history_path(out_dir, b_name, split),
            (n_parents_split, n_channels, (round(hist_s / dt) + eeg_factor - 1) // eeg_factor),
            "pre-branch history",
        )
        _check_store_shape(
            region_history_path(out_dir, b_name, split),
            (n_parents_split, n_regions, round(hist_s * region_fs)),
            "pre-branch history",
        )
        for arm_name in b_info["arms"]:
            arm_str = str(arm_name)
            _check_store_shape(
                eeg_path(out_dir, b_name, arm_str, split),
                (n_rollouts_split, n_channels, n_eeg),
                "Rollout",
            )
            _check_store_shape(
                region_path(out_dir, b_name, arm_str, split),
                (n_rollouts_split, n_regions, n_region),
                "Rollout",
            )


def validate_archive(out_dir: Path, manifest: Mapping[str, Any]) -> tuple[RolloutIdentity, ...]:
    """Validate manifest schema, identity completeness, arm pairing, and on-disk ``.npy`` stores."""
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = f"Unsupported or legacy manifest schema_version: {version!r} (expected {SCHEMA_VERSION})."
        raise ValueError(msg)

    raw_identities = manifest.get("identities")
    if not isinstance(raw_identities, list) or not raw_identities:
        msg = "Manifest is missing a non-empty 'identities' list."
        raise ValueError(msg)

    identities = validate_identities([RolloutIdentity.from_dict(item) for item in raw_identities])
    expected = _expected_manifest_identities(manifest)
    if len(identities) != len(expected):
        msg = f"Manifest identity count mismatch: found {len(identities)}, expected {len(expected)}."
        raise ValueError(msg)

    for idx, (actual, want) in enumerate(zip(identities, expected, strict=True)):
        if actual.full_key != want.full_key or actual.split != want.split:
            msg = f"Manifest identity mismatch at index {idx}: found {actual}, expected {want}."
            raise ValueError(msg)

    pair_seeds: dict[tuple[str, str, int, str, int], int] = {}
    for ident in identities:
        p_key = (ident.split, *ident.pair_key)
        if p_key in pair_seeds and pair_seeds[p_key] != ident.replicate_seed:
            msg = (
                f"Mismatched replicate_seed across paired arms for {p_key}: "
                f"{pair_seeds[p_key]} vs {ident.replicate_seed}."
            )
            raise ValueError(msg)
        pair_seeds[p_key] = ident.replicate_seed

    splits = manifest["splits"]
    for split in SPLITS:
        if split in splits:
            _validate_split_stores(out_dir, split, splits[split], manifest)

    return identities


def load_manifest(out_dir: Path) -> dict[str, Any]:
    """Read and strictly validate the manifest and stores of a generated ensemble."""
    manifest_file = out_dir / "manifest.json"
    if not manifest_file.exists():
        msg = f"Manifest not found: {manifest_file}"
        raise FileNotFoundError(msg)
    manifest: dict[str, Any] = json.loads(manifest_file.read_text(encoding="utf-8"))
    validate_archive(out_dir, manifest)
    return manifest


def manifest_identities(
    manifest: Mapping[str, Any],
    *,
    split: str | None = None,
    plant: str | None = None,
    branch: str | None = None,
    arm: str | None = None,
) -> tuple[RolloutIdentity, ...]:
    """Return validated Rollout identities from ``manifest``, optionally filtered by metadata fields."""
    all_identities = validate_identities([RolloutIdentity.from_dict(item) for item in manifest["identities"]])
    return tuple(
        ident
        for ident in all_identities
        if (split is None or ident.split == split)
        and (plant is None or ident.plant == plant)
        and (branch is None or ident.branch == branch)
        and (arm is None or ident.arm == arm)
    )


def resolve_identity(
    manifest: Mapping[str, Any],
    key: RolloutIdentity | tuple[str, int, str, int, str],
) -> tuple[RolloutIdentity, int, int]:
    """Resolve ``key`` against ``manifest`` and return ``(identity, parent_idx, row_idx)``."""
    target_key = key.full_key if isinstance(key, RolloutIdentity) else key
    identities = validate_identities([RolloutIdentity.from_dict(item) for item in manifest["identities"]])
    matched = next((ident for ident in identities if ident.full_key == target_key), None)
    if matched is None:
        msg = f"RolloutIdentity {target_key} not found in manifest."
        raise KeyError(msg)
    if isinstance(key, RolloutIdentity) and (
        key.split != matched.split or (key.replicate_seed not in (0, matched.replicate_seed))
    ):
        msg = f"RolloutIdentity metadata mismatch for {target_key}: got {key}, manifest has {matched}."
        raise ValueError(msg)

    split_info = manifest["splits"][matched.split]
    n_rep = int(split_info["n_replicates"])
    plant_seeds = [int(s) for s in split_info["seeds"][matched.plant]]
    if matched.parent_seed not in plant_seeds:
        msg = f"Parent seed {matched.parent_seed} not found in split {matched.split!r} for plant {matched.plant!r}."
        raise KeyError(msg)
    parent_idx = plant_seeds.index(matched.parent_seed)
    row_idx = parent_idx * n_rep + matched.replicate
    return matched, parent_idx, row_idx


def load_scalp_rollout(
    out_dir: Path,
    identity: RolloutIdentity | tuple[str, int, str, int, str],
    manifest: Mapping[str, Any] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Load ``(scalp_history, scalp_continuation)`` in mV at the stored EEG rate for ``identity``.

    Returns
    -------
    scalp_history
        Pre-branch Raw EEG history, shape ``(n_channels, n_hist_samples)``.
    scalp_continuation
        Post-branch Raw EEG continuation, shape ``(n_channels, n_rollout_samples)``.
    """
    active_manifest = load_manifest(out_dir) if manifest is None else manifest
    resolved, parent_idx, row_idx = resolve_identity(active_manifest, identity)
    hist_store = np.load(eeg_history_path(out_dir, resolved.branch, resolved.split), mmap_mode="r")
    cont_store = np.load(eeg_path(out_dir, resolved.branch, resolved.arm, resolved.split), mmap_mode="r")
    return (
        np.asarray(hist_store[parent_idx], dtype=np.float64).copy(),
        np.asarray(cont_store[row_idx], dtype=np.float64).copy(),
    )


def load_region_rollout(
    out_dir: Path,
    identity: RolloutIdentity | tuple[str, int, str, int, str],
    manifest: Mapping[str, Any] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Load ``(lfp_history, lfp_continuation)`` in mV at ``region_fs`` for ``identity``.

    Returns
    -------
    lfp_history
        Pre-branch regional LFP history, shape ``(n_regions, n_hist_samples)``.
    lfp_continuation
        Post-branch regional LFP continuation, shape ``(n_regions, n_rollout_samples)``.
    """
    active_manifest = load_manifest(out_dir) if manifest is None else manifest
    resolved, parent_idx, row_idx = resolve_identity(active_manifest, identity)
    hist_store = np.load(region_history_path(out_dir, resolved.branch, resolved.split), mmap_mode="r")
    cont_store = np.load(region_path(out_dir, resolved.branch, resolved.arm, resolved.split), mmap_mode="r")
    n_pre_spread = round(SPREAD_WINDOW_S * float(active_manifest["region_fs"]))
    return (
        np.asarray(hist_store[parent_idx], dtype=np.float64).copy(),
        np.asarray(cont_store[row_idx, :, n_pre_spread:], dtype=np.float64).copy(),
    )


def _extract_split_branch_arm_waveforms(  # noqa: PLR0913 -- archive directory, manifest, split/branch/arm, and filter config
    out_dir: Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    branch: str,
    arm: str,
    channels: IntArray,
    target_fs: float,
    cutoff_hz: float,
) -> tuple[tuple[RolloutIdentity, ...], FloatArray, FloatArray, FloatArray]:
    """Load and causally condition ``(identities, wave_times, hist_50, cont_50)`` for one ``(split, branch, arm)``."""
    b_ids = manifest_identities(manifest, split=split, branch=branch, arm=arm)
    if not b_ids:
        msg = f"No Rollout identities found for split={split!r}, branch={branch!r}, arm={arm!r}."
        raise ValueError(msg)

    fs_raw = float(manifest["fs"])
    stored_cutoff = manifest.get("eeg_conditioned_cutoff_hz")
    if stored_cutoff is not None and (target_fs != fs_raw or cutoff_hz != stored_cutoff):
        msg = f"Stored EEG was conditioned at {fs_raw:g} Hz with a {stored_cutoff:g} Hz cutoff."
        raise ValueError(msg)
    hist_store = np.load(eeg_history_path(out_dir, branch, split), mmap_mode="r")
    cont_store = np.load(eeg_path(out_dir, branch, arm, split), mmap_mode="r")
    split_info = manifest["splits"][split]
    n_rep = int(split_info["n_replicates"])
    plant_seeds = [int(s) for s in split_info["seeds"][b_ids[0].plant]]
    seed_to_parent = {seed: idx for idx, seed in enumerate(plant_seeds)}

    hist_list: list[FloatArray] = []
    cont_list: list[FloatArray] = []
    parent_hist_cache: dict[int, FloatArray] = {}
    for ident in b_ids:
        p_idx = seed_to_parent[int(ident.parent_seed)]
        row_idx = p_idx * n_rep + int(ident.replicate)
        if p_idx not in parent_hist_cache:
            parent_hist_cache[p_idx] = np.asarray(hist_store[p_idx][channels].T, dtype=np.float64)
        cont_raw = np.asarray(cont_store[row_idx][channels].T, dtype=np.float64)
        history, continuation = condition_waveform_pair(
            cont_raw,
            fs_raw,
            history_raw=parent_hist_cache[p_idx],
            cutoff_hz=cutoff_hz,
            target_fs=target_fs,
        )
        assert history is not None  # noqa: S101 -- the pair includes the supplied pre-branch history
        hist_list.append(history)
        cont_list.append(continuation)

    wave_times = np.arange(1, len(cont_list[0]) + 1, dtype=np.float64) / target_fs
    return b_ids, wave_times, np.stack(hist_list, axis=0), np.stack(cont_list, axis=0)


def _extract_split_branch_arm_seizure_state(  # noqa: PLR0913 -- archive directory, manifest, split/branch/arm, identities, and hop
    out_dir: Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    branch: str,
    arm: str,
    identities: Sequence[RolloutIdentity],
    hop_s: float,
) -> tuple[FloatArray, FloatArray]:
    """Load and compute causal regional LFP Seizure State ``(times, states)`` for ``identities``."""
    region_fs = float(manifest["region_fs"])
    hist_store = np.load(region_history_path(out_dir, branch, split), mmap_mode="r")
    cont_store = np.load(region_path(out_dir, branch, arm, split), mmap_mode="r")
    n_pre_spread = round(SPREAD_WINDOW_S * region_fs)
    split_info = manifest["splits"][split]
    n_rep = int(split_info["n_replicates"])
    plant_seeds = [int(s) for s in split_info["seeds"][identities[0].plant]]
    seed_to_parent = {seed: idx for idx, seed in enumerate(plant_seeds)}

    state_rows: list[FloatArray] = []
    state_times = np.empty(0, dtype=np.float64)
    for ident in identities:
        p_idx = seed_to_parent[int(ident.parent_seed)]
        row_idx = p_idx * n_rep + int(ident.replicate)
        lfp_hist = np.asarray(hist_store[p_idx], dtype=np.float64)
        lfp_cont = np.asarray(cont_store[row_idx, :, n_pre_spread:], dtype=np.float64)
        state_times, s_vals = compute_seizure_state(
            lfp_cont,
            region_fs,
            lfp_history=lfp_hist,
            hop_s=hop_s,
            window_s=SPREAD_WINDOW_S,
        )
        state_rows.append(s_vals)
    return state_times, np.stack(state_rows, axis=0)


def _extract_healthy_split_waveforms(  # noqa: PLR0913 -- archive dir, manifest, split, channels, and conditioning settings
    base_dir: Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    channels: IntArray,
    target_fs: float,
    cutoff_hz: float,
) -> tuple[tuple[RolloutIdentity, ...], list[FloatArray], FloatArray]:
    """Load conditioned zero-current healthy Waveforms and their branch histories."""
    branches = [str(b["name"]) for b in manifest["splits"][split]["branches"] if b["plant"] == "healthy"]
    ids_list: list[RolloutIdentity] = []
    hist_list: list[FloatArray] = []
    wave_list: list[FloatArray] = []
    for branch in branches:
        b_ids, _, b_hist, b_wave = _extract_split_branch_arm_waveforms(
            base_dir,
            manifest,
            split=split,
            branch=branch,
            arm="zero",
            channels=channels,
            target_fs=target_fs,
            cutoff_hz=cutoff_hz,
        )
        ids_list.extend(b_ids)
        hist_list.extend(b_hist)
        wave_list.append(b_wave)
    return tuple(ids_list), hist_list, np.concatenate(wave_list, axis=0)


def _fit_references_and_freeze_dev_parameters(
    base_dir: Path,
    manifest: Mapping[str, Any],
    *,
    channels: IntArray,
    protocol: EvaluationProtocol,
) -> tuple[
    HealthyWaveformReference,
    HealthySpectralReference,
    HealthyConnectivityReference,
    ConnectivitySupportSelection,
]:
    """Fit healthy references from matched Waveforms and histories and select connectivity support on dev."""
    ref_ids, ref_histories, ref_waveforms = _extract_healthy_split_waveforms(
        base_dir,
        manifest,
        split="reference",
        channels=channels,
        target_fs=protocol.target_fs,
        cutoff_hz=protocol.cutoff_hz,
    )
    waveform_reference = HealthyWaveformReference.fit(
        ref_waveforms, fs=protocol.target_fs, identities=ref_ids, split="reference"
    )
    ref_spec_frames = np.stack(
        [
            extract_spectral_frames(
                np.asarray(waveform),
                protocol.target_fs,
                history=history,
                geometry=protocol.stft_geometry,
                target_fs=protocol.target_fs,
            )[1]
            for waveform, history in zip(ref_waveforms, ref_histories, strict=True)
        ]
    )
    spectral_reference = HealthySpectralReference.fit(
        ref_spec_frames, fs=protocol.target_fs, geometry=protocol.stft_geometry, identities=ref_ids, split="reference"
    )

    dev_ids, dev_histories, dev_waveforms = _extract_healthy_split_waveforms(
        base_dir,
        manifest,
        split="dev",
        channels=channels,
        target_fs=protocol.target_fs,
        cutoff_hz=protocol.cutoff_hz,
    )

    connectivity_support = select_connectivity_support_window(
        dev_waveforms,
        protocol.target_fs,
        dev_histories=np.stack(dev_histories),
        identities=dev_ids,
        split="dev",
        candidate_windows_s=protocol.connectivity_candidate_windows_s,
        repeatability_target=protocol.connectivity_repeatability_target,
        horizons_s=protocol.horizons_s,
        hop_s=protocol.hop_s,
        n_segment=protocol.connectivity_n_segment,
        n_welch_hop=protocol.connectivity_n_welch_hop,
        band_hz=protocol.connectivity_band_hz,
    )
    w_conn = connectivity_support.selected_window_s

    ref_conn_exts = [
        extract_connectivity_frames(
            ref_waveforms[r_idx],
            protocol.target_fs,
            history=ref_histories[r_idx],
            window_s=w_conn,
            hop_s=protocol.hop_s,
            n_segment=protocol.connectivity_n_segment,
            n_welch_hop=protocol.connectivity_n_welch_hop,
            band_hz=protocol.connectivity_band_hz,
            target_fs=protocol.target_fs,
        )
        for r_idx in range(ref_waveforms.shape[0])
    ]
    connectivity_reference = HealthyConnectivityReference.fit(
        ref_conn_exts, window_s=w_conn, identities=ref_ids, split="reference"
    )

    dev_conn_values = np.stack(
        [
            extract_connectivity_frames(
                dev_waveforms[r_idx],
                protocol.target_fs,
                history=dev_histories[r_idx],
                window_s=w_conn,
                hop_s=protocol.hop_s,
                n_segment=protocol.connectivity_n_segment,
                n_welch_hop=protocol.connectivity_n_welch_hop,
                band_hz=protocol.connectivity_band_hz,
                target_fs=protocol.target_fs,
            ).values
            for r_idx in range(dev_waveforms.shape[0])
        ],
        axis=0,
    )
    eligible = (np.isfinite(dev_conn_values).mean(axis=(0, 1)) >= CONNECTIVITY_VALID_PAIR_THRESHOLD) & np.isfinite(
        connectivity_reference.q05
    )
    connectivity_reference = replace(connectivity_reference, eligible=eligible)
    return (
        waveform_reference,
        spectral_reference,
        connectivity_reference,
        connectivity_support,
    )


def _extract_eval_branch_arm_observables(  # noqa: PLR0913 -- archive dir, manifest, branch/arm, channels, and extraction config
    base_dir: Path,
    manifest: Mapping[str, Any],
    *,
    branch: str,
    arm_str: str,
    channels: IntArray,
    protocol: EvaluationProtocol,
    w_conn: float,
) -> tuple[
    tuple[RolloutIdentity, ...],
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
]:
    """Extract Waveform, Spectral, and Connectivity trajectories for one ``split='eval'`` branch and arm."""
    b_ids, wave_times, b_hist, b_wave = _extract_split_branch_arm_waveforms(
        base_dir,
        manifest,
        split="eval",
        branch=branch,
        arm=arm_str,
        channels=channels,
        target_fs=protocol.target_fs,
        cutoff_hz=protocol.cutoff_hz,
    )
    spec_rows: list[FloatArray] = []
    conn_vals_rows: list[FloatArray] = []
    conn_mask_rows: list[BoolArray] = []
    spec_times = np.empty(0, dtype=np.float64)
    conn_times = np.empty(0, dtype=np.float64)
    for r_idx in range(b_wave.shape[0]):
        spec_times, s_frames = extract_spectral_frames(
            b_wave[r_idx],
            protocol.target_fs,
            history=b_hist[r_idx],
            geometry=protocol.stft_geometry,
            target_fs=protocol.target_fs,
        )
        c_ext: ConnectivityExtraction = extract_connectivity_frames(
            b_wave[r_idx],
            protocol.target_fs,
            history=b_hist[r_idx],
            window_s=w_conn,
            hop_s=protocol.hop_s,
            n_segment=protocol.connectivity_n_segment,
            n_welch_hop=protocol.connectivity_n_welch_hop,
            band_hz=protocol.connectivity_band_hz,
            target_fs=protocol.target_fs,
        )
        conn_times = c_ext.times
        spec_rows.append(s_frames)
        conn_vals_rows.append(c_ext.values)
        conn_mask_rows.append(c_ext.valid_mask)
    return (
        b_ids,
        wave_times,
        b_wave,
        spec_times,
        np.stack(spec_rows, axis=0),
        conn_times,
        np.stack(conn_vals_rows, axis=0),
        np.stack(conn_mask_rows, axis=0),
    )


@dataclass(frozen=True)
class EvaluationProtocol:
    """Estimator geometry, horizons, and statistics shared by the runner and scorer."""

    channel_set: str = "all62"
    target_fs: float = DEFAULT_WAVEFORM_FS
    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    stft_geometry: StftGeometry = field(default_factory=baseline_stft_geometry)
    spectral_one_sided: bool = True
    connectivity_candidate_windows_s: tuple[float, ...] = CONNECTIVITY_CANDIDATE_WINDOWS_S
    connectivity_repeatability_target: float = 0.25
    connectivity_band_hz: tuple[float, float] = SEIZURE_BAND_HZ
    connectivity_n_segment: int = 25
    connectivity_n_welch_hop: int = 3
    hop_s: float = DEFAULT_FRAME_HOP_S
    horizons_s: tuple[float, ...] = CONTROL_HORIZONS_S

    @property
    def frame_hop(self) -> int:
        """Return the number of Waveform samples per response Frame."""
        return max(1, round(self.hop_s * self.target_fs))

    def geometry_metadata(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Describe the configured Waveform, spectral, and connectivity extraction."""
        waveform_geometry: dict[str, Any] = {
            "fs": float(self.target_fs),
            "cutoff_hz": float(self.cutoff_hz),
            "frame_hop": int(self.frame_hop),
        }
        spectral_geometry: dict[str, Any] = {
            **self.stft_geometry.model_dump(mode="json"),
            "fs": float(self.target_fs),
            "one_sided": bool(self.spectral_one_sided),
            "healthy_envelope_quantile": HEALTHY_SPECTRAL_QUANTILE,
        }
        connectivity_geometry: dict[str, Any] = {
            "fs": float(self.target_fs),
            "hop_s": float(self.hop_s),
            "n_segment": int(self.connectivity_n_segment),
            "n_welch_hop": int(self.connectivity_n_welch_hop),
            "band_hz": [float(self.connectivity_band_hz[0]), float(self.connectivity_band_hz[1])],
            "min_segments": int(MIN_CONNECTIVITY_SEGMENTS),
            "psd_floor": float(DEFAULT_PSD_FLOOR),
            "valid_pair_threshold": float(CONNECTIVITY_VALID_PAIR_THRESHOLD),
            "candidate_windows_s": [float(w) for w in self.connectivity_candidate_windows_s],
            "repeatability_target": float(self.connectivity_repeatability_target),
        }
        return waveform_geometry, spectral_geometry, connectivity_geometry


def _align_spectral_penalty(penalty: FloatArray, native_times: FloatArray, response_times: FloatArray) -> FloatArray:
    """Align a spectral penalty at matching endpoints, leaving other response Frames missing."""
    if np.array_equal(native_times, response_times):
        return penalty
    native_ns = np.rint(native_times * 1e9).astype(np.int64)
    response_ns = np.rint(response_times * 1e9).astype(np.int64)
    _, native_indices, response_indices = np.intersect1d(
        native_ns, response_ns, assume_unique=True, return_indices=True
    )
    aligned = np.full((penalty.shape[0], len(response_times)), np.nan)
    aligned[:, response_indices] = penalty[:, native_indices]
    return aligned


def _try_load_cached_evaluation_bundle(  # noqa: PLR0913 -- cache path, manifest, expected metadata, and mismatch policy
    target_cache: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    identity_digest: str,
    waveform_geometry: dict[str, Any],
    spectral_geometry: dict[str, Any],
    connectivity_geometry: dict[str, Any],
    channel_set: str,
    horizons_tuple: tuple[float, ...],
    rebuild: bool,
    on_mismatch: Literal["raise", "rebuild"],
) -> ObservableEvaluationBundle | None:
    """Attempt to load and validate an existing evaluation score cache, returning ``None`` when absent or rebuilt."""
    if not target_cache.exists() or rebuild:
        return None
    expected_header = ScoreCacheMetadata(
        schema_version=int(manifest.get("schema_version", SCHEMA_VERSION)),
        manifest_hash=manifest_hash,
        identity_digest=identity_digest,
        waveform_geometry=waveform_geometry,
        spectral_geometry=spectral_geometry,
        connectivity_geometry=connectivity_geometry,
        selected_connectivity_window_s=float("nan"),
        channel_set=channel_set,
        horizons_s=horizons_tuple,
    )
    try:
        return load_evaluation_bundle(target_cache, manifest=manifest, expected_metadata=expected_header)
    except ValueError:
        if on_mismatch == "raise":
            raise
        return None


DEFAULT_EVALUATION_PROTOCOL = EvaluationProtocol()


def load_or_score_ensemble(
    out_dir: Path | str,
    *,
    protocol: EvaluationProtocol = DEFAULT_EVALUATION_PROTOCOL,
    rebuild: bool = False,
    on_mismatch: Literal["raise", "rebuild"] = "raise",
    cache_path: Path | str | None = None,
) -> ObservableEvaluationBundle:
    """Fit healthy references, score seizure replays, and cache the two criteria."""
    base_dir = Path(out_dir)
    manifest = load_manifest(base_dir)
    channel_set = protocol.channel_set
    if channel_set not in manifest.get("channel_sets", {}):
        msg = f"Unknown channel_set {channel_set!r}; available={list(manifest.get('channel_sets', {}))}."
        raise ValueError(msg)
    channels = np.asarray(manifest["channel_sets"][channel_set], dtype=np.intp)
    manifest_hash, identity_digest = compute_manifest_hashes(manifest)
    horizons_tuple = protocol.horizons_s
    waveform_geometry, spectral_geometry, connectivity_geometry = protocol.geometry_metadata()

    target_cache = (base_dir / EVALUATION_SCORES_FILE) if cache_path is None else Path(cache_path)
    cached_bundle = _try_load_cached_evaluation_bundle(
        target_cache,
        manifest=manifest,
        manifest_hash=manifest_hash,
        identity_digest=identity_digest,
        waveform_geometry=waveform_geometry,
        spectral_geometry=spectral_geometry,
        connectivity_geometry=connectivity_geometry,
        channel_set=channel_set,
        horizons_tuple=horizons_tuple,
        rebuild=rebuild,
        on_mismatch=on_mismatch,
    )
    if cached_bundle is not None and cached_bundle.potential_predictability:
        return cached_bundle

    (
        waveform_reference,
        spectral_reference,
        connectivity_reference,
        connectivity_support,
    ) = _fit_references_and_freeze_dev_parameters(
        base_dir,
        manifest,
        channels=channels,
        protocol=protocol,
    )
    w_conn = connectivity_support.selected_window_s

    seizure_replay_moments: dict[str, list[ReplayMoments]] = {rep: [] for rep in REPRESENTATIONS}
    zero_penalties_by_branch: dict[str, dict[str, FloatArray]] = {rep: {} for rep in REPRESENTATIONS}
    zero_states_by_branch: dict[str, FloatArray] = {}
    zero_ids_by_branch: dict[str, tuple[RolloutIdentity, ...]] = {}

    eval_frame_times = np.empty(0, dtype=np.float64)
    eval_state_times = np.empty(0, dtype=np.float64)

    for b_info in manifest["splits"]["eval"]["branches"]:
        if b_info["plant"] != "seizure":
            continue
        branch = str(b_info["name"])
        arm_str = "zero"
        (
            b_ids,
            wave_times,
            b_wave,
            spec_times,
            b_spec,
            conn_times,
            b_conn_vals,
            b_conn_mask,
        ) = _extract_eval_branch_arm_observables(
            base_dir,
            manifest,
            branch=branch,
            arm_str=arm_str,
            channels=channels,
            w_conn=w_conn,
            protocol=protocol,
        )
        eval_frame_times = conn_times

        for rep, values, times in (
            ("waveform", b_wave, wave_times),
            ("spectral", b_spec, spec_times),
            ("connectivity", b_conn_vals, conn_times),
        ):
            seizure_replay_moments[rep].append(compute_replay_moments(values, times, b_ids))

        eval_state_times, b_states = _extract_split_branch_arm_seizure_state(
            base_dir,
            manifest,
            split="eval",
            branch=branch,
            arm=arm_str,
            identities=b_ids,
            hop_s=protocol.frame_hop / protocol.target_fs,
        )
        pen_by_rep = {
            "waveform": compute_waveform_penalty(b_wave, waveform_reference, frame_hop=protocol.frame_hop),
            "spectral": _align_spectral_penalty(
                compute_spectral_penalty(b_spec, spectral_reference, one_sided=protocol.spectral_one_sided),
                spec_times,
                eval_frame_times,
            ),
            "connectivity": compute_connectivity_penalty(
                b_conn_vals, connectivity_reference, valid_mask=b_conn_mask
            ).penalty,
        }
        for rep in REPRESENTATIONS:
            zero_penalties_by_branch[rep][branch] = pen_by_rep[rep]
        zero_states_by_branch[branch] = b_states
        zero_ids_by_branch[branch] = b_ids

    seizure_relevance = {
        rep: score_zero_current_seizure_relevance(
            zero_penalties_by_branch[rep],
            zero_states_by_branch,
            identities=zero_ids_by_branch,
            penalty_times=eval_frame_times,
            state_times=eval_state_times,
        )
        for rep in REPRESENTATIONS
    }
    metadata = ScoreCacheMetadata(
        schema_version=int(manifest.get("schema_version", SCHEMA_VERSION)),
        manifest_hash=manifest_hash,
        identity_digest=identity_digest,
        waveform_geometry=waveform_geometry,
        spectral_geometry=spectral_geometry,
        connectivity_geometry=connectivity_geometry,
        selected_connectivity_window_s=float(w_conn),
        channel_set=channel_set,
        horizons_s=horizons_tuple,
    )
    bundle = ObservableEvaluationBundle(
        metadata=metadata,
        manifest=manifest,
        waveform_reference=waveform_reference,
        spectral_reference=spectral_reference,
        connectivity_reference=connectivity_reference,
        connectivity_support=connectivity_support,
        seizure_relevance=seizure_relevance,
        potential_predictability={
            rep: compute_potential_predictability(blocks) for rep, blocks in seizure_replay_moments.items() if blocks
        },
    )
    save_evaluation_bundle(target_cache, bundle)
    return bundle
