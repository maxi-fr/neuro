from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

# The plant is what the predictor is identified on; the seed only picks a realisation of it and
# ``log`` only selects what is written out, so neither belongs in its fingerprint.
_VOLATILE_DYNAMICS_KEYS = frozenset({"seed", "log"})


def plant_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the plant-defining blocks -- ``dynamics`` and ``sensors`` -- of a simulation config."""
    dynamics = {k: v for k, v in config.get("dynamics", {}).items() if k not in _VOLATILE_DYNAMICS_KEYS}
    payload = json.dumps({"dynamics": dynamics, "sensors": config.get("sensors")}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _generating_config(data_dir: Path) -> dict[str, Any] | None:
    """Get the config ``run_simulation`` copied in beside the trajectories, or ``None`` if unrecoverable.

    In an ``experiments:`` batch the first entry carries the full config and the rest are seed-only
    overrides, so the first entry is the run.
    """
    configs = sorted(data_dir.glob("*.yaml"))
    if len(configs) != 1:
        return None
    raw = yaml.safe_load(configs[0].read_text())
    return raw["experiments"][0] if "experiments" in raw else raw


def data_plant_fingerprint(data_dir: Path) -> str | None:
    """Fingerprint the plant a trajectory directory was generated on, or ``None`` if unrecoverable."""
    config = _generating_config(data_dir)
    return None if config is None else plant_fingerprint(config)


def check_excitation_alignment(data_dir: Path, downsample: int) -> None:
    """Warn when the excitation in ``data_dir`` switches off the grid ``downsample`` strides on.

    A block boundary landing mid-step leaves the strided control recording a command the plant only
    partly held, so the predictor is identified against an input that was never applied.
    """
    controller = (_generating_config(data_dir) or {}).get("controller", {})
    if not str(controller.get("class_path", "")).endswith("WaveformController"):
        return
    if controller.get("input_type") not in ("ras", "prbs"):
        return

    dt = float(controller["dt"])
    holds = np.atleast_1d(np.asarray(controller.get("hold_ms", 50.0), dtype=np.float64))
    # Mirrors the rounding ``build_input_schedule`` lays the blocks out on.
    hold_steps = np.maximum(1, np.round(holds / (dt * 1000.0)).astype(int))
    ragged = sorted(float(ms) for ms, steps in zip(holds, hold_steps, strict=True) if steps % downsample)
    if ragged:
        warnings.warn(
            f"excitation holds {ragged} ms in {data_dir} are not whole multiples of the "
            f"{downsample * dt:g} s predictor step; the strided control records commands the plant "
            f"only partly held.",
            stacklevel=2,
        )


@dataclass(frozen=True)
class TrainingProvenance:
    """What a predictor's training data was made of, carried in its checkpoint.

    Everything here is what the closed-loop config has to agree with but cannot read off the
    weights, so :func:`neuro.validation.validate_simulation_config` can catch a predictor deployed
    against a plant, an anti-alias filter or a current range other than the one it was fit on.

    Attributes
    ----------
    cutoff_hz : float | None
        Explicit low-pass cutoff the trajectories were decimated with; ``None`` means the decimated
        Nyquist rate, which is what :class:`neuro.filtering.AntiAliasEstimator` reproduces online.
    plant_fingerprint : str | None
        :func:`plant_fingerprint` of the config that generated the trajectories.
    u_max : float | None
        Largest absolute current in the training split, per electrode.
    """

    cutoff_hz: float | None = None
    plant_fingerprint: str | None = None

    @property
    def meta(self) -> dict[str, Any]:
        """Serializable representation, flattened into the checkpoint's own ``meta`` mapping."""
        return {"cutoff_hz": self.cutoff_hz, "plant_fingerprint": self.plant_fingerprint}

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> Self:
        """Read back from a checkpoint's ``meta``; checkpoints written before it recorded nothing."""
        return cls(
            cutoff_hz=meta.get("cutoff_hz"),
            plant_fingerprint=meta.get("plant_fingerprint"),
        )


def training_provenance(data_files: Sequence[str], cutoff_hz: float | None) -> TrainingProvenance:
    """Assemble the provenance of a training run whose split loaded ``data_files``."""
    return TrainingProvenance(
        cutoff_hz=cutoff_hz,
        plant_fingerprint=data_plant_fingerprint(Path(data_files[0]).parent),
    )
