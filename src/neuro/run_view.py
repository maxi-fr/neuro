from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from neuro.config import StftGeometry
from neuro.eeg import build_eeg_leadfield
from neuro.seizure import SEIZURE_PTP_MV, spread_profile_from_lfp
from neuro.stimulation.base import select_rows

if TYPE_CHECKING:
    from neuro.types import FloatArray


_NEAR_ZERO_THRESHOLD = 1e-6


@dataclass(frozen=True)
class PredictionSeries:
    """Decision times, Rollouts ``(n, H+1, p)``, and independently recorded outputs ``(n, p)``."""

    t: FloatArray
    predicted_y: FloatArray
    observed_y: FloatArray
    dt: float
    frequencies: FloatArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @classmethod
    def load(cls, path: Path) -> PredictionSeries:
        """Read a prepared Rollout collection without loading a Predictor."""
        with np.load(path, allow_pickle=False) as data:
            return cls(data["t"], data["predicted_y"], data["observed_y"], float(data["dt"]), data["frequencies"])

    def save(self, path: Path) -> None:
        """Persist the typed Rollout collection as numeric arrays."""
        np.savez_compressed(
            path,
            t=self.t,
            predicted_y=self.predicted_y,
            observed_y=self.observed_y,
            dt=self.dt,
            frequencies=self.frequencies,
        )


@dataclass
class Run:
    """One selected simulate run, with its config and component-rate arrays."""

    directory: Path
    config: dict[str, Any]
    arrays: dict[str, FloatArray] = field(repr=False)

    @classmethod
    def load(cls, directory: Path) -> Run:
        """Read one run on selection; directory discovery never loads its arrays."""
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        with np.load(directory / "log.npz", allow_pickle=False) as archive:
            arrays = dict(archive)
        return cls(directory, config, arrays)

    def signal(self, component: str, name: str) -> tuple[FloatArray, FloatArray]:
        """Return the component's own timestamps and its signal."""
        return self.arrays[f"{component}.t"], self.arrays[f"{component}.{name}"]

    def sampled_signal(self, component: str, name: str) -> FloatArray:
        """Sample a held component signal at controller decisions without looking ahead."""
        times, values = self.signal(component, name)
        decisions = self.arrays["controller.t"]
        indices = np.searchsorted(times, decisions, side="right") - 1
        if np.any(indices < 0):
            msg = f"{component}.{name} starts after the first controller decision."
            raise ValueError(msg)
        return values[indices].reshape(len(decisions), -1)

    def measurements(self) -> FloatArray:
        """Read decision measurements from the Estimator, or the Sensor for identity estimation."""
        if "estimator.x_hat" in self.arrays:
            return self.sampled_signal("estimator", "x_hat")
        return self.sampled_signal("sensor_0", "y_mea")

    def observed_outputs(self) -> FloatArray:
        """Read physical outputs independently of the controller's solved states."""
        if "sensor_0.lfp" in self.arrays:
            regional = self.sampled_signal("sensor_0", "lfp")
            leadfield = self.config["controller"]["problem"].get("leadfield")
            return regional if leadfield is None else regional @ np.asarray(leadfield).T
        return self.measurements()

    def frequencies(self) -> FloatArray:
        """Return pooled frequency centers for Observable Frames, or no bins for waveform outputs."""
        estimator = self.config.get("estimator", {})
        if "geometry" not in estimator:
            return np.empty(0, dtype=np.float64)
        geometry = StftGeometry.model_validate(estimator["geometry"])
        fs = 1.0 / (float(estimator["dt"]) * int(estimator.get("downsample", 1)))
        lo, hi = geometry.bin_range(fs)
        frequencies = np.fft.rfftfreq(geometry.n_segment, 1.0 / fs)[lo:hi]
        count = geometry.n_values(fs)
        return frequencies[: count * geometry.n_bin_pool].reshape(count, geometry.n_bin_pool).mean(axis=1)

    def planned_predictions(self) -> PredictionSeries:
        """Pair controller Rollouts with independently logged physical outputs."""
        times, outputs = self.signal("controller", "predicted_y")
        return PredictionSeries(
            times, outputs, self.observed_outputs(), float(self.config["controller"]["dt"]), self.frequencies()
        )

    def eeg(self) -> tuple[FloatArray, FloatArray]:
        """Read the Sensor's EEG log, including the explicit EEG field of oracle Sensors."""
        return self.signal("sensor_0", "eeg" if "sensor_0.eeg" in self.arrays else "y_mea")

    def eeg_channel_labels(self) -> list[str]:
        """Return the EEG labels in the same order as the logged channels."""
        _, labels = build_eeg_leadfield()
        selected = self.config.get("sensors", {}).get("measurement", {}).get("selected_channels")
        if selected is None:
            return [str(label) for label in labels[: self.eeg()[1].shape[1]]]
        indices = {str(label): index for index, label in enumerate(labels)}
        return [str(labels[indices[label] if isinstance(label, str) else label]) for label in selected]

    def electrode_labels(self) -> list[str]:
        """Return electrode labels for stimulation channels."""
        stim = self.config.get("dynamics", {}).get("stimulation", {})
        path_str = stim.get("field_projection_path")
        n_u = self.arrays["controller.u"].shape[-1] if "controller.u" in self.arrays else 0
        if path_str is not None and n_u > 0:
            path = Path(path_str)
            if path.exists():
                with np.load(path) as projection:
                    labels = projection["channel_labels"]
                wanted = stim.get("electrodes")
                rows = select_rows(labels, wanted) if wanted else slice(None)
                selected = [str(label) for label in labels[rows]]
                if len(selected) == n_u:
                    return selected
        return [f"E{i}" for i in range(n_u)]

    def metrics(self) -> dict[str, Any]:
        """Compute key summary metrics for this closed-loop run."""
        results: dict[str, Any] = {"Run": self.directory.name}
        if "sensor_0.y_mea" in self.arrays or "sensor_0.eeg" in self.arrays:
            _, y = self.eeg()
            results["EEG Energy mean(y²)"] = round(float(np.mean(y**2)), 4)
        if "controller.u" in self.arrays:
            _, u = self.signal("controller", "u")
            u_flat = u.reshape(u.shape[0], -1)
            results["Control Energy Σu²"] = round(float(np.sum(u_flat**2)), 4)
            results["Control L1 Σ|u|"] = round(float(np.sum(np.abs(u_flat))), 4)
            results["Max |u|"] = round(float(np.max(np.abs(u_flat))), 4)
            results["KCL Max |Σu|"] = f"{float(np.max(np.abs(u_flat.sum(axis=-1)))):.2e}"
            near_zero = np.abs(u_flat) < _NEAR_ZERO_THRESHOLD
            results["Mean Active Electrodes"] = round(float((~near_zero).sum(axis=1).mean()), 2)
        if "dynamics.lfp" in self.arrays:
            _, lfp_arr = self.signal("dynamics", "lfp")
            dt = float(self.config.get("dynamics", {}).get("dt", 1e-4))
            profile = spread_profile_from_lfp(lfp_arr.T, dt, threshold=SEIZURE_PTP_MV)
            results["Seizure Burden mean s(t)"] = round(float(profile.burden()), 4)
        return results


def discover_runs(root: Path) -> list[Path]:
    """Find run bundles recursively without loading Rollout arrays."""
    return sorted(path.parent for path in root.rglob("log.npz") if (path.parent / "config.yaml").is_file())


@dataclass
class ViewSettings:
    """Every plot choice needed to reopen a view, with paths relative to the collection."""

    mode: str = "Closed-loop runs"
    runs: list[str] = field(default_factory=list)
    replays: list[str] = field(default_factory=list)
    start: float = 0.0
    width: float = 12.0
    decision: float = 6.0
    neighbors: int = 0
    spacing: int = 5
    eeg_channels: list[int] = field(default_factory=lambda: [0])
    output_channels: list[int] = field(default_factory=lambda: [0])
    electrodes: list[int] = field(default_factory=lambda: [0])
    regions: list[int] = field(default_factory=list)
    offset: float = 0.0
    note: str = ""
    show_spectra: bool = True
    spectral_steps: int = 20
    spectral_frame: int = 1
    spectral_channels: list[int] = field(default_factory=lambda: [0])


def load_bookmarks(path: Path) -> dict[str, ViewSettings]:
    """Load named plot settings, or an empty collection before the first save."""
    if not path.exists():
        return {}
    return {name: ViewSettings(**values) for name, values in json.loads(path.read_text(encoding="utf-8")).items()}


def save_bookmark(path: Path, name: str, settings: ViewSettings) -> None:
    """Persist one named view without losing the collection's other bookmarks."""
    bookmarks = load_bookmarks(path)
    bookmarks[name] = settings
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps({key: asdict(value) for key, value in bookmarks.items()}, indent=2), encoding="utf-8")
    pending.replace(path)


def decision_indices(times: FloatArray, settings: ViewSettings) -> list[int]:
    """Select a decision and neighboring anchors at the requested decision spacing."""
    center = int(np.argmin(np.abs(times - settings.decision)))
    return [
        i
        for j in range(-settings.neighbors, settings.neighbors + 1)
        if 0 <= (i := center + j * settings.spacing) < len(times)
    ]
