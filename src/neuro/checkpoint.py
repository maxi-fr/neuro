from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.sparse

from neuro.observable import geometry_from_meta, geometry_meta
from neuro.predictor.artifact import ACTIVATIONS, Activation, Layers
from neuro.predictor.checkpoint import load_checkpoint, save_checkpoint
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.config import ObservableGeometry
    from neuro.esn import ESNArtifact
    from neuro.observable import ObservableArtifact
    from neuro.predictor.artifact import MLPArtifact
    from neuro.types import FloatArray, ObservableModel, SymbolicModel


@dataclass(frozen=True)
class MLPCheckpoint:
    """NumPy weights and recorded metadata of the autoregressive MLP, as read from its checkpoint.

    This is the torch-free twin of :class:`neuro.predictor.artifact.MLPArtifact` minus the numpy
    runtime: the CasADi adapter rebuilds its bridges from these buffers and the validator reads
    the recorded metadata off them, and neither path imports torch.

    Attributes
    ----------
    layers : Layers
        Per-layer ``(weight (out, in), bias (out,))`` pairs in forward-pass order.
    activation : Activation
        Activation applied after every layer except the last.
    n_y, n_u : int
        Past EEG and past control steps in the model's history window.
    horizon : int
        Direct-prediction horizon the model was trained on.
    n_channels, n_controls : int
        Physical EEG channel and control channel counts.
    hidden_size, depth : int
        Architecture record: hidden width and hidden-layer count.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    y_std, u_std : Standardizer
        Channel and control standardizers mapping raw units to model space.
    provenance : TrainingProvenance
        What the training data was made of; empty on checkpoints written before it was recorded.
    """

    layers: Layers
    activation: Activation
    n_y: int
    n_u: int
    horizon: int
    n_channels: int
    n_controls: int
    hidden_size: int
    depth: int
    dt: float
    downsample: int
    y_std: Standardizer
    u_std: Standardizer
    provenance: TrainingProvenance = field(default_factory=TrainingProvenance)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('mlp')."""
        return "mlp"

    @property
    def priming_steps(self) -> int:
        """Minimum number of history steps required to initialize the model state."""
        return max(self.n_y, self.n_u)

    @property
    def is_linear(self) -> bool:
        """Whether the MLP is linear (a single layer, i.e. no hidden layers)."""
        return len(self.layers) == 1

    @classmethod
    def from_artifact(cls, art: MLPArtifact) -> Self:
        """Build the checkpoint twin of an artifact: the legacy bridge for the artifact dispatch.

        The artifacts are on their way out (the contract ticket deletes them), so this exists only
        to keep the probe-script dispatch and the adapter constructors type-correct until then.
        """
        depth = len(art.layers) - 1
        return cls(
            layers=art.layers,
            activation=art.activation,
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            hidden_size=art.layers[0][0].shape[0] if depth else 0,
            depth=depth,
            dt=art.dt,
            downsample=art.downsample,
            y_std=art.y_std,
            u_std=art.u_std,
            provenance=art.provenance,
        )

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizers and metadata into one ``.npz`` checkpoint (a stem).

        The layout matches what :meth:`neuro.predictor.module.AutoregressiveMLP.save` writes, so
        the torch module's loader and this reader are interchangeable on the file.
        """
        meta = {
            "model_type": self.model_type,
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "dt": self.dt,
            "downsample": self.downsample,
            "n_layers": len(self.layers),
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer.{i}.weight"] = w
            arrays[f"layer.{i}.bias"] = b
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        save_checkpoint(path, meta=meta, arrays=arrays)


@dataclass(frozen=True)
class ESNCheckpoint:
    """NumPy weights and recorded metadata of the ESN, as read from its checkpoint.

    The torch-free twin of :class:`neuro.esn.ESNArtifact` minus the numpy runtime; the CasADi
    adapter and the validator read these buffers without importing torch.

    Attributes
    ----------
    w_in : FloatArray
        Dense input weight matrix (N, C + m + 1).
    w_out : FloatArray
        Dense readout weight matrix (C, N + 1).
    w_res : scipy.sparse.csr_matrix
        Sparse reservoir weight matrix (N, N).
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    horizon : int
        The native/trained horizon, identity metadata only.
    reservoir_size : int
        Number of reservoir units N.
    leak_rate : float
        Leakage alpha in (0, 1].
    spectral_radius, density, input_scaling : float
        Reservoir generation hyperparameters, recorded metadata.
    priming_steps : int
        Minimum number of absorbed samples before the state is ready.
    noise_sigma, ridge_lambda : float
        Harvest noise-injection sigma and readout-fit ridge regularization, recorded metadata.
    seed : int
        Reservoir generation seed, recorded metadata.
    y_std, u_std : Standardizer
        Channel and control standardizers mapping raw units to model space.
    provenance : TrainingProvenance
        What the training data was made of; empty on checkpoints written before it was recorded.
    """

    w_in: FloatArray
    w_out: FloatArray
    w_res: scipy.sparse.csr_matrix
    dt: float
    downsample: int
    horizon: int
    reservoir_size: int
    leak_rate: float
    spectral_radius: float
    priming_steps: int
    input_scaling: float
    density: float
    noise_sigma: float
    ridge_lambda: float
    seed: int
    y_std: Standardizer
    u_std: Standardizer
    provenance: TrainingProvenance = field(default_factory=TrainingProvenance)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('esn')."""
        return "esn"

    @property
    def n_channels(self) -> int:
        """Number of EEG channels C."""
        return self.w_out.shape[0]

    @property
    def n_controls(self) -> int:
        """Number of control input channels m, read off W_in's ``[z; v; 1]`` input width."""
        return self.w_in.shape[1] - self.n_channels - 1

    @classmethod
    def from_artifact(cls, art: ESNArtifact) -> Self:
        """Build the checkpoint twin of an artifact: the legacy bridge for the artifact dispatch.

        The artifacts are on their way out (the contract ticket deletes them), so this exists only
        to keep the probe-script dispatch and the adapter constructors type-correct until then.
        """
        return cls(
            w_in=art.w_in,
            w_out=art.w_out,
            w_res=art.w_res,
            dt=art.dt,
            downsample=art.downsample,
            horizon=art.horizon,
            reservoir_size=art.reservoir_size,
            leak_rate=art.leak_rate,
            spectral_radius=art.spectral_radius,
            priming_steps=art.priming_steps,
            input_scaling=art.input_scaling,
            density=art.density,
            noise_sigma=art.noise_sigma,
            ridge_lambda=art.ridge_lambda,
            seed=art.seed,
            y_std=art.y_std,
            u_std=art.u_std,
            provenance=art.provenance,
        )

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizers and metadata into one ``.npz`` checkpoint (a stem).

        The layout matches what :meth:`neuro.predictor.esn_module.ESNModule.save` writes, so the
        torch module's loader and this reader are interchangeable on the file.
        """
        meta = {
            "model_type": self.model_type,
            "dt": self.dt,
            "downsample": self.downsample,
            "horizon": self.horizon,
            "reservoir_size": self.reservoir_size,
            "leak_rate": self.leak_rate,
            "spectral_radius": self.spectral_radius,
            "priming_steps": self.priming_steps,
            "input_scaling": self.input_scaling,
            "density": self.density,
            "noise_sigma": self.noise_sigma,
            "ridge_lambda": self.ridge_lambda,
            "seed": self.seed,
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {
            "W_in": self.w_in,
            "W_out": self.w_out,
            "W_res.data": np.asarray(self.w_res.data, dtype=np.float64),
            "W_res.indices": self.w_res.indices,
            "W_res.indptr": self.w_res.indptr,
            "W_res.shape": np.array(self.w_res.shape),
        }
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        save_checkpoint(path, meta=meta, arrays=arrays)


@dataclass(frozen=True)
class ObservableCheckpoint:
    """NumPy weights and recorded metadata of the observable predictor, as read from its checkpoint.

    The torch-free twin of :class:`neuro.observable.ObservableArtifact` minus the numpy runtime;
    the CasADi adapter and the validator read these buffers without importing torch.

    Attributes
    ----------
    lift, transition : Layers
        ``E`` maps the history state to ``z_0``; ``f_theta(z_m, u_bar_m)`` advances one Frame and
        is shared across every Frame.
    readout : tuple[FloatArray, FloatArray]
        The affine ``(C, c)`` emitting standardized log-Observable.
    activation : Activation
        Activation applied after every layer of ``lift`` and ``transition`` except their last.
    n_y, n_u : int
        Past EEG and past control steps in the history window ``E`` lifts from.
    horizon : int
        The Control Horizon in samples the model was fit at; the Frame count follows from it and
        ``geometry``.
    n_channels, n_controls : int
        Physical EEG channel count and control electrode count.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    geometry : ObservableGeometry
        The Observable grid the model was trained against.
    y_std, u_std, l_std : Standardizer
        Channel, control and log-Observable standardizers.
    provenance : TrainingProvenance
        What the training data was made of.
    """

    lift: Layers
    transition: Layers
    readout: tuple[FloatArray, FloatArray]
    activation: Activation
    n_y: int
    n_u: int
    horizon: int
    n_channels: int
    n_controls: int
    dt: float
    downsample: int
    geometry: ObservableGeometry
    y_std: Standardizer
    u_std: Standardizer
    l_std: Standardizer
    provenance: TrainingProvenance = field(default_factory=TrainingProvenance)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('observable')."""
        return "observable"

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        return 1.0 / self.dt

    @property
    def priming_steps(self) -> int:
        """Minimum number of history samples required to prime the state."""
        return max(self.n_y, self.n_u)

    @property
    def z_dim(self) -> int:
        """Dimension of the lifted Frame state."""
        return self.readout[0].shape[1]

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        return self.geometry.n_values(self.fs)

    def n_frames(self, horizon: int | None = None) -> int:
        """Frames the recursion emits over ``horizon`` samples (default: the trained horizon)."""
        return self.geometry.n_frames(self.horizon if horizon is None else horizon, self.fs)

    @classmethod
    def from_artifact(cls, art: ObservableArtifact) -> Self:
        """Build the checkpoint twin of an artifact: the legacy bridge for the artifact dispatch.

        The artifacts are on their way out (the contract ticket deletes them), so this exists only
        to keep the probe-script dispatch and the adapter constructors type-correct until then.
        """
        return cls(
            lift=art.lift,
            transition=art.transition,
            readout=art.readout,
            activation=art.activation,
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            dt=art.dt,
            downsample=art.downsample,
            geometry=art.geometry,
            y_std=art.y_std,
            u_std=art.u_std,
            l_std=art.l_std,
            provenance=art.provenance,
        )

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizers and metadata into one ``.npz`` checkpoint (a stem).

        The layout matches what :meth:`neuro.predictor.observable_module.StepwiseObservableMLP.save`
        writes, so the torch module's loader and this reader are interchangeable on the file.
        """
        # The module builds every hidden layer at the same width, so the architecture record is
        # readable off the first layer of each block.
        meta = {
            "model_type": self.model_type,
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "dt": self.dt,
            "downsample": self.downsample,
            "z_dim": self.z_dim,
            "lift_hidden": self.lift[0][0].shape[0],
            "lift_depth": len(self.lift) - 1,
            "transition_hidden": self.transition[0][0].shape[0],
            "transition_depth": len(self.transition) - 1,
            "n_lift_layers": len(self.lift),
            "n_transition_layers": len(self.transition),
            "geometry": geometry_meta(self.geometry),
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {}
        for prefix, block in (("lift", self.lift), ("transition", self.transition)):
            for i, (w, b) in enumerate(block):
                arrays[f"{prefix}.{i}.weight"] = w
                arrays[f"{prefix}.{i}.bias"] = b
        arrays["readout.weight"], arrays["readout.bias"] = self.readout
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        arrays.update(self.l_std.arrays("l"))
        save_checkpoint(path, meta=meta, arrays=arrays)


def load_mlp(path: str | Path) -> MLPCheckpoint:
    """Read the MLP checkpoint's weights, standardizers and recorded metadata (a suffix-less stem)."""
    meta, arrays = load_checkpoint(path)
    if meta.get("model_type") != "mlp":
        msg = f"checkpoint at {path} is model_type {meta.get('model_type')!r}, not 'mlp'."
        raise ValueError(msg)
    if meta["activation"] not in ACTIVATIONS:
        msg = f"Unsupported activation: {meta['activation']!r}"
        raise ValueError(msg)
    layers = tuple(
        (
            np.asarray(arrays[f"layer.{i}.weight"], dtype=np.float64),
            np.asarray(arrays[f"layer.{i}.bias"], dtype=np.float64),
        )
        for i in range(int(meta["n_layers"]))
    )
    depth = len(layers) - 1
    return MLPCheckpoint(
        layers=layers,
        activation=meta["activation"],
        n_y=int(meta["n_y"]),
        n_u=int(meta["n_u"]),
        horizon=int(meta["horizon"]),
        n_channels=int(meta["n_channels"]),
        n_controls=int(meta["n_controls"]),
        hidden_size=int(meta["hidden_size"]) if "hidden_size" in meta else (layers[0][0].shape[0] if depth else 0),
        depth=depth,
        dt=float(meta["dt"]),
        downsample=int(meta["downsample"]),
        y_std=Standardizer.from_arrays(arrays, "y"),
        u_std=Standardizer.from_arrays(arrays, "u"),
        provenance=TrainingProvenance.from_meta(meta),
    )


def load_esn(path: str | Path) -> ESNCheckpoint:
    """Read the ESN checkpoint's weights, standardizers and recorded metadata (a suffix-less stem)."""
    meta, arrays = load_checkpoint(path)
    if meta.get("model_type") != "esn":
        msg = f"checkpoint at {path} is model_type {meta.get('model_type')!r}, not 'esn'."
        raise ValueError(msg)
    w_res = scipy.sparse.csr_matrix(
        (arrays["W_res.data"], arrays["W_res.indices"], arrays["W_res.indptr"]),
        shape=tuple(int(s) for s in arrays["W_res.shape"]),
    )
    return ESNCheckpoint(
        w_in=np.asarray(arrays["W_in"], dtype=np.float64),
        w_out=np.asarray(arrays["W_out"], dtype=np.float64),
        w_res=w_res,
        dt=float(meta["dt"]),
        downsample=int(meta["downsample"]),
        horizon=int(meta["horizon"]),
        reservoir_size=int(meta["reservoir_size"]),
        leak_rate=float(meta["leak_rate"]),
        spectral_radius=float(meta["spectral_radius"]),
        priming_steps=int(meta["priming_steps"]),
        input_scaling=float(meta["input_scaling"]),
        density=float(meta["density"]),
        noise_sigma=float(meta["noise_sigma"]),
        ridge_lambda=float(meta["ridge_lambda"]),
        seed=int(meta["seed"]),
        y_std=Standardizer.from_arrays(arrays, "y"),
        u_std=Standardizer.from_arrays(arrays, "u"),
        provenance=TrainingProvenance.from_meta(meta),
    )


def load_observable(path: str | Path) -> ObservableCheckpoint:
    """Read the observable checkpoint's weights, standardizers and recorded metadata (a stem)."""
    meta, arrays = load_checkpoint(path)
    if meta.get("model_type") != "observable":
        msg = f"checkpoint at {path} is model_type {meta.get('model_type')!r}, not 'observable'."
        raise ValueError(msg)
    if meta["activation"] not in ACTIVATIONS:
        msg = f"Unsupported activation: {meta['activation']!r}"
        raise ValueError(msg)

    def stack(prefix: str, count: int) -> Layers:
        return tuple(
            (
                np.asarray(arrays[f"{prefix}.{i}.weight"], dtype=np.float64),
                np.asarray(arrays[f"{prefix}.{i}.bias"], dtype=np.float64),
            )
            for i in range(count)
        )

    return ObservableCheckpoint(
        lift=stack("lift", int(meta["n_lift_layers"])),
        transition=stack("transition", int(meta["n_transition_layers"])),
        readout=(
            np.asarray(arrays["readout.weight"], dtype=np.float64),
            np.asarray(arrays["readout.bias"], dtype=np.float64),
        ),
        activation=meta["activation"],
        n_y=int(meta["n_y"]),
        n_u=int(meta["n_u"]),
        horizon=int(meta["horizon"]),
        n_channels=int(meta["n_channels"]),
        n_controls=int(meta["n_controls"]),
        dt=float(meta["dt"]),
        downsample=int(meta["downsample"]),
        geometry=geometry_from_meta(meta["geometry"]),
        y_std=Standardizer.from_arrays(arrays, "y"),
        u_std=Standardizer.from_arrays(arrays, "u"),
        l_std=Standardizer.from_arrays(arrays, "l"),
        provenance=TrainingProvenance.from_meta(meta),
    )


Checkpoint = MLPCheckpoint | ESNCheckpoint | ObservableCheckpoint
"""Checkpoint dataclasses the torch-free reader yields and the CasADi adapter rebuilds from."""
RolloutCheckpoint = MLPCheckpoint | ESNCheckpoint
"""Checkpoints that free-run on the sample grid; the observable one forecasts the Observable instead."""


def load_any(path: str | Path) -> Checkpoint:
    """Read a single ``.npz`` predictor checkpoint (MLP, ESN or observable) from disk, torch-free."""
    meta, _ = load_checkpoint(path)
    model_type = meta["model_type"]
    if model_type == "mlp":
        return load_mlp(path)
    if model_type == "esn":
        return load_esn(path)
    if model_type == "observable":
        return load_observable(path)
    msg = f"unsupported model_type {model_type!r} in {Path(path).with_suffix('.npz')}"
    raise ValueError(msg)


def load_rollout(path: str | Path) -> RolloutCheckpoint:
    """Read a checkpoint that free-runs on the sample grid, rejecting an observable one."""
    ckpt = load_any(path)
    if isinstance(ckpt, ObservableCheckpoint):
        msg = f"{path} is an observable checkpoint; it forecasts the Observable and never a waveform."
        raise TypeError(msg)
    return ckpt


def build_symbolic_model(ckpt: Checkpoint) -> SymbolicModel | ObservableModel:
    """Build the appropriate CasADi adapter; the MPC branches on which of the two it gets."""
    from neuro.esn_predictor_casadi import (  # noqa: PLC0415 -- the adapters import this module; top-level would cycle
        ESNSymbolicModel,
    )
    from neuro.nn_predictor_casadi import (  # noqa: PLC0415 -- the adapters import this module; top-level would cycle
        NNSymbolicModel,
    )
    from neuro.observable_casadi import (  # noqa: PLC0415 -- the adapters import this module; top-level would cycle
        ObservableSymbolicModel,
    )

    if isinstance(ckpt, ESNCheckpoint):
        return ESNSymbolicModel(ckpt)
    if isinstance(ckpt, ObservableCheckpoint):
        return ObservableSymbolicModel(ckpt)
    return NNSymbolicModel(ckpt)
