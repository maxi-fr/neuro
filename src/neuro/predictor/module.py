from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from neuro.config import StftGeometry
from neuro.predictor.checkpoint import (
    layer_arrays,
    layers_from_arrays,
    load_checkpoint,
    require_activation,
    require_model_type,
    save_checkpoint,
)
from neuro.predictor.data import build_dataset_for_trajectory
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor

    from neuro.types import Activation, FloatArray


def activation_module(activation: Activation) -> nn.Module:
    """Return the torch activation module matching the named activation."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    return nn.Softplus()


def to_numpy(t: Tensor) -> FloatArray:
    """Detach a parameter into an owned float64 NumPy array."""
    return t.detach().cpu().numpy().astype(np.float64, copy=True)


class TrainingPredictor(ABC):
    """Training-side interface every torch predictor implements.

    ``forward`` is the batched standardized Rollout over the trained Span; ``save``/``load``
    round-trip the exchange checkpoint through ``to_checkpoint``/``from_checkpoint``. Channel and
    control counts, ``dt``, ``horizon`` and the standardizers are part of the contract by
    documentation, not abstract enforcement.
    """

    @abstractmethod
    def forward(self, y_hist: Tensor, u_hist: Tensor, u_future: Tensor) -> Tensor:
        """Roll out batched standardized inputs over the trained Span."""
        ...

    @abstractmethod
    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the exchange checkpoint is written from."""
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the module from a ``(meta, arrays)`` pair, in memory."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizer buffers and recorded metadata into one ``.npz`` checkpoint.

        ``path`` is a suffix-less stem. The layout is the one both sides read and write, so the
        jax inference side consumes what ``save`` writes without the module.
        """
        meta, arrays = self.to_checkpoint()
        save_checkpoint(path, meta=meta, arrays=arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Rebuild the module from a :meth:`save` checkpoint, restoring weights, buffers and metadata."""
        meta, arrays = load_checkpoint(path)
        return cls.from_checkpoint(meta, arrays)


def design_normal_equations(
    self: AutoregressiveMLP,
    trajectories: list[tuple[FloatArray, FloatArray]],
) -> tuple[FloatArray, FloatArray]:
    """Fold one-step input features and next-step targets into ``(G, P)``, bias column last.

    For every window on the shared grid the feature row pairs the past-output block with the control
    window shifted in by one step (the alignment ``forward`` uses). The target is the next
    standardized sample -- or, with the residual skip, that sample's delta from the window's last
    one, which is what the readout now predicts.
    """
    c = self.n_outputs
    m = self.n_controls
    y_len = self.n_y * c
    f = y_len + self.n_u * m + 1
    G = np.zeros((f, f), dtype=np.float64)
    P = np.zeros((f, c), dtype=np.float64)
    for u_raw, y_raw in trajectories:
        X, Y = build_dataset_for_trajectory(
            self.u_std.transform(np.asarray(u_raw, dtype=np.float64)),
            self.y_std.transform(np.asarray(y_raw, dtype=np.float64)),
            self.n_y,
            self.n_u,
            self.horizon,
        )
        X_1step = np.hstack([X[:, :y_len], X[:, y_len + m : y_len + (self.n_u + 1) * m]])
        X_design = np.hstack([X_1step, np.ones((X_1step.shape[0], 1))])
        G += X_design.T @ X_design
        targets = Y[:, :c]
        if self.residual:
            targets = targets - X[:, y_len - c : y_len]
        P += X_design.T @ targets
    return G, P


def install_readout(self: AutoregressiveMLP, A: FloatArray) -> None:
    """Write the ridge-fitted single layer ``A (n_outputs, f)``, bias column last."""
    layer = cast("nn.Linear", self.layers[0])
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(np.ascontiguousarray(A[:, :-1]), dtype=torch.float32))
        layer.bias.copy_(torch.as_tensor(A[:, -1], dtype=torch.float32))


class _AutoregressiveBase(nn.Module, TrainingPredictor):
    """Shared autoregressive history and Rollout machinery for waveform architectures."""

    horizon: int

    @abstractmethod
    def _predict_next(self, y_window: Tensor, u_window: Tensor) -> Tensor:
        """Predict one standardized sample from the current history windows."""
        ...

    def forward(self, y_hist: Tensor, u_hist: Tensor, u_future: Tensor) -> Tensor:
        """Roll out ``horizon`` steps while shifting shared output and control histories."""
        preds: list[Tensor] = []
        y_window, u_window = y_hist, u_hist
        for t in range(self.horizon):
            u_window = torch.cat([u_window[:, 1:], u_future[:, t : t + 1]], dim=1)
            y_next = self._predict_next(y_window, u_window)
            y_window = torch.cat([y_window[:, 1:], y_next.unsqueeze(1)], dim=1)
            preds.append(y_next)
        return torch.stack(preds, dim=1)


class AutoregressiveMLP(_AutoregressiveBase):
    """One-step MLP unrolled autoregressively over ``horizon`` steps in standardized space.

    The training side of the predictor: a batched ``forward`` over the trained Span plus the
    output and control standardizers held as float32 buffers, and the exchange-checkpoint
    ``save``/``load``. The runtime (``prime``/``step``/``rollout``/State Absorption) lives on the
    jax inference side; ``to_checkpoint``/``from_checkpoint`` are the hand-off.

    Attributes
    ----------
    layers : nn.Sequential
        The ``nn.Linear`` and activation stack in forward-pass order, all ``float32``.
    n_y, n_u : int
        Past output and past control steps in the model's history window.
    horizon : int
        Number of autoregressive steps per forward pass; BPTT depth equals it.
    n_channels : int
        Physical EEG channel count.
    n_controls : int
        Number of control channels.
    n_outputs : int
        Output width per step (equal to ``n_channels`` for the waveform predictor).
    activation : Activation
        Activation applied after every layer except the last.
    residual : bool
        Whether the output adds the window's last standardized sample, so the layers fit the
        one-step delta and the identity ``y_{k+1} = y_k`` stays free.
    dt : float
        The model's native time step, seconds; identity metadata only.
    y_center, y_scale, u_center, u_scale : Tensor
        Float32 standardizer buffers mapping raw units to the standardized training space.
    """

    y_center: Tensor
    y_scale: Tensor
    u_center: Tensor
    u_scale: Tensor
    downsample: int
    provenance: TrainingProvenance
    n_outputs: int
    geometry: StftGeometry | None
    # The Ridge-Fittable capability is attached per-instance on depth-0 models only (see
    # __init__); the declarations keep the bound methods visible to static checks while hidden-
    # layer instances still lack them at runtime, so the build-time capability check fails.
    design_normal_equations: Callable[..., tuple[FloatArray, FloatArray]]
    install_readout: Callable[..., None]

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        n_outputs: int,
        hidden_size: int,
        depth: int,
        activation: Activation = "relu",
        residual: bool = True,
        dt: float = 0.0,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
        geometry: StftGeometry | None = None,
    ) -> None:
        """Build the ``depth``-hidden-layer MLP and its standardizer buffers.

        ``n_outputs`` is the flattened per-position output width -- ``n_channels`` for the
        waveform kind, ``n_channels * n_values`` for the observable kind. ``y_std``/``u_std``
        become the module's float32 buffers; when omitted they
        default to the identity map, so a module built before the standardizers are fitted treats
        raw units as model space. The residual skip ``+ z_t`` is part of the architecture: when
        disabled the layers predict the absolute sample exactly as before.
        """
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.n_outputs = int(n_outputs)
        self.activation = activation
        self.residual = residual
        self.hidden_size = hidden_size
        self.depth = depth
        self.dt = float(dt)
        self.geometry = geometry
        # Recorded metadata the checkpoint persists and ``load`` restores; training sets them.
        self.downsample = 1
        self.provenance = TrainingProvenance()

        sizes = [n_y * self.n_outputs + n_u * n_controls, *[hidden_size] * depth, self.n_outputs]
        modules: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            modules.append(nn.Linear(n_in, n_out, dtype=torch.float32))
            if i < depth:
                modules.append(activation_module(activation))
        self.layers = nn.Sequential(*modules)

        if depth == 0:
            # The readout is the whole model, so the closed-form fit exists exactly here; a model
            # with hidden layers is nonlinear end-to-end and must fail the Ridge-Fittable
            # capability check at build time rather than carry a method that raises mid-fit.
            self.design_normal_equations = cast(
                "Callable[..., tuple[FloatArray, FloatArray]]", design_normal_equations.__get__(self)
            )
            self.install_readout = cast("Callable[..., None]", install_readout.__get__(self))

        if y_std is not None and (y_std.center.size != self.n_outputs or y_std.scale.size != self.n_outputs):
            msg = f"y_std length ({y_std.center.size}) must equal model n_outputs ({self.n_outputs})."
            raise ValueError(msg)
        if u_std is not None and (len(u_std.center) != n_controls or len(u_std.scale) != n_controls):
            msg = f"u_std length ({len(u_std.center)}) must equal model n_controls ({n_controls})."
            raise ValueError(msg)

        if y_std is None:
            y_shape = (
                (self.n_channels, self.n_outputs // self.n_channels) if geometry is not None else (self.n_outputs,)
            )
            y_std = Standardizer(center=np.zeros(y_shape), scale=np.ones(y_shape))
        u_std = u_std or Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls))
        self.register_buffer("y_center", torch.as_tensor(y_std.center, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_std.scale, dtype=torch.float32))
        self.register_buffer("u_center", torch.as_tensor(u_std.center, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(u_std.scale, dtype=torch.float32))

    def _predict_next(self, y_window: Tensor, u_window: Tensor) -> Tensor:
        """Predict one standardized sample from flattened output and control histories.

        Parameters
        ----------
        y_hist : Tensor
            Standardized past outputs/Frames of shape ``(batch, n_y, *output_shape)``.
        u_hist : Tensor
            Standardized past Control Currents of shape ``(batch, n_u, n_controls)``.
        u_future : Tensor
            Standardized future Control Currents of shape ``(batch, horizon, n_controls)``.

        Returns
        -------
        Tensor
            Standardized predicted Rollout of shape ``(batch, horizon, *output_shape)``.
        """
        features = torch.cat([y_window.flatten(1), u_window.flatten(1)], dim=1)
        delta = self.layers(features)
        if y_window.ndim > 3:  # noqa: PLR2004 -- structured Observable outputs add one axis
            delta = delta.reshape(delta.shape[0], *y_window.shape[2:])
        return y_window[:, -1, :] + delta if self.residual else delta

    @property
    def y_std(self) -> Standardizer:
        """Output standardizer reconstructed from the float32 buffers."""
        return Standardizer(
            center=self.y_center.detach().cpu().numpy(),
            scale=self.y_scale.detach().cpu().numpy(),
        )

    @property
    def u_std(self) -> Standardizer:
        """Control standardizer reconstructed from the float32 buffers."""
        return Standardizer(
            center=self.u_center.detach().cpu().numpy(),
            scale=self.u_scale.detach().cpu().numpy(),
        )

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the exchange checkpoint is written from."""
        linears = [m for m in self.layers if isinstance(m, nn.Linear)]
        meta = {
            "model_type": "mlp",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_outputs": self.n_outputs,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "residual": int(self.residual),
            "dt": self.dt,
            "downsample": self.downsample,
            "n_layers": len(linears),
            **self.provenance.meta,
        }
        if self.geometry is not None:
            meta["geometry"] = self.geometry.model_dump()
        arrays = layer_arrays(
            "layer", [to_numpy(lin.weight) for lin in linears], [to_numpy(lin.bias) for lin in linears]
        )
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        return meta, arrays

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the module from a ``(meta, arrays)`` pair, restoring weights, buffers and metadata."""
        require_model_type(meta, "mlp")
        require_activation(meta)
        geometry = StftGeometry.model_validate(meta["geometry"]) if "geometry" in meta else None
        model = cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            n_outputs=int(meta["n_outputs"]),
            hidden_size=int(meta["hidden_size"]),
            depth=int(meta["depth"]),
            activation=meta["activation"],
            residual=bool(meta["residual"]),
            dt=float(meta["dt"]),
            y_std=Standardizer.from_arrays(arrays, "y"),
            u_std=Standardizer.from_arrays(arrays, "u"),
            geometry=geometry,
        )
        model.downsample = int(meta["downsample"])
        model.provenance = TrainingProvenance.from_meta(meta)
        linears = [m for m in model.layers if isinstance(m, nn.Linear)]
        weights, biases = layers_from_arrays(arrays, "layer", len(linears))
        with torch.no_grad():
            for lin, weight, bias in zip(linears, weights, biases, strict=True):
                lin.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
        return model


class AutoregressiveCNN(_AutoregressiveBase):
    """Causal waveform CNN with a dense Control Current history head."""

    n_y: int
    n_u: int
    horizon: int
    n_channels: int
    n_controls: int
    n_outputs: int
    hidden_size: int
    depth: int
    kernel_size: int
    activation: Activation
    residual: bool
    dt: float
    downsample: int
    provenance: TrainingProvenance
    y_center: Tensor
    y_scale: Tensor
    u_center: Tensor
    u_scale: Tensor

    def __init__(  # noqa: PLR0913 -- architecture and standardizer record
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        n_outputs: int | None = None,
        hidden_size: int,
        depth: int,
        kernel_size: int = 3,
        activation: Activation = "relu",
        residual: bool = True,
        dt: float = 0.0,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
    ) -> None:
        """Build a causal stride-one temporal CNN for waveform Observables."""
        super().__init__()
        if depth < 1:
            msg = "CNN depth must be at least 1."
            raise ValueError(msg)
        if kernel_size < 1:
            msg = "CNN kernel_size must be at least 1."
            raise ValueError(msg)
        self.n_y, self.n_u, self.horizon = n_y, n_u, horizon
        if n_outputs is not None and n_outputs != n_channels:
            msg = "waveform CNN n_outputs must equal n_channels"
            raise ValueError(msg)
        self.n_channels, self.n_controls = n_channels, n_controls
        self.n_outputs, self.hidden_size, self.depth = n_channels, hidden_size, depth
        self.kernel_size, self.activation, self.residual, self.dt = kernel_size, activation, residual, float(dt)
        self.downsample = 1
        self.provenance = TrainingProvenance()
        convs: list[nn.Module] = []
        for i in range(depth):
            convs.append(
                nn.Conv1d(n_channels if i == 0 else hidden_size, hidden_size, kernel_size, dtype=torch.float32)
            )
            if i < depth - 1:
                convs.append(activation_module(activation))
        self.convs = nn.Sequential(*convs)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_u * n_controls, hidden_size, dtype=torch.float32),
            activation_module(activation),
            nn.Linear(hidden_size, n_channels, dtype=torch.float32),
        )
        y_std = y_std or Standardizer(center=np.zeros(n_channels), scale=np.ones(n_channels))
        u_std = u_std or Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls))
        self.register_buffer("y_center", torch.as_tensor(y_std.center, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_std.scale, dtype=torch.float32))
        self.register_buffer("u_center", torch.as_tensor(u_std.center, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(u_std.scale, dtype=torch.float32))

    def _one_step(self, y_window: Tensor, u_window: Tensor) -> Tensor:
        """Predict one standardized waveform sample from causal history windows."""
        z = y_window.transpose(1, 2)
        for layer in self.convs:
            if isinstance(layer, nn.Conv1d):
                z = functional.pad(z, (layer.kernel_size[0] - 1, 0))
            z = layer(z)
        features = torch.cat([z[:, :, -1], u_window.flatten(1)], dim=1)
        delta = self.head(features)
        return y_window[:, -1] + delta if self.residual else delta

    def _predict_next(self, y_window: Tensor, u_window: Tensor) -> Tensor:
        """Predict one standardized waveform sample through causal convolution and the head."""
        if y_window.ndim != 3:  # noqa: PLR2004 -- the CNN input has batch, time, channel axes
            msg = "AutoregressiveCNN expects waveform histories shaped (batch, time, channels)."
            raise ValueError(msg)
        return self._one_step(y_window, u_window)

    @property
    def y_std(self) -> Standardizer:
        """Return the output standardizer held by this module."""
        return Standardizer(center=to_numpy(self.y_center), scale=to_numpy(self.y_scale))

    @property
    def u_std(self) -> Standardizer:
        """Return the Control Current standardizer held by this module."""
        return Standardizer(center=to_numpy(self.u_center), scale=to_numpy(self.u_scale))

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the portable CNN checkpoint."""
        convs = [m for m in self.convs if isinstance(m, nn.Conv1d)]
        heads = [m for m in self.head if isinstance(m, nn.Linear)]
        meta = {
            "model_type": "cnn",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_outputs": self.n_outputs,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "residual": int(self.residual),
            "dt": self.dt,
            "downsample": self.downsample,
            "n_convs": len(convs),
            "n_head_layers": len(heads),
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {}
        for prefix, layers in (("conv", convs), ("head", heads)):
            arrays.update(
                layer_arrays(
                    prefix,
                    [to_numpy(m.weight) for m in layers],
                    [to_numpy(cast("Tensor", m.bias)) for m in layers],
                )
            )
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        return meta, arrays

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild a CNN from its portable checkpoint."""
        require_model_type(meta, "cnn")
        require_activation(meta)
        model = cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            hidden_size=int(meta["hidden_size"]),
            depth=int(meta["depth"]),
            kernel_size=int(meta["kernel_size"]),
            activation=meta["activation"],
            residual=bool(meta["residual"]),
            dt=float(meta["dt"]),
            y_std=Standardizer.from_arrays(arrays, "y"),
            u_std=Standardizer.from_arrays(arrays, "u"),
        )
        model.downsample = int(meta["downsample"])
        model.provenance = TrainingProvenance.from_meta(meta)
        convs = [m for m in model.convs if isinstance(m, nn.Conv1d)]
        heads = [m for m in model.head if isinstance(m, nn.Linear)]
        for layers, prefix in ((convs, "conv"), (heads, "head")):
            weights, biases = layers_from_arrays(arrays, prefix, len(layers))
            with torch.no_grad():
                for layer, weight, bias in zip(layers, weights, biases, strict=True):
                    layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                    cast("Tensor", layer.bias).copy_(torch.as_tensor(bias, dtype=torch.float32))
        return model
