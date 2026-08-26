from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
from torch import nn

from neuro.observable import control_means, geometry_from_meta, geometry_meta, log_observable
from neuro.predictor.checkpoint import layer_arrays, layers_from_arrays, require_activation, require_model_type
from neuro.predictor.data import build_dataset_for_trajectory, extract_future_windows
from neuro.predictor.module import TrainingPredictor, activation_module, to_numpy
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.config import ObservableGeometry
    from neuro.types import Activation, FloatArray


def mlp_stack(sizes: list[int], activation: Activation) -> nn.Sequential:
    """Stack ``nn.Linear`` layers over ``sizes`` with the activation after all but the last."""
    modules: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
        modules.append(nn.Linear(n_in, n_out, dtype=torch.float32))
        if i < len(sizes) - 2:
            modules.append(activation_module(activation))
    return nn.Sequential(*modules)


class StepwiseObservableMLP(nn.Module, TrainingPredictor):
    """One-Frame-per-step Observable predictor: the training side of the observable model.

    The standardized batched ``forward`` unrolls the lift, the shared Frame transition and the
    log readout for training (BPTT depth is the Frame count, not the Control Horizon), plus the
    standardizer buffers and the exchange-checkpoint ``save``/``load``. The runtime recursion and
    State Absorption live on the jax inference side; ``to_checkpoint``/``from_checkpoint`` are the
    hand-off. The Ridge capability stays here: the shared readout is linear in the lifted Frame
    states, so the depth-0 module is still Ridge-Fittable.

    Attributes
    ----------
    lift : nn.Sequential
        ``E``: the history state ``[y_window | u_window]`` -> ``z_0``.
    transition : nn.Sequential
        ``f_theta(z_m, u_bar_m)``, one map shared across every Frame.
    readout : nn.Linear
        The affine ``C z + c`` emitting standardized log-Observable.
    aggregate : Tensor
        The fixed ``(n_frames, horizon)`` operator averaging controls over each Frame's support.
    y_center, y_scale, u_center, u_scale, l_center, l_scale : Tensor
        Float32 standardizer buffers mapping raw units to the standardized training space.
    """

    y_center: Tensor
    y_scale: Tensor
    u_center: Tensor
    u_scale: Tensor
    l_center: Tensor
    l_scale: Tensor
    downsample: int
    provenance: TrainingProvenance

    def __init__(  # noqa: PLR0913 -- geometry, state/lift/transition shapes and standardizers are the constructor surface
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        geometry: ObservableGeometry,
        fs: float,
        z_dim: int,
        lift_hidden: int,
        lift_depth: int,
        transition_hidden: int,
        transition_depth: int,
        activation: Activation = "softplus",
        residual: bool = True,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
        l_std: Standardizer | None = None,
    ) -> None:
        """Build the lift, the shared Frame transition, the log readout and the standardizer buffers.

        ``y_std``/``u_std``/``l_std`` become the module's float32 buffers; when omitted they
        default to the identity map, so a module built before the standardizers are fitted treats
        raw units as model space. With the residual skip the readout emits per-Frame deltas that
        the recursion accumulates.
        """
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.geometry = geometry
        self.fs = fs
        self.z_dim = z_dim
        self.lift_hidden = lift_hidden
        self.lift_depth = lift_depth
        self.transition_hidden = transition_hidden
        self.transition_depth = transition_depth
        self.activation = activation
        self.residual = residual
        self.n_values = geometry.n_values(fs)
        self.n_outputs = n_channels * self.n_values
        # Recorded metadata the checkpoint persists and ``load`` restores; training sets them.
        self.downsample = 1
        self.provenance = TrainingProvenance()

        n_hist = n_y * n_channels + n_u * n_controls
        self.lift = mlp_stack([n_hist, *[lift_hidden] * lift_depth, z_dim], activation)
        self.transition = mlp_stack([z_dim + n_controls, *[transition_hidden] * transition_depth, z_dim], activation)
        self.readout = nn.Linear(z_dim, self.n_outputs, dtype=torch.float32)
        self.register_buffer(
            "aggregate",
            torch.as_tensor(control_means(geometry, horizon, fs), dtype=torch.float32),
        )

        y_std = y_std or Standardizer(center=np.zeros(n_channels), scale=np.ones(n_channels))
        u_std = u_std or Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls))
        l_std = l_std or Standardizer(center=np.zeros(self.n_outputs), scale=np.ones(self.n_outputs))
        self.register_buffer("y_center", torch.as_tensor(y_std.center, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_std.scale, dtype=torch.float32))
        self.register_buffer("u_center", torch.as_tensor(u_std.center, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(u_std.scale, dtype=torch.float32))
        self.register_buffer("l_center", torch.as_tensor(l_std.center, dtype=torch.float32))
        self.register_buffer("l_scale", torch.as_tensor(l_std.scale, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Forecast standardized log-Observable: ``(B, n_hist + horizon * m) -> (B, n_frames, C * F)``.

        ``x`` is the incumbent history-plus-future-control row, so the two paths share one dataset
        builder. Averaging the *standardized* controls equals standardizing their mean, because both
        maps are affine. With the residual the readout emits each Frame's delta and the recursion
        accumulates it, exactly as the inference side carries it.
        """
        batch = x.shape[0]
        u_future = x[:, self._n_hist :].reshape(batch, self.horizon, self.n_controls)
        u_bar = torch.einsum("mt,btc->bmc", self.aggregate, u_future)

        z = self.lift(x[:, : self._n_hist])
        frames = []
        if self.residual:
            carry = torch.zeros(batch, self.n_outputs, dtype=torch.float32, device=x.device)
        for m in range(u_bar.shape[1]):
            z = self.transition(torch.cat([z, u_bar[:, m]], dim=1))
            if self.residual:
                carry = carry + self.readout(z)
                frames.append(carry)
            else:
                frames.append(self.readout(z))
        return torch.stack(frames, dim=1)

    @property
    def dt(self) -> float:
        """The model's native time step, seconds: the sample period the Frame grid resolves at."""
        return 1.0 / self.fs

    def n_frames(self, horizon: int | None = None) -> int:
        """Frames the recursion emits over ``horizon`` samples (default: the trained horizon)."""
        return self.geometry.n_frames(self.horizon if horizon is None else horizon, self.fs)

    @property
    def _n_hist(self) -> int:
        """Register width: the standardized EEG window plus the raw control window."""
        return self.n_y * self.n_channels + self.n_u * self.n_controls

    @property
    def y_std(self) -> Standardizer:
        """Channel standardizer reconstructed from the float32 buffers."""
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

    @property
    def l_std(self) -> Standardizer:
        """Log-Observable standardizer reconstructed from the float32 buffers."""
        return Standardizer(
            center=self.l_center.detach().cpu().numpy(),
            scale=self.l_scale.detach().cpu().numpy(),
        )

    def design_normal_equations(
        self, trajectories: list[tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """Harvest the per-Frame lifted states ``z_m`` and their Frame targets into ``(G, P)``.

        For every window on the shared grid the readout pairs with its Frame's standardized
        log-Observable target -- or, with the residual skip, that Frame's delta from the previous
        Frame's target, zero before the first Frame. The bias column (last) is the constant-1
        feature.
        """
        f = self.z_dim + 1
        c = self.n_outputs
        n_hist = self._n_hist
        n_frames = self.n_frames()
        G = np.zeros((f, f), dtype=np.float64)
        P = np.zeros((f, c), dtype=np.float64)

        for u_raw, y_raw in trajectories:
            u_arr = np.asarray(u_raw, dtype=np.float64)
            y_arr = np.asarray(y_raw, dtype=np.float64)
            X, _ = build_dataset_for_trajectory(
                self.u_std.transform(u_arr), self.y_std.transform(y_arr), self.n_y, self.n_u, self.horizon
            )
            y_fut = extract_future_windows(y_arr, self.n_y, self.n_u, self.horizon)
            targets = self.l_std.transform(
                log_observable(y_fut.reshape(-1, self.horizon, self.n_channels), self.geometry, self.fs).reshape(
                    -1, n_frames, c
                )
            )
            n_samples = X.shape[0]
            H_mat = np.empty((n_samples * n_frames, f), dtype=np.float64)
            H_mat[:, -1] = 1.0
            z = self.lift(torch.as_tensor(X[:, :n_hist], dtype=torch.float32))
            u_bar = torch.einsum(
                "mt,btc->bmc",
                self.aggregate,
                torch.as_tensor(X[:, n_hist:].reshape(n_samples, self.horizon, self.n_controls), dtype=torch.float32),
            )
            for m in range(n_frames):
                z = self.transition(torch.cat([z, u_bar[:, m]], dim=1))
                H_mat[m * n_samples : (m + 1) * n_samples, : self.z_dim] = to_numpy(z)
            # Frame-major rows, paired with the frame-major targets below. With the residual skip
            # the readout fits Frame-to-Frame deltas rather than absolute levels.
            T_mat = targets.transpose(1, 0, 2).reshape(-1, c)
            if self.residual:
                T_mat = T_mat.copy()
                T_mat[n_samples:] -= T_mat[:-n_samples]
            G += H_mat.T @ H_mat
            P += H_mat.T @ T_mat
        return G, P

    def install_readout(self, A: FloatArray) -> None:
        """Write the ridge-fitted shared readout ``A (n_outputs, z_dim + 1)``, bias column last."""
        with torch.no_grad():
            self.readout.weight.copy_(torch.as_tensor(np.ascontiguousarray(A[:, :-1]), dtype=torch.float32))
            self.readout.bias.copy_(torch.as_tensor(A[:, -1], dtype=torch.float32))

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Build the ``(meta, arrays)`` pair the exchange checkpoint is written from."""
        linears = [m for m in self.lift if isinstance(m, nn.Linear)]
        transitions = [m for m in self.transition if isinstance(m, nn.Linear)]
        meta = {
            "model_type": "observable",
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "dt": self.dt,
            "downsample": self.downsample,
            "z_dim": self.z_dim,
            "lift_hidden": self.lift_hidden,
            "lift_depth": self.lift_depth,
            "transition_hidden": self.transition_hidden,
            "transition_depth": self.transition_depth,
            "n_lift_layers": len(linears),
            "n_transition_layers": len(transitions),
            "residual": int(self.residual),
            "geometry": geometry_meta(self.geometry),
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {}
        for prefix, blocks in (("lift", linears), ("transition", transitions)):
            arrays.update(
                layer_arrays(prefix, [to_numpy(lin.weight) for lin in blocks], [to_numpy(lin.bias) for lin in blocks])
            )
        arrays["readout.weight"] = to_numpy(self.readout.weight)
        arrays["readout.bias"] = to_numpy(self.readout.bias)
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        arrays.update(self.l_std.arrays("l"))
        return meta, arrays

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild the module from a ``(meta, arrays)`` pair, restoring weights, buffers and metadata."""
        require_model_type(meta, "observable")
        require_activation(meta)
        model = cls(
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            geometry=geometry_from_meta(meta["geometry"]),
            fs=1.0 / float(meta["dt"]),
            z_dim=int(meta["z_dim"]),
            lift_hidden=int(meta["lift_hidden"]),
            lift_depth=int(meta["lift_depth"]),
            transition_hidden=int(meta["transition_hidden"]),
            transition_depth=int(meta["transition_depth"]),
            activation=meta["activation"],
            residual=bool(meta.get("residual", False)),
            y_std=Standardizer.from_arrays(arrays, "y"),
            u_std=Standardizer.from_arrays(arrays, "u"),
            l_std=Standardizer.from_arrays(arrays, "l"),
        )
        model.downsample = int(meta["downsample"])
        model.provenance = TrainingProvenance.from_meta(meta)
        with torch.no_grad():
            for block, prefix in ((model.lift, "lift"), (model.transition, "transition")):
                blocks = [m for m in block if isinstance(m, nn.Linear)]
                weights, biases = layers_from_arrays(arrays, prefix, len(blocks))
                for lin, weight, bias in zip(blocks, weights, biases, strict=True):
                    lin.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                    lin.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
            model.readout.weight.copy_(torch.as_tensor(np.asarray(arrays["readout.weight"]), dtype=torch.float32))
            model.readout.bias.copy_(torch.as_tensor(np.asarray(arrays["readout.bias"]), dtype=torch.float32))
        return model
