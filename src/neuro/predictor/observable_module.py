from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self

import torch
from torch import nn

from neuro.observable import ObservableArtifact, control_means
from neuro.predictor.module import activation_module, to_numpy

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from neuro.config import ObservableGeometry
    from neuro.predictor.artifact import Activation
    from neuro.transforms import Standardizer


def mlp_stack(sizes: list[int], activation: Activation) -> nn.Sequential:
    """Stack ``nn.Linear`` layers over ``sizes`` with the activation after all but the last."""
    modules: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
        modules.append(nn.Linear(n_in, n_out, dtype=torch.float32))
        if i < len(sizes) - 2:
            modules.append(activation_module(activation))
    return nn.Sequential(*modules)


class ObservableMLP(nn.Module):
    """Lift the history state, recurse one shared transition per Frame, read out standardized log power.

    BPTT depth is the Frame count, not the Control Horizon, so the backward pass costs 2--3 nodes at
    the deployed geometry rather than 75.

    Attributes
    ----------
    lift : nn.Sequential
        ``E``: the history state ``[y_window | u_window]`` -> ``z_0``.
    transition : nn.Sequential
        ``f_theta(z_m, u_bar_m)``, one map shared across every Frame.
    readout : nn.Linear
        The affine ``C z + c`` emitting standardized log-Observable; no exponential and no
        positivity constraint, so the floor stays on measured power where it belongs.
    aggregate : Tensor
        The fixed ``(n_frames, horizon)`` operator averaging controls over each Frame's support.
    """

    def __init__(  # noqa: PLR0913
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
    ) -> None:
        """Build the lift, the shared Frame transition and the log readout."""
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.geometry = geometry
        self.fs = fs
        self.activation = activation
        self.n_values = geometry.n_values(fs)

        n_hist = n_y * n_channels + n_u * n_controls
        self.lift = mlp_stack([n_hist, *[lift_hidden] * lift_depth, z_dim], activation)
        self.transition = mlp_stack([z_dim + n_controls, *[transition_hidden] * transition_depth, z_dim], activation)
        self.readout = nn.Linear(z_dim, n_channels * self.n_values, dtype=torch.float32)
        self.register_buffer(
            "aggregate",
            torch.as_tensor(control_means(geometry, horizon, fs), dtype=torch.float32),
        )

    @property
    def n_frames(self) -> int:
        """Number of Frames the recursion emits over the trained horizon."""
        return self.geometry.n_frames(self.horizon, self.fs)

    def forward(self, x: Tensor) -> Tensor:
        """Forecast standardized log-Observable: ``(B, n_hist + horizon * m) -> (B, n_frames, C * F)``.

        ``x`` is the incumbent history-plus-future-control row, so the two paths share one dataset
        builder. Averaging the *standardized* controls equals standardizing their mean, because both
        maps are affine.
        """
        n_hist = self.n_y * self.n_channels + self.n_u * self.n_controls
        u_future = x[:, n_hist:].reshape(x.shape[0], self.horizon, self.n_controls)
        u_bar = torch.einsum("mt,btc->bmc", self.aggregate, u_future)

        z = self.lift(x[:, :n_hist])
        frames = []
        for m in range(u_bar.shape[1]):
            z = self.transition(torch.cat([z, u_bar[:, m]], dim=1))
            frames.append(self.readout(z))
        return torch.stack(frames, dim=1)

    def to_artifact(
        self,
        dt: float,
        downsample: int,
        y_std: Standardizer,
        u_std: Standardizer,
        l_std: Standardizer,
    ) -> ObservableArtifact:
        """Freeze the trained weights and standardizers into a framework-free artifact."""

        def layers(block: nn.Sequential) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
            return tuple((to_numpy(m.weight), to_numpy(m.bias)) for m in block if isinstance(m, nn.Linear))

        return ObservableArtifact(
            lift=layers(self.lift),
            transition=layers(self.transition),
            readout=(to_numpy(self.readout.weight), to_numpy(self.readout.bias)),
            activation=self.activation,
            n_y=self.n_y,
            n_u=self.n_u,
            horizon=self.horizon,
            n_channels=self.n_channels,
            n_controls=self.n_controls,
            dt=dt,
            downsample=downsample,
            geometry=self.geometry,
            y_std=y_std,
            u_std=u_std,
            l_std=l_std,
        )

    @classmethod
    def from_artifact(cls, art: ObservableArtifact) -> Self:
        """Rebuild the module carrying the artifact's weights."""
        model = cls(
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            geometry=art.geometry,
            fs=art.fs,
            z_dim=art.z_dim,
            lift_hidden=art.lift[0][0].shape[0],
            lift_depth=len(art.lift) - 1,
            transition_hidden=art.transition[0][0].shape[0],
            transition_depth=len(art.transition) - 1,
            activation=art.activation,
        )
        with torch.no_grad():
            for block, weights in ((model.lift, art.lift), (model.transition, art.transition)):
                linears = (m for m in block if isinstance(m, nn.Linear))
                for lin, (w, b) in zip(linears, weights, strict=True):
                    lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                    lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
            model.readout.weight.copy_(torch.as_tensor(art.readout[0], dtype=torch.float32))
            model.readout.bias.copy_(torch.as_tensor(art.readout[1], dtype=torch.float32))
        return model
