from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch import nn

from neuro.predictor.artifact import MLPArtifact

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.predictor.artifact import Activation
    from neuro.transforms import Standardizer
    from neuro.types import FloatArray


def _activation_module(activation: Activation) -> nn.Module:
    """Return the torch activation module matching the named activation."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    return nn.Softplus()


def _to_numpy(t: Tensor) -> FloatArray:
    """Detach a parameter into an owned float64 NumPy array."""
    return t.detach().cpu().numpy().astype(np.float64, copy=True)


class AutoregressiveMLP(nn.Module):
    """One-step MLP unrolled autoregressively over ``horizon`` steps in standardized channel space.

    Attributes
    ----------
    layers : nn.Sequential
        The ``nn.Linear`` and activation stack in forward-pass order, all ``float32``.
    n_y, n_u : int
        Past EEG and past control steps in the model's history window.
    horizon : int
        Number of autoregressive steps per forward pass; BPTT depth equals it.
    n_channels : int
        Physical EEG channel count.
    n_controls : int
        Number of control channels.
    activation : Activation
        Activation applied after every layer except the last.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_y: int,
        n_u: int,
        horizon: int,
        n_channels: int,
        n_controls: int,
        hidden_size: int,
        depth: int,
        activation: Activation = "relu",
    ) -> None:
        """Build the ``depth``-hidden-layer MLP."""
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.activation = activation

        sizes = [n_y * n_channels + n_u * n_controls, *[hidden_size] * depth, n_channels]
        modules: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            modules.append(nn.Linear(n_in, n_out, dtype=torch.float32))
            if i < depth:
                modules.append(_activation_module(activation))
        self.layers = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Roll out ``horizon`` steps: ``(B, n_y*C + (n_u + horizon)*m) -> (B, horizon*C)``."""
        batch = x.shape[0]
        n_z = self.n_y * self.n_channels
        n_u_past = self.n_u * self.n_controls

        y_window = x[:, :n_z].reshape(batch, self.n_y, self.n_channels)
        u_window = x[:, n_z : n_z + n_u_past].reshape(batch, self.n_u, self.n_controls)
        u_future = x[:, n_z + n_u_past :].reshape(batch, self.horizon, self.n_controls)

        preds = []
        for t in range(self.horizon):
            # The feature row's u-window ends one step *behind* its y-window (see the slicing in
            # build_dataset_for_trajectory), so the control shifts in before the MLP call to bring
            # the two level. MLPArtifact.rollout shifts *after* because a prime() state is already
            # level -- opposite order, same rule: both windows end at t when y_{t+1} is predicted.
            u_window = torch.cat([u_window[:, 1:], u_future[:, t : t + 1]], dim=1)
            mlp_in = torch.cat([y_window.reshape(batch, -1), u_window.reshape(batch, -1)], dim=1)
            y_next = self.layers(mlp_in)
            y_window = torch.cat([y_window[:, 1:], y_next[:, None, :]], dim=1)
            preds.append(y_next)

        return torch.cat(preds, dim=1)

    def to_artifact(self, dt: float, downsample: int, y_std: Standardizer, u_std: Standardizer) -> MLPArtifact:
        """Freeze the trained weights and standardizers into a framework-free artifact."""
        linears = (m for m in self.layers if isinstance(m, nn.Linear))
        layers = tuple((_to_numpy(lin.weight), _to_numpy(lin.bias)) for lin in linears)
        return MLPArtifact(
            layers=layers,
            activation=self.activation,
            n_y=self.n_y,
            n_u=self.n_u,
            horizon=self.horizon,
            n_channels=self.n_channels,
            n_controls=self.n_controls,
            dt=dt,
            downsample=downsample,
            y_std=y_std,
            u_std=u_std,
        )

    @classmethod
    def from_artifact(cls, art: MLPArtifact) -> Self:
        """Rebuild the module carrying the artifact's weights."""
        model = cls(
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            hidden_size=art.layers[0][0].shape[0],
            depth=len(art.layers) - 1,
            activation=art.activation,
        )
        linears = (m for m in model.layers if isinstance(m, nn.Linear))
        with torch.no_grad():
            for lin, (w, b) in zip(linears, art.layers, strict=True):
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        return model
