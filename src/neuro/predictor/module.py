from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import torch
from torch import nn

from neuro.predictor.artifact import MLPArtifact

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.predictor.artifact import Activation
    from neuro.transforms import Pipeline
    from neuro.types import FloatArray


def _activate(z: Tensor, activation: Activation) -> Tensor:
    """Apply the named activation elementwise (matching :func:`neuro.predictor.artifact._activate`)."""
    if activation == "relu":
        return torch.relu(z)
    if activation == "tanh":
        return torch.tanh(z)
    if activation == "softplus":
        return torch.nn.functional.softplus(z)
    msg = f"Unsupported activation: {activation}"
    raise ValueError(msg)


def _buffer(a: FloatArray | None) -> Tensor | None:
    """Copy an optional NumPy array into a float64 tensor."""
    return None if a is None else torch.tensor(np.asarray(a, dtype=np.float64))


def _to_numpy(t: Tensor) -> FloatArray:
    """Detach a parameter into an owned float64 NumPy array."""
    return t.detach().cpu().numpy().astype(np.float64, copy=True)


class AutoregressiveMLP(nn.Module):
    """One-step MLP unrolled autoregressively over ``horizon`` steps, entirely in model space.

    Both the EEG and the control blocks of the input are expected already transformed
    (as :func:`neuro.predictor.data.transform_features` produces them); the module itself
    knows nothing about raw units. The PCA decode arrays ride along as buffers so the loss
    and the training loop can reach channel space without threading them through signatures.

    Attributes
    ----------
    layers : nn.ModuleList
        The ``nn.Linear`` stack in forward-pass order, all ``float64``.
    n_y, n_u : int
        Past EEG and past control steps in the model's history window.
    horizon : int
        Number of autoregressive steps per forward pass; BPTT depth equals it.
    n_channels : int
        Model-space channel count: the latent dimension ``k`` under PCA, else the raw EEG count.
    n_controls : int
        Number of control channels.
    activation : Activation
        Activation applied after every layer except the last.
    decode_basis : Tensor | None
        PCA basis buffer ``(k, n_eeg_channels)``, or ``None`` without a projection.
    decode_mean : Tensor | None
        PCA per-channel mean buffer ``(n_eeg_channels,)``, paired with ``decode_basis``.
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
        decode_basis: FloatArray | None = None,
        decode_mean: FloatArray | None = None,
    ) -> None:
        """Build the ``depth``-hidden-layer MLP and register the optional PCA decode buffers."""
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.activation = activation

        sizes = [n_y * n_channels + n_u * n_controls, *[hidden_size] * depth, n_channels]
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out, dtype=torch.float64) for n_in, n_out in itertools.pairwise(sizes)
        )
        self.register_buffer("decode_basis", _buffer(decode_basis))
        self.register_buffer("decode_mean", _buffer(decode_mean))

    def _forward_1step(self, x: Tensor) -> Tensor:
        """One-step MLP forward on the model-space input ``[y_window | u_window]``, ``(B, in) -> (B, k)``."""
        for layer in self.layers[:-1]:
            x = _activate(layer(x), self.activation)
        return self.layers[-1](x)

    def forward(self, x: Tensor) -> Tensor:
        """Roll out ``horizon`` steps: ``(B, n_y*k + (n_u + horizon)*n_controls) -> (B, horizon*k)``."""
        batch = x.shape[0]
        n_z = self.n_y * self.n_channels
        n_u_past = self.n_u * self.n_controls

        y_window = x[:, :n_z].reshape(batch, self.n_y, self.n_channels)
        u_window = x[:, n_z : n_z + n_u_past].reshape(batch, self.n_u, self.n_controls)
        u_future = x[:, n_z + n_u_past :].reshape(batch, self.horizon, self.n_controls)

        preds = []
        for t in range(self.horizon):
            # The control window shifts *before* the MLP call, so the newest control is already
            # in the window when y_next is predicted.
            u_window = torch.cat([u_window[:, 1:], u_future[:, t : t + 1]], dim=1)
            mlp_in = torch.cat([y_window.reshape(batch, -1), u_window.reshape(batch, -1)], dim=1)
            y_next = self._forward_1step(mlp_in)
            y_window = torch.cat([y_window[:, 1:], y_next[:, None, :]], dim=1)
            preds.append(y_next)

        return torch.cat(preds, dim=1)

    def to_artifact(self, dt: float, downsample: int, y_pipeline: Pipeline, u_pipeline: Pipeline) -> MLPArtifact:
        """Freeze the trained weights and the given transforms into a framework-free artifact."""
        # ModuleList iteration is typed as bare Module, so the Linear parameters need a cast.
        linears = (cast("nn.Linear", layer) for layer in self.layers)
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
            y_pipeline=y_pipeline,
            u_pipeline=u_pipeline,
        )

    @classmethod
    def from_artifact(cls, art: MLPArtifact) -> Self:
        """Rebuild the module carrying the artifact's weights and PCA decode arrays."""
        pca = art.y_pipeline.pca
        model = cls(
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            hidden_size=art.layers[0][0].shape[0],
            depth=len(art.layers) - 1,
            activation=art.activation,
            decode_basis=None if pca is None else pca.basis,
            decode_mean=None if pca is None else pca.mean,
        )
        with torch.no_grad():
            for layer, (w, b) in zip(model.layers, art.layers, strict=True):
                lin = cast("nn.Linear", layer)
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float64))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float64))
        return model
