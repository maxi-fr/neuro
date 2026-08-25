from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch import nn

from neuro.predictor.artifact import MLPArtifact
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray


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


class AutoregressiveMLP(nn.Module):
    """One-step MLP unrolled autoregressively over ``horizon`` steps in standardized channel space.

    Alongside the training ``forward`` the module carries the full raw-units Predictor runtime --
    ``prime``, ``step``, ``rollout``, ``absorb``, ``is_ready``, ``initial_state`` and their
    batched forms -- with the channel and control standardizers held as float32 buffers, so
    callers exchange raw units only. The opaque state is a shift register holding the
    standardized EEG window followed by the raw control window.

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
    dt : float
        The model's native time step, seconds; identity metadata only.
    y_center, y_scale, u_center, u_scale : Tensor
        Float32 standardizer buffers mapping raw units to the standardized training space.
    """

    y_center: Tensor
    y_scale: Tensor
    u_center: Tensor
    u_scale: Tensor

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
        dt: float = 0.0,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
    ) -> None:
        """Build the ``depth``-hidden-layer MLP and its standardizer buffers.

        ``y_std``/``u_std`` become the module's float32 buffers; when omitted they default to the
        identity map, so a module built before the standardizers are fitted treats raw units as
        model space.
        """
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.activation = activation
        self.dt = float(dt)

        sizes = [n_y * n_channels + n_u * n_controls, *[hidden_size] * depth, n_channels]
        modules: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            modules.append(nn.Linear(n_in, n_out, dtype=torch.float32))
            if i < depth:
                modules.append(activation_module(activation))
        self.layers = nn.Sequential(*modules)

        y_std = y_std or Standardizer(center=np.zeros(n_channels), scale=np.ones(n_channels))
        u_std = u_std or Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls))
        self.register_buffer("y_center", torch.as_tensor(y_std.center, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_std.scale, dtype=torch.float32))
        self.register_buffer("u_center", torch.as_tensor(u_std.center, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(u_std.scale, dtype=torch.float32))

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
    def n_outputs(self) -> int:
        """Output width per position: one EEG sample across all channels."""
        return self.n_channels

    @property
    def priming_steps(self) -> int:
        """Minimum number of history steps required to prime the state."""
        return max(self.n_y, self.n_u)

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_channels)`` into standardized channel space."""
        return self.y_std.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_channels)`` from standardized space."""
        return self.y_std.inverse_transform(np.asarray(z, dtype=np.float64))

    def _forward_1step(self, y_window: FloatArray, u_window: FloatArray) -> FloatArray:
        """One-step MLP forward on standardized windows -> next standardized sample(s).

        ``y_window`` and ``u_window`` are ``(..., n_y, n_channels)`` and ``(..., n_u, n_controls)``
        with a leading batch dim when present; returns ``(..., n_channels)``.
        """
        batch = y_window.shape[:-2]
        x = np.concatenate(
            [
                y_window.reshape(*batch, self.n_y * self.n_channels),
                u_window.reshape(*batch, self.n_u * self.n_controls),
            ],
            axis=-1,
        )
        with torch.no_grad():
            z = self.layers(torch.as_tensor(x, dtype=torch.float32))
        return to_numpy(z)

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into an initial state: standardized EEG window + raw control window.

        ``y_hist (k, n_channels)`` and ``u_hist (k, n_controls)`` both end at the same step
        ``t - 1``, so :meth:`rollout` predicts from ``t`` onwards.
        """
        y_arr = np.asarray(y_hist, dtype=np.float64)
        u_arr = np.asarray(u_hist, dtype=np.float64)
        z_past = self.encode(y_arr)[-self.n_y :].reshape(-1)
        u_past = u_arr[-self.n_u :].reshape(-1)
        return np.concatenate([z_past, u_past])

    def step(self, state: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Advance one position (one sample): apply raw control ``u`` -> ``(state', output)``.

        ``u`` is the control applied at this prediction step, so it enters the control window only
        after the step's prediction used the previous one -- both windows end at ``t`` when
        ``y_{t+1}`` is predicted, the training alignment.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window_raw = state_arr[n_z:].reshape(self.n_u, self.n_controls)

        z_next = self._forward_1step(y_window, self.u_std.transform(u_window_raw))
        y_window = np.concatenate([y_window[1:], z_next[None, :]], axis=0)
        u_window = np.concatenate([u_window_raw[1:], u_arr[None, :]], axis=0)
        state_next = np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        return state_next, self.decode(z_next)

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from ``state`` under raw ``u_future`` -> raw ``(steps, n_channels)``.

        ``u_future[t]`` is the control applied at prediction step ``t``, so it first enters the
        window for step ``t + 1`` and the last entry is never consumed; the length is not bounded
        by ``horizon``. Equivalent to a loop of :meth:`step`.
        """
        u_arr = np.asarray(u_future, dtype=np.float64)
        preds = np.empty((len(u_arr), self.n_channels), dtype=np.float64)
        for t in range(len(u_arr)):
            state, y = self.step(state, u_arr[t])
            preds[t] = y
        return preds

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
        y_arr = np.asarray(y_hists, dtype=np.float64)
        u_arr = np.asarray(u_hists, dtype=np.float64)
        n_batch = y_arr.shape[0]
        z_past = self.encode(y_arr)[:, -self.n_y :].reshape(n_batch, -1)
        u_past = u_arr[:, -self.n_u :].reshape(n_batch, -1)
        return np.concatenate([z_past, u_past], axis=-1)

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, state)`` and raw ``(B, steps, n_controls)``.

        Returns ``(B, steps, n_channels)``.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        u_arr = np.asarray(u_futures, dtype=np.float64)
        n_batch, steps = states_arr.shape[0], u_arr.shape[1]
        n_z = self.n_y * self.n_channels
        y_window = states_arr[:, :n_z].reshape(n_batch, self.n_y, self.n_channels)
        u_window = self.u_std.transform(states_arr[:, n_z:].reshape(n_batch, self.n_u, self.n_controls))

        preds = np.empty((n_batch, steps, self.n_channels), dtype=np.float64)
        for t in range(steps):
            z_next = self._forward_1step(y_window, u_window)
            preds[:, t] = self.decode(z_next)
            y_window = np.concatenate([y_window[:, 1:], z_next[:, None, :]], axis=1)
            u_window = np.concatenate([u_window[:, 1:], self.u_std.transform(u_arr[:, t, None, :])], axis=1)
        return preds

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new raw measurement ``y`` and applied control ``u`` into the shift register."""
        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z:].reshape(self.n_u, self.n_controls)

        z = self.encode(np.asarray(y, dtype=np.float64).reshape(-1))
        y_window = np.concatenate([y_window[1:], z[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        return np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        return not np.isnan(np.asarray(state, dtype=np.float64)[: self.n_y * self.n_channels]).any()

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window and zero-padded control window: nothing absorbed yet."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def to_artifact(self, dt: float, downsample: int, y_std: Standardizer, u_std: Standardizer) -> MLPArtifact:
        """Freeze the trained weights and standardizers into a framework-free artifact."""
        linears = (m for m in self.layers if isinstance(m, nn.Linear))
        layers = tuple((to_numpy(lin.weight), to_numpy(lin.bias)) for lin in linears)
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
        """Rebuild the module carrying the artifact's weights, dt and standardizer buffers."""
        model = cls(
            n_y=art.n_y,
            n_u=art.n_u,
            horizon=art.horizon,
            n_channels=art.n_channels,
            n_controls=art.n_controls,
            hidden_size=art.layers[0][0].shape[0],
            depth=len(art.layers) - 1,
            activation=art.activation,
            dt=art.dt,
            y_std=art.y_std,
            u_std=art.u_std,
        )
        linears = (m for m in model.layers if isinstance(m, nn.Linear))
        with torch.no_grad():
            for lin, (w, b) in zip(linears, art.layers, strict=True):
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        return model
