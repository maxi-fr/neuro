from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch import nn

from neuro.observable import ObservableArtifact, control_means, geometry_from_meta, geometry_meta, log_observable
from neuro.predictor.artifact import ACTIVATIONS
from neuro.predictor.checkpoint import load_checkpoint, save_checkpoint
from neuro.predictor.data import build_dataset_for_trajectory, extract_future_windows
from neuro.predictor.module import activation_module, to_numpy
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from neuro.config import ObservableGeometry
    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray


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


class StepwiseObservableMLP(nn.Module):
    """One-Frame-per-step Observable predictor satisfying the Predictor protocol directly.

    The runtime recurses on the Frame grid: ``step`` advances exactly one Frame through the shared
    transition and readout and emits that Frame's raw log-Observable, and ``rollout`` aggregates
    raw future controls into Frame means via :func:`neuro.observable.control_means` before
    unrolling ``step``. The standardized batched ``forward`` unrolls the same recursion for
    training (BPTT depth is the Frame count, not the Control Horizon), so callers exchange raw
    units only at the protocol boundary. There is no artifact hand-off: the standardizers live in
    this module as float32 buffers.

    The opaque state is ``[standardized EEG window | raw control window | lifted Frame state]``:
    the register mirrors the waveform module's shift register (``absorb``/``is_ready``), and the
    module lifts once at ``prime``/``absorb`` so ``step``/``rollout`` only recurse on the carried
    Frame state. The incumbent one-shot :class:`ObservableMLP` and the artifact stay in place for
    the controller until the contract ticket; this module is the protocol-compliant replacement.

    Attributes
    ----------
    lift : nn.Sequential
        ``E``: the history register ``[y_window | u_window]`` -> ``z_0``.
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
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
        l_std: Standardizer | None = None,
    ) -> None:
        """Build the lift, the shared Frame transition, the log readout and the standardizer buffers.

        ``y_std``/``u_std``/``l_std`` become the module's float32 buffers; when omitted they
        default to the identity map, so a module built before the standardizers are fitted treats
        raw units as model space.
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
        maps are affine.
        """
        batch = x.shape[0]
        u_future = x[:, self._n_hist :].reshape(batch, self.horizon, self.n_controls)
        u_bar = torch.einsum("mt,btc->bmc", self.aggregate, u_future)

        z = self.lift(x[:, : self._n_hist])
        frames = []
        for m in range(u_bar.shape[1]):
            z = self.transition(torch.cat([z, u_bar[:, m]], dim=1))
            frames.append(self.readout(z))
        return torch.stack(frames, dim=1)

    @property
    def dt(self) -> float:
        """The model's native time step, seconds: the sample period the Frame grid resolves at."""
        return 1.0 / self.fs

    @property
    def priming_steps(self) -> int:
        """Minimum number of history samples required to prime the state."""
        return max(self.n_y, self.n_u)

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

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_channels)`` into standardized channel space."""
        return self.y_std.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_channels)`` from standardized space."""
        return self.y_std.inverse_transform(np.asarray(z, dtype=np.float64))

    def _lift_batch(self, register: FloatArray) -> FloatArray:
        """Lift batch shift registers ``(B, n_hist)`` to Frame states ``(B, z_dim)``."""
        n_z = self.n_y * self.n_channels
        y_std = register[:, :n_z]
        u_raw = register[:, n_z:].reshape(-1, self.n_u, self.n_controls)
        lift_in = np.concatenate(
            [y_std, self.u_std.transform(u_raw).reshape(-1, self.n_u * self.n_controls)],
            axis=-1,
        )
        with torch.no_grad():
            z = self.lift(torch.as_tensor(lift_in, dtype=torch.float32))
        return to_numpy(z)

    def _step_batch(self, z: FloatArray, u_bar: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Advance lifted states ``(B, z_dim)`` one Frame under raw Frame-mean controls ``(B, n_controls)``.

        Returns the next lifted state ``(B, z_dim)`` and that Frame's raw log-Observable
        ``(B, n_channels * n_values)``.
        """
        u_std = self.u_std.transform(u_bar)
        with torch.no_grad():
            z_next = self.transition(torch.as_tensor(np.concatenate([z, u_std], axis=-1), dtype=torch.float32))
            l_std = self.readout(z_next)
        return to_numpy(z_next), self.l_std.inverse_transform(to_numpy(l_std))

    def design_normal_equations(
        self, trajectories: list[tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """Harvest the per-Frame lifted states ``z_m`` and their Frame targets into ``(G, P)``.

        Every window on the shared grid lifts once and recurses through the shared transition;
        each post-transition state pairs with its Frame's standardized log-Observable target, so
        the readout -- shared across Frames -- is fitted on every ``(z_m, target)`` pair of every
        window. The bias column (last) is the constant-1 feature.
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
                self.u_std.transform(u_arr), self.encode(y_arr), self.n_y, self.n_u, self.horizon
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
            # Frame-major rows, paired with the frame-major targets below.
            T_mat = targets.transpose(1, 0, 2).reshape(-1, c)
            G += H_mat.T @ H_mat
            P += H_mat.T @ T_mat
        return G, P

    def install_readout(self, A: FloatArray) -> None:
        """Write the ridge-fitted shared readout ``A (n_outputs, z_dim + 1)``, bias column last."""
        with torch.no_grad():
            self.readout.weight.copy_(torch.as_tensor(np.ascontiguousarray(A[:, :-1]), dtype=torch.float32))
            self.readout.bias.copy_(torch.as_tensor(A[:, -1], dtype=torch.float32))

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into an initial state: the register plus the lifted Frame state.

        ``y_hist (k, n_channels)`` and ``u_hist (k, n_controls)`` both end at the same step
        ``t - 1``, so :meth:`rollout` predicts from ``t`` onwards. The register part is
        byte-identical in layout to :meth:`neuro.observable.ObservableArtifact.prime`.
        """
        y_arr = np.asarray(y_hist, dtype=np.float64)
        u_arr = np.asarray(u_hist, dtype=np.float64)
        register = np.concatenate([self.encode(y_arr)[-self.n_y :].reshape(-1), u_arr[-self.n_u :].reshape(-1)])
        return np.concatenate([register, self._lift_batch(register[None, :])[0]])

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
        y_arr = np.asarray(y_hists, dtype=np.float64)
        u_arr = np.asarray(u_hists, dtype=np.float64)
        n_batch = y_arr.shape[0]
        register = np.concatenate(
            [
                self.encode(y_arr)[:, -self.n_y :].reshape(n_batch, -1),
                u_arr[:, -self.n_u :].reshape(n_batch, -1),
            ],
            axis=-1,
        )
        return np.concatenate([register, self._lift_batch(register)], axis=-1)

    def step(self, state: FloatArray, u_bar: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Advance one position (one Frame): apply raw Frame-mean control ``u_bar`` -> ``(state', output)``.

        ``u_bar`` is one Frame-mean ``(n_controls,)`` in raw units. The register is carried
        unchanged; only the lifted Frame state advances through the shared transition and readout.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        u_arr = np.asarray(u_bar, dtype=np.float64).reshape(-1)
        z = state_arr[self._n_hist :]
        z_next, l_raw = self._step_batch(z[None, :], u_arr[None, :])
        return np.concatenate([state_arr[: self._n_hist], z_next[0]]), l_raw[0]

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from ``state`` under raw ``u_future`` -> raw ``(n_frames, n_channels * n_values)``.

        Raw future controls are aggregated into Frame means via :func:`control_means` and the
        result is one :meth:`step` per Frame. The length of ``u_future`` is not bounded by
        ``horizon``.
        """
        u_arr = np.asarray(u_future, dtype=np.float64)
        n_frames = self.geometry.n_frames(len(u_arr), self.fs)
        u_bar = control_means(self.geometry, len(u_arr), self.fs) @ u_arr

        preds = np.empty((n_frames, self.n_outputs), dtype=np.float64)
        for m in range(n_frames):
            state, l_raw = self.step(state, u_bar[m])
            preds[m] = l_raw
        return preds

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, state)`` and raw ``(B, horizon, n_controls)``.

        Returns ``(B, n_frames, n_channels * n_values)``. The register is constant across the
        rollout, so only the lifted Frame states are carried between Frames.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        u_arr = np.asarray(u_futures, dtype=np.float64)
        n_batch, horizon = u_arr.shape[0], u_arr.shape[1]
        n_frames = self.geometry.n_frames(horizon, self.fs)
        u_bar = np.einsum("mt,btc->bmc", control_means(self.geometry, horizon, self.fs), u_arr)

        z = states_arr[:, self._n_hist :]
        preds = np.empty((n_batch, n_frames, self.n_outputs), dtype=np.float64)
        for m in range(n_frames):
            z, l_raw = self._step_batch(z, u_bar[:, m])
            preds[:, m] = l_raw
        return preds

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new raw measurement ``y`` and applied control ``u``, re-lifting the Frame state."""
        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z : self._n_hist].reshape(self.n_u, self.n_controls)

        z_new = self.encode(np.asarray(y, dtype=np.float64).reshape(-1))
        y_window = np.concatenate([y_window[1:], z_new[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        register = np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        return np.concatenate([register, self._lift_batch(register[None, :])[0]])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        return not np.isnan(np.asarray(state, dtype=np.float64)[: self.n_y * self.n_channels]).any()

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window, zero-padded control window, and their (NaN) lifted Frame state."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        register = np.concatenate([y_buf, u_buf])
        return np.concatenate([register, self._lift_batch(register[None, :])[0]])

    def save(self, path: str | Path) -> None:
        """Persist weights, standardizer buffers and recorded metadata into one ``.npz`` checkpoint.

        ``path`` is a suffix-less stem. The layout -- a JSON ``meta`` block carrying the geometry,
        provenance and model_type, the block weight arrays and the standardizer arrays -- is the
        one the artifact loaders already read, so the torch-free control path keeps consuming what
        ``save`` writes without the module.
        """
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
            "geometry": geometry_meta(self.geometry),
            **self.provenance.meta,
        }
        arrays: dict[str, FloatArray] = {}
        for prefix, blocks in (("lift", linears), ("transition", transitions)):
            for i, lin in enumerate(blocks):
                arrays[f"{prefix}.{i}.weight"] = to_numpy(lin.weight)
                arrays[f"{prefix}.{i}.bias"] = to_numpy(lin.bias)
        arrays["readout.weight"] = to_numpy(self.readout.weight)
        arrays["readout.bias"] = to_numpy(self.readout.bias)
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))
        arrays.update(self.l_std.arrays("l"))
        save_checkpoint(path, meta=meta, arrays=arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Rebuild the module from a :meth:`save` checkpoint, restoring weights, buffers and metadata."""
        meta, arrays = load_checkpoint(path)
        if meta.get("model_type") != "observable":
            msg = f"checkpoint at {path} is model_type {meta.get('model_type')!r}, not 'observable'."
            raise ValueError(msg)
        if meta["activation"] not in ACTIVATIONS:
            msg = f"Unsupported activation: {meta['activation']!r}"
            raise ValueError(msg)
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
            y_std=Standardizer.from_arrays(arrays, "y"),
            u_std=Standardizer.from_arrays(arrays, "u"),
            l_std=Standardizer.from_arrays(arrays, "l"),
        )
        model.downsample = int(meta["downsample"])
        model.provenance = TrainingProvenance.from_meta(meta)
        with torch.no_grad():
            for block, prefix in ((model.lift, "lift"), (model.transition, "transition")):
                for i, lin in enumerate(m for m in block if isinstance(m, nn.Linear)):
                    lin.weight.copy_(torch.as_tensor(np.asarray(arrays[f"{prefix}.{i}.weight"]), dtype=torch.float32))
                    lin.bias.copy_(torch.as_tensor(np.asarray(arrays[f"{prefix}.{i}.bias"]), dtype=torch.float32))
            model.readout.weight.copy_(torch.as_tensor(np.asarray(arrays["readout.weight"]), dtype=torch.float32))
            model.readout.bias.copy_(torch.as_tensor(np.asarray(arrays["readout.bias"]), dtype=torch.float32))
        return model
