from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from neuro.predictor.module import to_numpy
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    import scipy.sparse
    from torch import Tensor

    from neuro.types import FloatArray


def _coo_buffer(w_res: scipy.sparse.csr_matrix) -> Tensor:
    """Copy a scipy CSR reservoir into a float32 sparse COO torch buffer."""
    coo = w_res.tocoo()
    indices = torch.as_tensor(np.stack([coo.row, coo.col]), dtype=torch.int64)
    values = torch.as_tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=w_res.shape, check_invariants=False)


class ESNModule(nn.Module):
    """Echo State Network predictor as a torch module satisfying the Predictor protocol.

    Reservoir generation stays the one-time scipy preprocessing step; its outputs -- the sparse
    ``W_res`` and dense ``W_in`` -- are copied into float32 buffers, and the runtime (``absorb``,
    ``readout``, ``step``, ``rollout``) runs in torch. The opaque state is the reservoir vector
    followed by a step counter, so ``is_ready`` can compare against ``priming_steps`` without an
    ndarray subclass. Raw units at the boundary: the channel and control standardizers are
    float32 buffers, invisible through the protocol.

    Attributes
    ----------
    w_res : Tensor
        Sparse COO float32 reservoir matrix (N, N).
    w_in : Tensor
        Dense float32 input weight matrix (N, C + m + 1), its last column the constant-1 bias.
    w_out : Tensor
        Dense float32 readout matrix (C, N + 1) mapping ``[h; 1]`` to standardized EEG.
    leak_rate : float
        Leakage alpha in (0, 1].
    reservoir_size : int
        Number of reservoir units N.
    n_channels, n_controls : int
        EEG and control channel counts.
    priming_steps : int
        Minimum number of absorbed samples before the state is ready.
    horizon : int
        The native/trained horizon, identity metadata only, not a bound on ``rollout``.
    dt : float
        The model's native time step, seconds.
    y_center, y_scale, u_center, u_scale : Tensor
        Float32 standardizer buffers mapping raw units to the standardized model space.
    """

    w_res: Tensor
    w_in: Tensor
    w_out: Tensor
    y_center: Tensor
    y_scale: Tensor
    u_center: Tensor
    u_scale: Tensor

    def __init__(  # noqa: PLR0913
        self,
        *,
        w_res: scipy.sparse.csr_matrix,
        w_in: FloatArray,
        w_out: FloatArray,
        leak_rate: float,
        priming_steps: int,
        horizon: int,
        dt: float = 0.0,
        y_std: Standardizer | None = None,
        u_std: Standardizer | None = None,
    ) -> None:
        """Build the module from the scipy-generated reservoir and the fitted readout.

        ``w_res``, ``w_in`` and ``w_out`` are the one-time scipy preprocessing outputs
        (:func:`neuro.esn.generate_reservoir` and the ridge fit) copied into float32 buffers;
        ``y_std``/``u_std`` become the module's float32 buffers, defaulting to the identity map.
        """
        super().__init__()
        self.reservoir_size = w_res.shape[0]
        self.n_channels = w_out.shape[0]
        self.n_controls = w_in.shape[1] - self.n_channels - 1
        self.leak_rate = float(leak_rate)
        self.priming_steps = int(priming_steps)
        self.horizon = int(horizon)
        self.dt = float(dt)

        self.register_buffer("w_res", _coo_buffer(w_res))
        self.register_buffer("w_in", torch.as_tensor(w_in, dtype=torch.float32))
        self.register_buffer("w_out", torch.as_tensor(w_out, dtype=torch.float32))

        y_std = y_std or Standardizer(center=np.zeros(self.n_channels), scale=np.ones(self.n_channels))
        u_std = u_std or Standardizer(center=np.zeros(self.n_controls), scale=np.ones(self.n_controls))
        self.register_buffer("y_center", torch.as_tensor(y_std.center, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_std.scale, dtype=torch.float32))
        self.register_buffer("u_center", torch.as_tensor(u_std.center, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(u_std.scale, dtype=torch.float32))

    @property
    def n_outputs(self) -> int:
        """Output width per position: one EEG sample across all channels."""
        return self.n_channels

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

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_channels)`` into standardized model space."""
        return self.y_std.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_channels)`` from standardized model space."""
        return self.y_std.inverse_transform(np.asarray(z, dtype=np.float64))

    def _apply_w_res(self, h: Tensor) -> Tensor:
        """Apply the reservoir ``W_res @ h`` for ``h (..., N)`` -> ``(..., N)`` (``h @ W_res.T``)."""
        if h.ndim == 1:
            return torch.sparse.mm(self.w_res, h.unsqueeze(-1)).squeeze(-1)
        return torch.sparse.mm(self.w_res, h.T).T

    def _readout(self, h: Tensor) -> Tensor:
        """One-step-ahead standardized prediction ``z_hat`` from ``h (..., N)`` -> ``(..., C)``."""
        ones = h.new_ones((*h.shape[:-1], 1))
        return torch.cat([h, ones], dim=-1) @ self.w_out.T

    def _absorb(self, h: Tensor, z: Tensor, v: Tensor) -> Tensor:
        """Advance ``h (..., N)`` one step, absorbing the standardized input ``(z, v)``."""
        x_in = torch.cat([z, v, v.new_ones((*v.shape[:-1], 1))], dim=-1)
        alpha = self.leak_rate
        return (1.0 - alpha) * h + alpha * torch.tanh(self._apply_w_res(h) + x_in @ self.w_in.T)

    def _split(self, state: FloatArray) -> tuple[FloatArray, float]:
        """Split an opaque state into the reservoir vector ``(N,)`` and the absorbed-step counter."""
        state_arr = np.asarray(state, dtype=np.float64)
        return state_arr[: self.reservoir_size], float(state_arr[self.reservoir_size])

    def _pack(self, h: FloatArray, count: float) -> FloatArray:
        """Join the reservoir vector ``(N,)`` and the step counter into an opaque state ``(N + 1,)``."""
        return np.concatenate([np.asarray(h, dtype=np.float64), np.asarray([count], dtype=np.float64)])

    def initial_state(self) -> FloatArray:
        """Return the unprimed state: a zero reservoir vector and zero absorbed steps."""
        return self._pack(np.zeros(self.reservoir_size, dtype=np.float64), 0.0)

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the state has absorbed at least ``priming_steps`` samples."""
        return self._split(state)[1] >= self.priming_steps

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new raw measurement ``y`` and applied control ``u``, advancing the counter."""
        h, count = self._split(state)
        z = torch.as_tensor(self.encode(np.asarray(y, dtype=np.float64).reshape(-1)), dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u, dtype=np.float64).reshape(-1)), dtype=torch.float32)
        h_next = self._absorb(torch.as_tensor(h, dtype=torch.float32), z, v)
        return self._pack(to_numpy(h_next), count + 1.0)

    def step(self, state: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Advance one position (one sample) free-running under raw control ``u`` -> ``(state', output)``.

        The readout of the current reservoir state is the emitted raw output; the same
        standardized readout feeds the update, exactly the numpy runtime's ``step``.
        """
        h, count = self._split(state)
        h_t = torch.as_tensor(h, dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u, dtype=np.float64).reshape(-1)), dtype=torch.float32)
        z_hat = self._readout(h_t)
        h_next = self._absorb(h_t, z_hat, v)
        return self._pack(to_numpy(h_next), count + 1.0), self.decode(to_numpy(z_hat))

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from ``state`` under raw ``u_future`` -> raw ``(steps, n_channels)``.

        ``u_future[t]`` is the control applied at prediction step ``t``; the length is not
        bounded by ``horizon``. Equivalent to a loop of :meth:`step`.
        """
        h, _ = self._split(state)
        h_t = torch.as_tensor(h, dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u_future, dtype=np.float64)), dtype=torch.float32)
        preds = np.empty((v.shape[0], self.n_channels), dtype=np.float64)
        for t in range(v.shape[0]):
            z_hat = self._readout(h_t)
            preds[t] = self.decode(to_numpy(z_hat))
            h_t = self._absorb(h_t, z_hat, v[t])
        return preds

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into the initial state, the reservoir started at zero (Priming).

        ``y_hist (k, n_channels)`` and ``u_hist (k, n_controls)`` both end at the same step
        ``t - 1``, so :meth:`rollout` predicts from ``t`` onwards. The state's step counter is
        ``k``, so :meth:`is_ready` reports True exactly when ``k >= priming_steps``.
        """
        z = torch.as_tensor(self.encode(np.asarray(y_hist, dtype=np.float64)), dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u_hist, dtype=np.float64)), dtype=torch.float32)
        h_t = torch.zeros(self.reservoir_size, dtype=torch.float32)
        for t in range(z.shape[0]):
            h_t = self._absorb(h_t, z[t], v[t])
        return self._pack(to_numpy(h_t), float(z.shape[0]))

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
        z = torch.as_tensor(self.encode(np.asarray(y_hists, dtype=np.float64)), dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u_hists, dtype=np.float64)), dtype=torch.float32)
        h_t = torch.zeros((z.shape[0], self.reservoir_size), dtype=torch.float32)
        for t in range(z.shape[1]):
            h_t = self._absorb(h_t, z[:, t], v[:, t])
        counts = np.full((z.shape[0],), float(z.shape[1]), dtype=np.float64)
        return np.concatenate([to_numpy(h_t), counts[:, None]], axis=-1)

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, state)`` and raw ``(B, steps, n_controls)``.

        Returns ``(B, steps, n_channels)``.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        h_t = torch.as_tensor(states_arr[:, : self.reservoir_size], dtype=torch.float32)
        v = torch.as_tensor(self.u_std.transform(np.asarray(u_futures, dtype=np.float64)), dtype=torch.float32)
        preds = np.empty((v.shape[0], v.shape[1], self.n_channels), dtype=np.float64)
        for t in range(v.shape[1]):
            z_hat = self._readout(h_t)
            preds[:, t] = self.decode(to_numpy(z_hat))
            h_t = self._absorb(h_t, z_hat, v[:, t])
        return preds

    def solve_ridge(self, G: FloatArray, P: FloatArray, ridge_lambda: float) -> FloatArray:
        """Solve the ridge readout ``W_out = (G + lambda I)^-1 P`` in torch, bias column unregularized.

        Reproduces :func:`neuro.esn.solve_ridge` -- the last diagonal entry of ``G`` gets no
        regularization. Runs in float64 so the one-time fit matches the incumbent NumPy solve to
        LAPACK precision.
        """
        g = torch.as_tensor(np.asarray(G, dtype=np.float64))
        p = torch.as_tensor(np.asarray(P, dtype=np.float64))
        reg = ridge_lambda * torch.eye(g.shape[0], dtype=torch.float64)
        reg[-1, -1] = 0.0  # do not regularize the bias column
        return to_numpy(torch.linalg.solve(g + reg, p)).T
