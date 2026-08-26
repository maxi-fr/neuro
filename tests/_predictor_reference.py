"""Float64 numpy reference runtimes for the predictor tests, rebuilt from checkpoint buffers.

The legacy artifacts and their NumPy runtimes are deleted (the contract ticket), but the tests
still need a float64 reference to pin the float32 torch modules and the CasADi adapters against.
These hand-rolled rollouts are that reference: they read only the checkpoint dataclasses' float64
buffers and the shared geometry helpers, never a module and never torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.observable import control_means

if TYPE_CHECKING:
    from neuro.checkpoint import MLPCheckpoint, ObservableCheckpoint
    from neuro.types import FloatArray, Layers


def activate(z: FloatArray, activation: str) -> FloatArray:
    """Apply the named activation elementwise in float64."""
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "tanh":
        return np.tanh(z)
    return np.logaddexp(z, 0.0)


def mlp_forward(x: FloatArray, layers: Layers, activation: str) -> FloatArray:
    """Evaluate an MLP forward pass on ``(..., in_size)``; the activation follows all but the last layer."""
    for w, b in layers[:-1]:
        x = activate(x @ w.T + b, activation)
    w_last, b_last = layers[-1]
    return x @ w_last.T + b_last


def mlp_prime(ckpt: MLPCheckpoint, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
    """Prime the shift register: the standardized EEG tail followed by the raw control tail."""
    y_arr = np.asarray(y_hist, dtype=np.float64)
    u_arr = np.asarray(u_hist, dtype=np.float64)
    z_past = ckpt.y_std.transform(y_arr)[-ckpt.n_y :].reshape(-1)
    u_past = u_arr[-ckpt.n_u :].reshape(-1)
    return np.concatenate([z_past, u_past])


def mlp_rollout(ckpt: MLPCheckpoint, state: FloatArray, u_future: FloatArray) -> FloatArray:
    """Free-run the checkpoint's float64 weights from ``state`` under raw controls -> raw EEG.

    ``u_future[t]`` is the control applied at prediction step ``t``, so it first enters the window
    for step ``t + 1`` and the last entry is never consumed; the prime() state is already level.
    """
    w_future = ckpt.u_std.transform(np.asarray(u_future, dtype=np.float64))
    steps = len(w_future)
    n_z = ckpt.n_y * ckpt.n_channels
    y_window = np.asarray(state, dtype=np.float64)[:n_z].reshape(ckpt.n_y, ckpt.n_channels)
    u_window = ckpt.u_std.transform(np.asarray(state, dtype=np.float64)[n_z:].reshape(ckpt.n_u, ckpt.n_controls))

    preds = np.empty((steps, ckpt.n_channels), dtype=np.float64)
    for t in range(steps):
        y_next = mlp_forward(np.concatenate([y_window.reshape(-1), u_window.reshape(-1)]), ckpt.layers, ckpt.activation)
        if ckpt.residual:
            y_next = y_next + y_window[-1]
        y_window = np.concatenate([y_window[1:], y_next[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], w_future[t][None, :]], axis=0)
        preds[t] = y_next
    return ckpt.y_std.inverse_transform(preds)


def observable_prime(ckpt: ObservableCheckpoint, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
    """Prime the history register: the standardized EEG tail followed by the raw control tail."""
    y_arr = np.asarray(y_hist, dtype=np.float64)
    u_arr = np.asarray(u_hist, dtype=np.float64)
    z_past = ckpt.y_std.transform(y_arr)[-ckpt.n_y :].reshape(-1)
    u_past = u_arr[-ckpt.n_u :].reshape(-1)
    return np.concatenate([z_past, u_past])


def observable_forecast(ckpt: ObservableCheckpoint, state: FloatArray, u_future: FloatArray) -> FloatArray:
    """Forecast the raw log-Observable from ``state`` under raw controls -> ``(n_frames, C, F)``.

    The lift runs once; each post-transition Frame state pairs with its standardized log readout,
    decoded to raw units and reshaped channel-major over the ``(channel, value)`` grid.
    """
    u_arr = np.asarray(u_future, dtype=np.float64)
    horizon = len(u_arr)
    u_bar = ckpt.u_std.transform(control_means(ckpt.geometry, horizon, ckpt.fs) @ u_arr)

    n_z = ckpt.n_y * ckpt.n_channels
    state_arr = np.asarray(state, dtype=np.float64)
    u_past = ckpt.u_std.transform(state_arr[n_z:].reshape(ckpt.n_u, ckpt.n_controls))
    lift_in = np.concatenate([state_arr[:n_z], u_past.reshape(-1)])
    z = mlp_forward(lift_in, ckpt.lift, ckpt.activation)

    frames = []
    if ckpt.residual:
        carry = np.zeros(ckpt.n_channels * ckpt.n_values)
        for m in range(len(u_bar)):
            z = mlp_forward(np.concatenate([z, u_bar[m]]), ckpt.transition, ckpt.activation)
            carry = carry + (z @ ckpt.readout[0].T + ckpt.readout[1])
            frames.append(carry)
    else:
        for m in range(len(u_bar)):
            z = mlp_forward(np.concatenate([z, u_bar[m]]), ckpt.transition, ckpt.activation)
            frames.append(z @ ckpt.readout[0].T + ckpt.readout[1])
    stacked = ckpt.l_std.inverse_transform(np.stack(frames, axis=0))
    return stacked.reshape(len(frames), ckpt.n_channels, ckpt.n_values)
