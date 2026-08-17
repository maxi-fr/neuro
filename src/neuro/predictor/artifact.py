from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neuro.transforms import Pipeline

if TYPE_CHECKING:
    from neuro.types import FloatArray

Activation = Literal["relu", "tanh", "softplus"]


def _activate(z: FloatArray, activation: Activation) -> FloatArray:
    """Apply the named activation elementwise (matching ``_mlp_forward_ca`` and ``torch.nn.functional``)."""
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "tanh":
        return np.tanh(z)
    if activation == "softplus":
        return np.logaddexp(z, 0.0)
    msg = f"Unsupported activation: {activation}"
    raise ValueError(msg)


@dataclass(frozen=True)
class MLPArtifact:
    """Framework-free autoregressive MLP predictor: NumPy weights, native dt, and transforms.

    Attributes
    ----------
    layers : tuple[tuple[FloatArray, FloatArray], ...]
        Per-layer ``(weight (out, in), bias (out,))`` pairs in forward-pass order.
    activation : Activation
        Activation applied after every layer except the last.
    n_y : int
        Number of past EEG (output) steps in the model's history window.
    n_u : int
        Number of past control (input) steps in the model's history window.
    horizon : int
        Direct-prediction horizon the model was trained on.
    n_channels : int
        Model-space channel count: the latent dimension ``k`` under PCA, else the raw EEG count.
    n_controls : int
        Number of control (input) channels.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    y_pipeline : Pipeline
        Raw EEG -> model space map: a channel-space :class:`~neuro.transforms.Standardizer`
        followed by an optional :class:`~neuro.transforms.PCAProjection`. Encodes measured EEG
        the model consumes and decodes its predictions back to raw EEG channels.
    u_pipeline : Pipeline
        Raw control -> model space map (a single standardizer).
    """

    layers: tuple[tuple[FloatArray, FloatArray], ...]
    activation: Activation
    n_y: int
    n_u: int
    horizon: int
    n_channels: int
    n_controls: int
    dt: float
    downsample: int
    y_pipeline: Pipeline
    u_pipeline: Pipeline

    @property
    def model_type(self) -> str:
        """Model architecture type string ('mlp')."""
        return "mlp"

    @property
    def n_eeg_channels(self) -> int:
        """Number of raw EEG channels (the PCA basis' input dimension, else ``n_channels``)."""
        pca = self.y_pipeline.pca
        return pca.basis.shape[1] if pca is not None else self.n_channels

    @property
    def priming_steps(self) -> int:
        """Minimum number of history steps required to initialize the model state."""
        return max(self.n_y, self.n_u)

    @property
    def is_linear(self) -> bool:
        """Whether the MLP is linear (a single layer, i.e. no hidden layers)."""
        return len(self.layers) == 1

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_eeg_channels)`` into model space (standardize, then project)."""
        return self.y_pipeline.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_eeg_channels)`` from model-space values."""
        return self.y_pipeline.inverse_transform(np.asarray(z, dtype=np.float64))

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into an initial state (model-space y, **raw** u tail).

        y_hist (k, n_eeg_channels), u_hist (k, n_controls).
        """
        y_arr = np.asarray(y_hist, dtype=np.float64)
        u_arr = np.asarray(u_hist, dtype=np.float64)
        z_past = self.encode(y_arr)[-self.n_y :].reshape(-1)
        u_past = u_arr[-self.n_u :].reshape(-1)
        return np.concatenate([z_past, u_past])

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_eeg_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
        y_arr = np.asarray(y_hists, dtype=np.float64)
        u_arr = np.asarray(u_hists, dtype=np.float64)
        n_batch = y_arr.shape[0]
        z_past = self.encode(y_arr)[:, -self.n_y :].reshape(n_batch, -1)
        u_past = u_arr[:, -self.n_u :].reshape(n_batch, -1)
        return np.concatenate([z_past, u_past], axis=-1)

    def forward_1step(self, y_flat: FloatArray, u_flat: FloatArray) -> FloatArray:
        """One-step MLP forward on the model-space input ``[y_window | u_window]``.

        ``y_flat`` is ``(..., n_y * n_channels)`` and ``u_flat`` ``(..., n_u * n_controls)``, both
        flattened row-major with the newest step last; returns ``(..., n_channels)``.
        """
        x = np.concatenate([y_flat, u_flat], axis=-1)
        for w, b in self.layers[:-1]:
            x = _activate(x @ w.T + b, self.activation)
        w_last, b_last = self.layers[-1]
        return x @ w_last.T + b_last

    def rollout(self, state: FloatArray, u_future: FloatArray) -> FloatArray:
        """Free-run from ``state`` under raw ``u_future`` (steps, n_controls) -> (steps, n_eeg_channels)."""
        w_future = self.u_pipeline.transform(np.asarray(u_future, dtype=np.float64))
        steps = len(w_future)

        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = self.u_pipeline.transform(state_arr[n_z:].reshape(self.n_u, self.n_controls))

        preds = np.empty((steps, self.n_channels), dtype=np.float64)
        for t in range(steps):
            # The control window shifts *before* the MLP call, so the newest control is already
            # in the window when y_next is predicted.
            u_window = np.concatenate([u_window[1:], w_future[t][None, :]], axis=0)
            y_next = self.forward_1step(y_window.reshape(-1), u_window.reshape(-1))
            y_window = np.concatenate([y_window[1:], y_next[None, :]], axis=0)
            preds[t] = y_next

        return self.decode(preds)

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, state)`` and raw ``(B, steps, n_controls)``.

        Returns ``(B, steps, n_eeg_channels)``.
        """
        w_future = self.u_pipeline.transform(np.asarray(u_futures, dtype=np.float64))
        steps = w_future.shape[1]

        states_arr = np.asarray(states, dtype=np.float64)
        n_batch = states_arr.shape[0]
        n_z = self.n_y * self.n_channels
        y_window = states_arr[:, :n_z].reshape(n_batch, self.n_y, self.n_channels)
        u_window = self.u_pipeline.transform(states_arr[:, n_z:].reshape(n_batch, self.n_u, self.n_controls))

        preds = np.empty((n_batch, steps, self.n_channels), dtype=np.float64)
        for t in range(steps):
            u_window = np.concatenate([u_window[:, 1:], w_future[:, t, None, :]], axis=1)
            y_next = self.forward_1step(y_window.reshape(n_batch, -1), u_window.reshape(n_batch, -1))
            y_window = np.concatenate([y_window[:, 1:], y_next[:, None, :]], axis=1)
            preds[:, t] = y_next

        return self.decode(preds)

    @property
    def meta(self) -> dict[str, Any]:
        """Serializable dictionary representation of artifact metadata."""
        return {
            "model_type": self.model_type,
            "activation": self.activation,
            "n_y": self.n_y,
            "n_u": self.n_u,
            "horizon": self.horizon,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
            "n_eeg_channels": self.n_eeg_channels,
            "dt": self.dt,
            "downsample": self.downsample,
            "n_layers": len(self.layers),
            "y_pipeline": self.y_pipeline.step_tags(),
            "u_pipeline": self.u_pipeline.step_tags(),
        }

    @classmethod
    def load(cls, artifact: str | Path) -> MLPArtifact:
        """Load the single-``.npz`` artifact from disk (``artifact`` is a suffix-less stem)."""
        path = Path(artifact).with_suffix(".npz")
        with np.load(path) as npz:
            meta: dict[str, Any] = json.loads(str(npz["meta"]))
            layers = tuple(
                (
                    np.asarray(npz[f"layer.{i}.weight"], dtype=np.float64),
                    np.asarray(npz[f"layer.{i}.bias"], dtype=np.float64),
                )
                for i in range(int(meta["n_layers"]))
            )
            arrays = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files if k.startswith(("y.", "u."))}

        return cls(
            layers=layers,
            activation=meta["activation"],
            n_y=int(meta["n_y"]),
            n_u=int(meta["n_u"]),
            horizon=int(meta["horizon"]),
            n_channels=int(meta["n_channels"]),
            n_controls=int(meta["n_controls"]),
            dt=float(meta["dt"]),
            downsample=int(meta["downsample"]),
            y_pipeline=Pipeline.from_serialized("y", meta["y_pipeline"], arrays),
            u_pipeline=Pipeline.from_serialized("u", meta["u_pipeline"], arrays),
        )

    def save(self, artifact: str | Path) -> None:
        """Persist weights, transforms and metadata into one ``.npz`` (``artifact`` is a stem).

        ``meta`` is stored as a 0-d unicode array holding JSON, so loading needs no ``allow_pickle``.
        """
        path = Path(artifact).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray[Any, Any]] = {"meta": np.array(json.dumps(self.meta))}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer.{i}.weight"] = w
            arrays[f"layer.{i}.bias"] = b
        arrays.update(self.y_pipeline.array_dict("y"))
        arrays.update(self.u_pipeline.array_dict("u"))

        np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
