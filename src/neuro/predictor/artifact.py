from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

Activation = Literal["relu", "tanh", "softplus"]
ACTIVATIONS: frozenset[str] = frozenset(get_args(Activation))


def _activate(z: FloatArray, activation: Activation) -> FloatArray:
    """Apply the named activation elementwise (matching ``_mlp_forward_ca`` and ``torch.nn.functional``).

    The name is validated once, in :meth:`MLPArtifact.load`, so the softplus branch is the fallthrough.
    """
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "tanh":
        return np.tanh(z)
    return np.logaddexp(z, 0.0)


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
        Physical EEG channel count.
    n_controls : int
        Number of control (input) channels.
    dt : float
        The model's native time step, seconds.
    downsample : int
        Downsampling factor relative to the simulation's base ``dt``.
    y_std : Standardizer
        Channel standardizer mapping raw EEG to standardized model space and back.
    u_std : Standardizer
        Control standardizer mapping raw controls to standardized model space.
    provenance : TrainingProvenance
        What the training data was made of; empty on artifacts written before it was recorded.
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
    y_std: Standardizer
    u_std: Standardizer
    provenance: TrainingProvenance = field(default_factory=TrainingProvenance)

    @property
    def model_type(self) -> str:
        """Model architecture type string ('mlp')."""
        return "mlp"

    @property
    def priming_steps(self) -> int:
        """Minimum number of history steps required to initialize the model state."""
        return max(self.n_y, self.n_u)

    @property
    def is_linear(self) -> bool:
        """Whether the MLP is linear (a single layer, i.e. no hidden layers)."""
        return len(self.layers) == 1

    def encode(self, y: FloatArray) -> FloatArray:
        """Map raw EEG ``(..., n_channels)`` into standardized space."""
        return self.y_std.transform(np.asarray(y, dtype=np.float64))

    def decode(self, z: FloatArray) -> FloatArray:
        """Reconstruct raw EEG ``(..., n_channels)`` from standardized space."""
        return self.y_std.inverse_transform(np.asarray(z, dtype=np.float64))

    def prime(self, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
        """Absorb raw history into an initial state (model-space y, **raw** u tail).

        y_hist (k, n_channels), u_hist (k, n_controls), both ending at the same step ``t-1``,
        so :meth:`rollout` predicts from ``t`` onwards.
        """
        y_arr = np.asarray(y_hist, dtype=np.float64)
        u_arr = np.asarray(u_hist, dtype=np.float64)
        z_past = self.encode(y_arr)[-self.n_y :].reshape(-1)
        u_past = u_arr[-self.n_u :].reshape(-1)
        return np.concatenate([z_past, u_past])

    def prime_many(self, y_hists: FloatArray, u_hists: FloatArray) -> FloatArray:
        """Batched :meth:`prime`: ``(B, k, n_channels)`` and ``(B, k, n_controls)`` -> ``(B, state)``."""
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
        """Free-run from ``state`` under raw ``u_future`` (steps, n_controls) -> (steps, n_channels).

        ``u_future[t]`` is the control applied *at* prediction step ``t``, so it first enters the
        window for step ``t + 1`` and the last entry is never consumed. Both windows of a
        :meth:`prime` state already end at the same step, so shifting a control in ahead of the
        first prediction would run the model one control step past its training alignment.
        """
        w_future = self.u_std.transform(np.asarray(u_future, dtype=np.float64))
        steps = len(w_future)

        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = self.u_std.transform(state_arr[n_z:].reshape(self.n_u, self.n_controls))

        preds = np.empty((steps, self.n_channels), dtype=np.float64)
        for t in range(steps):
            y_next = self.forward_1step(y_window.reshape(-1), u_window.reshape(-1))
            y_window = np.concatenate([y_window[1:], y_next[None, :]], axis=0)
            u_window = np.concatenate([u_window[1:], w_future[t][None, :]], axis=0)
            preds[t] = y_next

        return self.decode(preds)

    def rollout_many(self, states: FloatArray, u_futures: FloatArray) -> FloatArray:
        """Batched :meth:`rollout`: ``(B, state)`` and raw ``(B, steps, n_controls)``.

        Returns ``(B, steps, n_channels)``.
        """
        w_future = self.u_std.transform(np.asarray(u_futures, dtype=np.float64))
        steps = w_future.shape[1]

        states_arr = np.asarray(states, dtype=np.float64)
        n_batch = states_arr.shape[0]
        n_z = self.n_y * self.n_channels
        y_window = states_arr[:, :n_z].reshape(n_batch, self.n_y, self.n_channels)
        u_window = self.u_std.transform(states_arr[:, n_z:].reshape(n_batch, self.n_u, self.n_controls))

        preds = np.empty((n_batch, steps, self.n_channels), dtype=np.float64)
        for t in range(steps):
            y_next = self.forward_1step(y_window.reshape(n_batch, -1), u_window.reshape(n_batch, -1))
            y_window = np.concatenate([y_window[:, 1:], y_next[:, None, :]], axis=1)
            u_window = np.concatenate([u_window[:, 1:], w_future[:, t, None, :]], axis=1)
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
            "dt": self.dt,
            "downsample": self.downsample,
            "n_layers": len(self.layers),
            **self.provenance.meta,
        }

    @classmethod
    def load(cls, artifact: str | Path) -> MLPArtifact:
        """Load the single-``.npz`` artifact from disk (``artifact`` is a suffix-less stem)."""
        path = Path(artifact).with_suffix(".npz")
        with np.load(path) as npz:
            meta: dict[str, Any] = json.loads(str(npz["meta"]))
            if meta["activation"] not in ACTIVATIONS:
                msg = f"Unsupported activation: {meta['activation']!r}"
                raise ValueError(msg)
            layers = tuple(
                (
                    np.asarray(npz[f"layer.{i}.weight"], dtype=np.float64),
                    np.asarray(npz[f"layer.{i}.bias"], dtype=np.float64),
                )
                for i in range(int(meta["n_layers"]))
            )
            y_std = Standardizer.from_arrays(npz, "y")
            u_std = Standardizer.from_arrays(npz, "u")

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
            y_std=y_std,
            u_std=u_std,
            provenance=TrainingProvenance.from_meta(meta),
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
        arrays.update(self.y_std.arrays("y"))
        arrays.update(self.u_std.arrays("u"))

        np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
