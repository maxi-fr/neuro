from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Self

import casadi as ca
import numpy as np

from neuro.esn import ESNArtifact
from neuro.transforms import unzscore, zscore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import scipy.sparse


def _scipy_csr_to_casadi(mat: scipy.sparse.csr_matrix) -> ca.DM:
    """Convert a scipy CSR matrix to a CasADi sparse DM matrix."""
    coo = mat.tocoo()
    return ca.DM.triplet(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), mat.shape[0], mat.shape[1])


@dataclass(frozen=True)
class ESNSymbolicModel:
    """CasADi-symbolic bridge for the ESN predictor.

    Attributes
    ----------
    artifact : ESNArtifact
        The loaded ESN artifact containing weight matrices, pipelines, and metadata.
    """

    artifact: ESNArtifact

    @classmethod
    def from_artifact(cls, artifact: str | Path) -> Self:
        """Build the symbolic model by loading an ESN artifact from disk."""
        return cls(ESNArtifact.load(artifact))

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of the reservoir state vector h as a column vector (N, 1)."""
        return (self.artifact.reservoir_size, 1)

    history_depth: int = 0

    @property
    def n_controls(self) -> int:
        """Number of control input channels m."""
        return self.artifact.n_controls

    @property
    def n_channels(self) -> int:
        """Number of model-space y output channels C."""
        return self.artifact.n_channels

    @property
    def n_eeg_channels(self) -> int:
        """Number of raw EEG output channels."""
        return self.artifact.n_eeg_channels

    @property
    def free_syms(self) -> dict[str, ca.MX]:
        """Free symbolic parameters; always empty -- this model is purely numeric."""
        return {}

    @cached_property
    def _w_res_dm(self) -> ca.DM:
        """Sparse CasADi DM matrix for W_res."""
        return _scipy_csr_to_casadi(self.artifact.w_res)

    @cached_property
    def _w_in_dm(self) -> ca.DM:
        """Dense CasADi DM matrix for W_in."""
        return ca.DM(self.artifact.w_in)

    @cached_property
    def _w_out_dm(self) -> ca.DM:
        """Dense CasADi DM matrix for W_out."""
        return ca.DM(self.artifact.w_out)

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance reservoir state h by one free-running step under raw control u.

        Parameters
        ----------
        history : Sequence[ca.SX | ca.MX]
            Single-element sequence containing current state h, shape (N, 1).
        u : ca.SX | ca.MX
            Raw control input, shape (n_controls, 1).

        Returns
        -------
        h_next : ca.SX | ca.MX
            Next reservoir state, shape (N, 1).
        """
        (h,) = history
        u_std = self.artifact.u_pipeline.standardizer
        if u_std is None:
            msg = "u-pipeline must carry a standardizer"
            raise ValueError(msg)

        n_ctrl = self.n_controls
        u_center = np.broadcast_to(u_std.center, (n_ctrl,)).reshape(-1, 1)
        u_scale = np.broadcast_to(u_std.scale, (n_ctrl,)).reshape(-1, 1)
        v = zscore(u, u_center, u_scale)  # ty:ignore[invalid-argument-type]

        h_aug = ca.vertcat(h, 1.0)
        z_hat = ca.mtimes(self._w_out_dm, h_aug)
        in_vec = ca.vertcat(z_hat, v, 1.0)

        alpha = self.artifact.leak_rate
        net = ca.mtimes(self._w_res_dm, h) + ca.mtimes(self._w_in_dm, in_vec)
        return (1.0 - alpha) * h + alpha * ca.tanh(net)

    def output(self, h: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Decode readout prediction from reservoir state h to raw EEG space."""
        h_aug = ca.vertcat(h, 1.0)
        z_hat = ca.mtimes(self._w_out_dm, h_aug)

        pca = self.artifact.y_pipeline.pca
        y_std = ca.mtimes(pca.basis.T, z_hat) + pca.mean.reshape(-1, 1) if pca is not None else z_hat

        std = self.artifact.y_pipeline.standardizer
        if std is None:
            msg = "y-pipeline must carry a standardizer"
            raise ValueError(msg)
        n_eeg = self.n_eeg_channels
        center = np.broadcast_to(std.center, (n_eeg,)).reshape(-1, 1)
        scale = np.broadcast_to(std.scale, (n_eeg,)).reshape(-1, 1)
        return unzscore(y_std, center, scale)

    @cached_property
    def f_step(self) -> ca.Function:
        """Reusable compiled single-step function (h, u) -> h_next."""
        h_sym = ca.MX.sym("h", *self.state_shape)
        u_sym = ca.MX.sym("u", self.n_controls, 1)
        return ca.Function("F_step_esn", [h_sym, u_sym], [self.step([h_sym], u_sym)])

    @cached_property
    def f_out(self) -> ca.Function:
        """Reusable compiled output function h -> y."""
        h_sym = ca.MX.sym("h", *self.state_shape)
        return ca.Function("F_out_esn", [h_sym], [self.output(h_sym)])
