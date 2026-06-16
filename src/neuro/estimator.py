# ruff: noqa: N806
from __future__ import annotations

import dataclasses
from typing import Any, Self

import numba
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from simulate.estimator import Estimator

from neuro.jansen_rit import JansenRitParams, _heun_step_jit, sigmoid_jit


@numba.njit(fastmath=True, cache=True)
def _fx_step_jit(  # noqa: PLR0913
    x_aug: np.ndarray,
    dt: float,
    u_node: np.ndarray,
    k: int,
    max_history_len: int,
    delay_steps: np.ndarray,
    history: np.ndarray,
    params_tuple: tuple[Any, ...],
) -> np.ndarray:
    """Evaluate the state transition function for a single sigma point.

    This advances the dynamic states of a single augmented state vector by one
    deterministic Heun step, while keeping the parameters (global coupling K and
    connection weights W) constant. It supports connection delays through a circular
    history buffer of regional outputs.

    Parameters
    ----------
    x_aug : np.ndarray
        The augmented state vector.
    dt : float
        Integration step size in seconds.
    u_node : np.ndarray
        The projected tES stimulation current on nodes, of shape (M,).
    k : int
        The current simulation step index.
    max_history_len : int
        The length of the circular history buffer.
    delay_steps : np.ndarray
        Integer delay steps matrix of shape (M, M).
    history : np.ndarray
        Circular buffer containing historical regional outputs, of shape (max_history_len, M).
    params_tuple : tuple of Any
        A tuple of physics parameters for the Jansen-Rit model, formatted by JansenRitParams.to_numba_tuple.

    Returns
    -------
    np.ndarray
        The predicted next augmented state vector.
    """
    n_nodes = delay_steps.shape[0]

    x_dyn = np.reshape(x_aug[: 6 * n_nodes], (6, n_nodes))
    k_coupling = x_aug[6 * n_nodes]
    w_matrix = np.reshape(x_aug[6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2], (n_nodes, n_nodes))

    e0 = params_tuple[8]
    v0 = params_tuple[9]
    r = params_tuple[10]
    s_y_curr = sigmoid_jit(x_dyn[1] - x_dyn[2], e0, v0, r)

    coupling = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        c_i = 0.0
        for j in range(n_nodes):
            delay = delay_steps[i, j]
            if delay == 0:
                c_i += w_matrix[i, j] * s_y_curr[j]
            else:
                row = (k - delay) % max_history_len
                c_i += w_matrix[i, j] * history[row, j]
        coupling[i] = k_coupling * c_i

    if n_nodes == 1:
        x_next = _heun_step_jit(x_dyn, u_node[0], params_tuple, dt, 0.0, coupling[0])
    else:
        xi_arr = np.zeros(n_nodes, dtype=np.float64)
        x_next = _heun_step_jit(x_dyn, u_node, params_tuple, dt, xi_arr, coupling)

    x_aug_next = np.empty_like(x_aug)
    x_aug_next[: 6 * n_nodes] = x_next.flatten()
    x_aug_next[6 * n_nodes :] = x_aug[6 * n_nodes :]
    return x_aug_next


@numba.njit(fastmath=True, cache=True)
def _hx_step_jit(
    x_aug: np.ndarray,
    gain: np.ndarray,
    selected_channels: np.ndarray,
    n_nodes: int,
    estimate_eeg_gains: bool,  # noqa: FBT001
) -> np.ndarray:
    """Evaluate the measurement function for a single sigma point.

    Maps the dynamic state variables of the augmented state vector to scalp EEG
    potentials using the forward operator (gain matrix), then returns only the
    subset of channels that are configured as active measurements.

    Parameters
    ----------
    x_aug : np.ndarray
        The augmented state vector.
    gain : np.ndarray
        The EEG forward operator matrix of shape (n_channels, M).
    selected_channels : np.ndarray
        Indices of active EEG channels to select, of shape (n_selected_channels,).
    n_nodes : int
        The number of nodes M in the Jansen-Rit network.
    estimate_eeg_gains : bool
        If True, projects outputs via estimated diagonal gains.

    Returns
    -------
    np.ndarray
        The projected scalp EEG values at the selected channels, of shape (n_selected_channels,).
    """
    x_dyn = np.reshape(x_aug[: 6 * n_nodes], (6, n_nodes))
    y_node = x_dyn[1] - x_dyn[2]

    if estimate_eeg_gains:
        g_start = 6 * n_nodes + 1 + n_nodes**2
        gains = x_aug[g_start : g_start + n_nodes]
        y_eeg = gains * y_node
    else:
        y_eeg = gain @ y_node

    return y_eeg[selected_channels]


@dataclasses.dataclass(frozen=True)
class UKFEstimatorLog:
    """Dataclass log snapshot of the UKFEstimator state."""

    p_diag: np.ndarray
    K_est: float
    w_est: np.ndarray
    g_est: np.ndarray | None = None


class UKFEstimator(Estimator[UKFEstimatorLog]):
    """Unscented Kalman Filter (UKF) estimator for Jansen-Rit model state and parameters."""

    def __init__(  # noqa: PLR0913, PLR0915, C901, PLR0912
        self,
        dt: float,
        n_nodes: int,
        gain: np.ndarray,
        gamma: np.ndarray | None = None,
        delays: np.ndarray | None = None,
        selected_channels: list[int] | np.ndarray | None = None,
        q_x5: float = 1e-3,
        q_k: float = 1e-5,
        q_w: float = 1e-5,
        r_channel: float = 1e-2,
        p_state: float = 1e-2,
        p_k: float = 1e-2,
        p_w: float = 1e-2,
        initial_k: float = 0.0,
        initial_w: np.ndarray | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        params: JansenRitParams | None = None,
        estimate_eeg_gains: bool = False,  # noqa: FBT001, FBT002
        q_g: float = 1e-5,
        p_g: float = 1e-2,
        initial_g: np.ndarray | None = None,
    ) -> None:
        """Initialize the UKF Estimator.

        This sets up the Unscented Kalman Filter for the Jansen-Rit model, configuring
        augmented state sizes, noise covariances Q and R, initial state covariance P0,
        and setting up circular history buffers for regional delays.

        Parameters
        ----------
        dt : float
            Simulation and estimator time step in seconds.
        n_nodes : int
            Number of regions/nodes in the network.
        gain : np.ndarray
            The scalp EEG forward operator gain matrix of shape (n_channels, n_nodes).
        gamma : np.ndarray or None, optional
            The spatial projection matrix for tES current, of shape (n_electrodes, n_nodes).
        delays : np.ndarray or None, optional
            Conduction delays matrix between regions in milliseconds, of shape (n_nodes, n_nodes).
        selected_channels : list of int or np.ndarray or None, optional
            Indices of active EEG channels to select for observations. If None, all channels are used.
        q_x5 : float, default 1e-3
            Process noise variance for the excitatory-interneuron state (x5') of each node.
        q_k : float, default 1e-5
            Process noise variance for the estimated global coupling scale K.
        q_w : float, default 1e-5
            Process noise variance for the estimated connection weights w_ij.
        r_channel : float, default 1e-2
            Measurement noise variance per active scalp EEG channel.
        p_state : float, default 1e-2
            Initial state variance for each of the hidden dynamic states.
        p_k : float, default 1e-2
            Initial state variance for the global coupling scale K.
        p_w : float, default 1e-2
            Initial state variance for each of the connection weights w_ij.
        initial_k : float, default 0.0
            Initial estimate of the global coupling scale K.
        initial_w : np.ndarray or None, optional
            Initial estimate of the connection weights matrix of shape (n_nodes, n_nodes).
            If None, defaults to the connectome structural weights.
        alpha : float, default 1e-3
            UKF scaling parameter alpha (spread of the sigma points).
        beta : float, default 2.0
            UKF scaling parameter beta (incorporates prior distribution knowledge).
        kappa : float, default 0.0
            UKF scaling parameter kappa.
        params : JansenRitParams or None, optional
            Jansen-Rit model parameters.
        estimate_eeg_gains : bool, default False
            If True, assume diagonal leadfield projection and estimate channel gains online.
        q_g : float, default 1e-5
            Process noise variance for estimated EEG gains.
        p_g : float, default 1e-2
            Initial covariance variance for EEG gains.
        initial_g : np.ndarray or None, optional
            Initial EEG channel gains (defaults to ones).
        """
        super().__init__(dt)
        self.n_nodes = n_nodes
        self.estimate_eeg_gains = estimate_eeg_gains

        self.gain = np.asarray(gain, dtype=np.float64)
        if selected_channels is None:
            self.selected_channels = np.arange(self.gain.shape[0], dtype=np.int64)
        else:
            self.selected_channels = np.asarray(selected_channels, dtype=np.int64)
        self.dim_z = len(self.selected_channels)

        self.gamma_2d = None if gamma is None else np.atleast_2d(gamma)
        self.n_elec = 1 if self.gamma_2d is None else self.gamma_2d.shape[0]

        self.dim_x = 6 * n_nodes + 1 + n_nodes**2
        if estimate_eeg_gains:
            self.dim_x += n_nodes

        if delays is None:
            delays = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self.delay_steps = np.round(delays / (dt * 1000.0)).astype(np.int64)
        self.max_history_len = int(np.max(self.delay_steps)) + 1

        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)

        if params is None:
            params = JansenRitParams()
        a_vec = params.A
        if np.isscalar(a_vec):
            a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
        self.net_params = dataclasses.replace(params, A=a_vec)
        self.params_tuple = self.net_params.to_numba_tuple(n_nodes)

        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=alpha, beta=beta, kappa=kappa)

        def fx(x_aug: np.ndarray, dt_step: float, u_node: np.ndarray | float, k: int) -> np.ndarray:
            if isinstance(u_node, np.ndarray):
                if u_node.ndim == 0:
                    val = float(u_node.item())
                    u_node_arr = np.full(self.n_nodes, val, dtype=np.float64)
                else:
                    u_node_arr = np.asarray(u_node, dtype=np.float64)
            else:
                u_node_arr = np.full(self.n_nodes, float(u_node), dtype=np.float64)

            return _fx_step_jit(
                x_aug,
                dt_step,
                u_node_arr,
                k,
                self.max_history_len,
                self.delay_steps,
                self.history,
                self.params_tuple,
            )

        def hx(x_aug: np.ndarray) -> np.ndarray:
            return _hx_step_jit(
                x_aug,
                self.gain,
                self.selected_channels,
                self.n_nodes,
                self.estimate_eeg_gains,
            )

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=dt,
            hx=hx,
            fx=fx,
            points=points,
        )

        # Set initial state
        x0_aug = np.zeros(self.dim_x, dtype=np.float64)
        x0_aug[6 * n_nodes] = initial_k
        if initial_w is not None:
            x0_aug[6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2] = initial_w.flatten()

        current_idx = 6 * n_nodes + 1 + n_nodes**2
        if estimate_eeg_gains:
            if initial_g is not None:
                x0_aug[current_idx : current_idx + n_nodes] = initial_g
            else:
                x0_aug[current_idx : current_idx + n_nodes] = 1.0

        self.ukf.x = x0_aug

        # Seed history
        x_dyn0 = x0_aug[: 6 * n_nodes].reshape(6, n_nodes)
        s_y0 = sigmoid_jit(x_dyn0[1] - x_dyn0[2], self.net_params.e0, self.net_params.v0, self.net_params.r)
        self.history[:, :] = s_y0

        # initial covariance P0
        P0 = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        P0[: 6 * n_nodes, : 6 * n_nodes] = p_state * np.eye(6 * n_nodes)
        P0[6 * n_nodes, 6 * n_nodes] = p_k
        P0[6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2, 6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2] = (
            p_w * np.eye(n_nodes**2)
        )

        current_idx = 6 * n_nodes + 1 + n_nodes**2
        if estimate_eeg_gains:
            P0[current_idx : current_idx + n_nodes, current_idx : current_idx + n_nodes] = p_g * np.eye(n_nodes)

        self.ukf.P = P0

        # process noise Q
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        for i in range(n_nodes):
            Q[6 * i + 4, 6 * i + 4] = q_x5
        Q[6 * n_nodes, 6 * n_nodes] = q_k
        Q[6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2, 6 * n_nodes + 1 : 6 * n_nodes + 1 + n_nodes**2] = (
            q_w * np.eye(n_nodes**2)
        )

        current_idx = 6 * n_nodes + 1 + n_nodes**2
        if estimate_eeg_gains:
            Q[current_idx : current_idx + n_nodes, current_idx : current_idx + n_nodes] = q_g * np.eye(n_nodes)

        self.ukf.Q = Q

        # measurement noise R
        self.ukf.R = r_channel * np.eye(self.dim_z)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:  # noqa: PLR0915
        """Instantiate the UKFEstimator from a raw configuration dictionary.

        Loads structural connectome connectivity using `speed`, optionally subsets the
        network down to `n_nodes`, resolves channel string/integer mappings from
        `selected_channels`, and reads all UKF noise/scaling parameters.

        Parameters
        ----------
        config : dict of str to Any
            The configuration dictionary containing estimator settings.

        Returns
        -------
        UKFEstimator
            The instantiated UKFEstimator object.
        """
        from neuro.connectome import load_connectome  # noqa: PLC0415

        dt = float(config["dt"])
        speed = float(config.get("speed", 50.0))
        connectome = load_connectome(speed=speed)

        n_nodes_val = config.get("n_nodes")
        if n_nodes_val is not None:
            n_nodes = int(n_nodes_val)
            from dataclasses import replace  # noqa: PLC0415

            # TODO: smarter reduction of connectome
            connectome = replace(
                connectome,
                weights=connectome.weights[:n_nodes, :n_nodes],
                tract_lengths=connectome.tract_lengths[:n_nodes, :n_nodes],
                centres=connectome.centres[:n_nodes],
                region_labels=connectome.region_labels[:n_nodes],
                hemispheres=connectome.hemispheres[:n_nodes],
                delays=connectome.delays[:n_nodes, :n_nodes],
                gain=connectome.gain[:, :n_nodes],
                region_index={label: idx for idx, label in enumerate(connectome.region_labels[:n_nodes])},
            )

        n_nodes = connectome.weights.shape[0]
        gain = connectome.gain
        delays = connectome.delays

        selected_channels_cfg = config.get("selected_channels")
        selected_channels = None
        if selected_channels_cfg is not None:
            selected_channels = []
            for ch in selected_channels_cfg:
                if isinstance(ch, str):
                    selected_channels.append(connectome.channel_index[ch])
                else:
                    selected_channels.append(int(ch))
            selected_channels = np.array(selected_channels, dtype=np.int64)

        target_electrode = config.get("target_electrode")
        gamma = connectome.gamma
        if target_electrode is not None:
            if isinstance(target_electrode, (str, int, np.integer)):
                electrodes_list = [target_electrode]
            else:
                electrodes_list = list(target_electrode)

            resolved_electrodes = []
            for el in electrodes_list:
                if isinstance(el, (int, np.integer)):
                    resolved_electrodes.append(str(connectome.channel_labels[int(el)]))
                else:
                    resolved_electrodes.append(str(el))

            from neuro.connectome import compute_gamma  # noqa: PLC0415

            gamma = compute_gamma(
                connectome.centres,
                target_electrode=resolved_electrodes
                if not isinstance(target_electrode, (str, int, np.integer))
                else resolved_electrodes[0],
                sigma=config.get("gamma_sigma", 20.0),
            )

        params_config = config.get("params", {})
        params = JansenRitParams.from_config(params_config)

        q_x5 = float(config.get("q_x5", 1e-3))
        q_K = float(config.get("q_K", 1e-5))
        q_w = float(config.get("q_w", 1e-5))
        r_channel = float(config.get("r_channel", 1e-2))

        p_state = float(config.get("p_state", 1e-2))
        p_K = float(config.get("p_K", 1e-2))
        p_w = float(config.get("p_w", 1e-2))

        initial_K = float(config.get("initial_K", config.get("K", 0.0)))

        initial_w = config.get("initial_w")
        initial_w = np.asarray(initial_w, dtype=np.float64) if initial_w is not None else connectome.weights.copy()

        alpha = float(config.get("alpha", 1e-3))
        beta = float(config.get("beta", 2.0))
        kappa = float(config.get("kappa", 0.0))

        estimate_eeg_gains = bool(config.get("estimate_eeg_gains", False))

        q_g = float(config.get("q_g", 1e-5))
        p_g = float(config.get("p_g", 1e-2))

        initial_g = config.get("initial_g")
        if initial_g is not None:
            initial_g = np.asarray(initial_g, dtype=np.float64)

        return cls(
            dt=dt,
            n_nodes=n_nodes,
            gain=gain,
            gamma=gamma,
            delays=delays,
            selected_channels=selected_channels,
            q_x5=q_x5,
            q_k=q_K,
            q_w=q_w,
            r_channel=r_channel,
            p_state=p_state,
            p_k=p_K,
            p_w=p_w,
            initial_k=initial_K,
            initial_w=initial_w,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            params=params,
            estimate_eeg_gains=estimate_eeg_gains,
            q_g=q_g,
            p_g=p_g,
            initial_g=initial_g,
        )

    def _project_control(self, u: float | np.ndarray) -> np.ndarray | float:
        """Project per-electrode tES current ``u`` onto nodes via ``gamma``."""
        u_vec = np.asarray(u, dtype=np.float64).reshape(-1)
        if not np.any(u_vec):
            return 0.0
        if self.gamma_2d is None:
            msg = "tES stimulation is active but connectome.gamma is not configured."
            raise ValueError(msg)
        if u_vec.size == 1:
            u_vec = np.broadcast_to(u_vec, (self.n_elec,))
        elif u_vec.size != self.n_elec:
            msg = f"control has {u_vec.size} electrodes but gamma has {self.n_elec}"
            raise ValueError(msg)
        return u_vec @ self.gamma_2d

    def update(
        self,
        t: float,
        y_mea: float | np.ndarray,
        u: float | np.ndarray,
    ) -> tuple[float | np.ndarray, UKFEstimatorLog]:
        """Run one step of state transition prediction and measurement update.

        This performs the predict step (advancing dynamic states using a deterministic
        Heun step and history-based delays) and the update step (updating augmented states
        using active EEG channel measurements), enforces physical parameter bounds/zero-diagonal
        constraints, records history, and returns the full estimated augmented state and log.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.
        y_mea : float or np.ndarray
            The full measured scalp EEG channel vector of shape (n_channels,).
        u : float or np.ndarray
            The per-electrode control input (tES currents) of shape (n_electrodes,).

        Returns
        -------
        x_hat : np.ndarray
            The full estimated augmented state vector.
        log : UKFEstimatorLog
            A log containing estimates.
        """
        u_node = self._project_control(u)
        k = round(t / self.dt)

        self.ukf.predict(u_node=u_node, k=k)

        y_mea_vec = np.atleast_1d(y_mea)
        if y_mea_vec.shape[0] == self.dim_z:
            y_mea_sub = y_mea_vec
        elif y_mea_vec.shape[0] == self.gain.shape[0]:
            y_mea_sub = y_mea_vec[self.selected_channels]
        else:
            msg = f"y_mea shape {y_mea_vec.shape} matches neither dim_z {self.dim_z} nor full channels {self.gain.shape[0]}"
            raise ValueError(msg)
        self.ukf.update(y_mea_sub)

        # Enforce bounds
        self.ukf.x[6 * self.n_nodes] = max(0.0, float(self.ukf.x[6 * self.n_nodes]))

        W_slice = self.ukf.x[6 * self.n_nodes + 1 : 6 * self.n_nodes + 1 + self.n_nodes**2]
        W = W_slice.reshape(self.n_nodes, self.n_nodes)
        np.clip(W, 0.0, None, out=W)
        np.fill_diagonal(W, 0.0)
        self.ukf.x[6 * self.n_nodes + 1 : 6 * self.n_nodes + 1 + self.n_nodes**2] = W.flatten()

        current_idx = 6 * self.n_nodes + 1 + self.n_nodes**2
        g_est = None
        if self.estimate_eeg_gains:
            g = self.ukf.x[current_idx : current_idx + self.n_nodes]
            np.clip(g, 0.0, None, out=g)
            self.ukf.x[current_idx : current_idx + self.n_nodes] = g
            g_est = g.copy()

        x_dyn = self.ukf.x[: 6 * self.n_nodes].reshape(6, self.n_nodes)
        s_y = sigmoid_jit(x_dyn[1] - x_dyn[2], self.net_params.e0, self.net_params.v0, self.net_params.r)
        self.history[k % self.max_history_len, :] = s_y

        log = UKFEstimatorLog(
            p_diag=np.diag(self.ukf.P).copy(),
            K_est=float(self.ukf.x[6 * self.n_nodes]),
            w_est=W.copy(),
            g_est=g_est,
        )

        return self.ukf.x.copy(), log
