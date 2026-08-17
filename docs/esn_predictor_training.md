# ESN Predictor — Training Setup

A detailed, end-to-end description of how the Echo State Network (ESN) EEG predictor is
configured, harvested, and trained. The predictor is a recurrent reservoir model with a closed-form
linear readout solved via ridge regression, designed to operate in raw model space or latent PCA space,
and integrated into CasADi for model predictive control (MPC).

Source of truth:

- Training pipeline & data prep: [`src/neuro/esn_training.py`](../src/neuro/esn_training.py)
- Model + artifact + inference: [`src/neuro/esn.py`](../src/neuro/esn.py)
- CasADi symbolic integration: [`src/neuro/esn_predictor_casadi.py`](../src/neuro/esn_predictor_casadi.py)
- Config schema: [`src/neuro/config.py`](../src/neuro/config.py)
- Entry points: [`scripts/train_esn.py`](../scripts/train_esn.py),
  [`scripts/sweep_esn.py`](../scripts/sweep_esn.py)
- Example config: [`configs/nn_predictor/esn_8s.yaml`](../configs/nn_predictor/esn_8s.yaml)
- Neural network predictor reference: [`docs/nn_predictor_training.md`](nn_predictor_training.md)

> All array computations run in **float64** precision (`np.float64`).

---

## 1. Problem statement

Let

- $y_t \in \mathbb{R}^{C}$ — measured EEG at (downsampled) time step $t$, $C = n_\text{ch}$ channels
  (the Jansen–Rit sensor EEG has $C = 62$ channels; EEG is in arbitrary units, not µV).
- $u_t \in \mathbb{R}^{m}$ — stimulation / control input at step $t$, $m = n_\text{controls}$ electrodes.

We want a predictor capable of multi-step-ahead forecasting over a horizon of $N$ steps:

$$
\hat{y}_{t+1:t+N} \;=\; F_\text{ESN}\!\big(\,\underbrace{y_{t-W+1:t}}_{\text{past EEG history}},\;
\underbrace{u_{t-W+1:t-1}}_{\text{past control history}},\;
\underbrace{u_{t:t+N-1}}_{\text{future control commands}}\,\big),
$$

where $W = \texttt{washout}$ is the length of the history window used to prime the reservoir state.

Unlike the MLP predictor ([`nn_predictor_training.md`](nn_predictor_training.md)), which relies on an explicit sliding-window concatenation of $n_y$ past EEG steps and $n_u$ past controls fed into a feedforward network $f_\theta$, the Echo State Network maintains a high-dimensional internal reservoir state $h_t \in \mathbb{R}^{N_\text{res}}$ ($N_\text{res} \gg C$, typically $500 \text{--} 2000$ units) that updates recursively over time:

1. **Teacher-forcing step (history priming):**
   $$
   h_{t+1} = (1 - \alpha)\, h_t + \alpha \tanh\big(W_\text{res} h_t + W_\text{in} [z_t;\, v_t;\, 1]\big)
   $$
   where $z_t \in \mathbb{R}^{d_y}$ is the model-space target EEG (dimension $d_y = k$ under PCA projection, else $d_y = C$), $v_t \in \mathbb{R}^m$ is the standardized control input, and $\alpha = \texttt{leak\_rate} \in (0, 1]$ is the leak rate.

2. **Readout step (one-step prediction):**
   $$
   \hat{z}_{t+1} = W_\text{out} \begin{bmatrix} h_{t+1} \\ 1 \end{bmatrix} \in \mathbb{R}^{d_y}
   $$

3. **Free-running step (autoregressive rollout):**
   $$
   h_{t+1} = (1 - \alpha)\, h_t + \alpha \tanh\big(W_\text{res} h_t + W_\text{in} [\hat{z}_t;\, v_t;\, 1]\big)
   $$
   where the model's own previous prediction $\hat{z}_t$ replaces ground-truth target $z_t$.

The internal reservoir weights $(W_\text{res}, W_\text{in})$ are generated randomly and kept fixed; only the linear readout matrix $W_\text{out} \in \mathbb{R}^{d_y \times (N_\text{res} + 1)}$ is learned from data.

### 1.1 Indexing notation conventions

To avoid ambiguity between global time-series progression, dataset windowing, and algorithmic loops:

| index | range / domain | scope & meaning |
| ----- | -------------- | --------------- |
| $t$   | $0, \dots, T-1$ | Continuous time step index along a full trajectory recording |
| $k$   | $0, \dots, W-1$ | Step index along a history window during reservoir priming |
| $j$   | $0, \dots, N-1$ | Relative step index during sequential free-running rollout unrolling ($\hat{z}_j$) |
| $i$   | $0, \dots, N-1$ | Relative horizon step index in algebraic loss and metric summations ($\text{NMSE}_i$) |
| $b$   | $0, \dots, M_\text{val}-1$ | Window / sample index within an evaluation dataset |
| $c$   | $0, \dots, C-1$ | EEG channel index ($C = n_\text{ch} = 62$) |

---

## 2. Training data

### 2.1 Where it comes from

Each trajectory is an `.npz` file produced by the tES simulation experiments (e.g. `data/experiment_excited_roast_8s/train/` or `data/experiment_excited/train/`). Files are discovered and sorted alphabetically by [`resolve_data_files`](../src/neuro/config.py). Each file contains two key arrays:

| key     | meaning             | shape    |
| ------- | ------------------- | -------- |
| `y_mea` | measured EEG output | $(T, C)$ |
| `u`     | stimulation input   | $(T, m)$ |

The stimulation sequence is persistently exciting broadband tES satisfying **Kirchhoff's current law** ($\sum_{i=1}^m u_{t,i} = 0$ across the $m$ electrodes at every step $t$).

### 2.2 Loading and downsampling

[`load_trajectory`](../src/neuro/predictor/data.py) loads up to `n_steps · downsample` raw samples (or the full file if `n_steps` is `null`) and decimates by taking every `downsample`-th sample:

$$
y^{\text{ds}}_k = y^{\text{raw}}_{k\cdot d}, \qquad
u^{\text{ds}}_k = u^{\text{raw}}_{k\cdot d}, \qquad k = 0,\dots,n_\text{steps}-1,
$$

with $d = \texttt{downsample}$. The effective sample step is

$$
\Delta t_\text{real} = \Delta t \cdot d.
$$

Example configuration (`configs/nn_predictor/esn_8s.yaml`): $\Delta t = 10^{-4}\,\text{s}$, $d = 200 \Rightarrow \Delta t_\text{real} = 2 \cdot 10^{-2}\,\text{s}$ (50 Hz), $n_\text{steps} = \texttt{null}$ (full 8 s trajectory = 400 downsampled steps). This $\Delta t_\text{real}$ is stored in the artifact metadata and used during inference.

---

## 3. Trajectory reservoir driving & state harvesting

### 3.1 Model-space dimensions and PCA projection

Let $d_y$ denote the dimension of the model-space EEG representation:

- If `model.latent_dim = k` is set ($k < C$), `y_pipeline` gains a fixed orthonormal PCA projection step after standardization. [`PCAProjection.fit`](../src/neuro/transforms.py) fits basis $E \in \mathbb{R}^{k \times C}$ and mean $\mu \in \mathbb{R}^{C}$ on standardized training EEG ($\tilde{y}_t \in \mathbb{R}^C$):
  $$
  z_t = (\tilde{y}_t - \mu) E^\top \in \mathbb{R}^k, \qquad \tilde{y}_t = z_t E + \mu \quad (\text{decode}),
  $$
  so $d_y = k$.
- If `model.latent_dim` is `null` (projection disabled), model space is direct standardized channel space $\tilde{y}_t$, so $z_t \equiv \tilde{y}_t \in \mathbb{R}^C$ and $d_y = C$.

The control inputs $u_t \in \mathbb{R}^m$ are pushed through `u_pipeline` (standardized) yielding $v_t \in \mathbb{R}^m$.

### 3.2 Continuous trajectory driving & target alignment

Unlike feedforward neural networks that shuffle sliding windows of inputs, an ESN is driven **continuously** along full trajectory recordings:

1. For each trajectory $(u_\text{raw}, y_\text{raw})$, compute model-space target sequence $z \in \mathbb{R}^{T \times d_y}$ and standardized control sequence $v \in \mathbb{R}^{T \times m}$.
2. Initialize reservoir state to zero: $h_0 = 0 \in \mathbb{R}^{N_\text{res}}$.
3. **Target Alignment & Timing:** At step $t$, state vector $h_t$ is recorded **before** absorbing input $(z_t, v_t)$, and paired with target $z_t$. This timing guarantees that $\hat{z}_t = W_\text{out} [h_t; 1]$ is a genuine **one-step-ahead prediction** of step $t$ rather than a state reconstruction.
4. **Teacher-Forcing State Update:** The state is advanced to $h_{t+1}$:
   $$
   h_{t+1} = (1 - \alpha)\, h_t + \alpha \tanh\big(W_\text{res} h_t + W_\text{in} [z_{\text{in},t};\, v_t;\, 1]\big),
   $$
   where $z_{\text{in},t} = z_t + \sigma \cdot \xi_t$ incorporates optional harvesting noise ($\sigma = \texttt{noise\_sigma}$, $\xi_t \sim \mathcal{N}(0, I_{d_y})$).
5. **Washout Period:** Initial $W = \texttt{washout}$ steps of each trajectory are discarded to allow transient dynamics from $h_0 = 0$ to decay. Only states and targets for $t \ge W$ are harvested into normal equation statistics.

---

## 4. Preprocessing: split and scaling

### 4.1 Train / validation split

Trajectories are split by file order prior to any harvesting ([`split_data_files`](../src/neuro/predictor/data.py)):

$$
n_\text{train} = \lfloor \texttt{train\_split} \cdot n_\text{files} \rfloor, \qquad
\text{train} = \text{files}_{0:n_\text{train}}, \qquad
\text{val} = \text{files}_{n_\text{train}:},
$$

ensuring whole trajectories remain intact and avoiding data leakage.

### 4.2 Transforms (pipelines)

Feature scalers are fit exclusively on the training split ([`prepare_training_data`](../src/neuro/esn_training.py)):

- **`y_pipeline`**: Contains a [`Standardizer`](../src/neuro/transforms.py) (and optionally a [`PCAProjection`](../src/neuro/transforms.py) if `latent_dim` is specified).
- **`u_pipeline`**: Contains a [`Standardizer`](../src/neuro/transforms.py) for control inputs.

Supported scaling methods:

- `scaler = "standard"`: subtract mean, divide by standard deviation.
- `scaler = "robust"`: subtract median, divide by interquartile range (IQR).
- `global_scaling = true`: single pooled scalar across all channels; `false`: per-channel scaling.

---

## 5. Model architecture & dynamic equations

### 5.1 Reservoir generation

Function [`generate_reservoir`](../src/neuro/esn.py) constructs the static weight matrices given PRNG seed `training.seed`:

1. **Reservoir matrix $W_\text{res} \in \mathbb{R}^{N_\text{res} \times N_\text{res}}$:**
   - Generated as a sparse SciPy `csr_matrix` with density $d_\text{res} = \texttt{density}$.
   - Non-zero entries drawn from uniform distribution $\mathcal{U}(-1, 1)$.
   - Spectral radius $\lambda_\text{max} = \max |\lambda(W_\text{res,raw})|$ is computed via `scipy.sparse.linalg.eigs` (or dense eigenvalue decomposition fallback).
   - Matrix is rescaled to target spectral radius $\rho = \texttt{spectral\_radius}$:
     $$
     W_\text{res} = W_\text{res,raw} \cdot \frac{\rho}{\lambda_\text{max}}.
     $$

2. **Input weight matrix $W_\text{in} \in \mathbb{R}^{N_\text{res} \times (d_y + m + 1)}$:**
   - Dense matrix with elements sampled independently from $\mathcal{U}(-\gamma, \gamma)$, where $\gamma = \texttt{input\_scaling}$.
   - Input vector block structure: $[z_t;\, v_t;\, 1] \in \mathbb{R}^{d_y + m + 1}$, representing current model-space EEG, model-space control, and constant unit bias.

### 5.2 Priming & rollout interface

[`ESNArtifact`](../src/neuro/esn.py) implements state absorption and forecasting:

- **`prime(y_hist, u_hist)`**:
  Transforms raw history $(y_\text{hist}, u_\text{hist})$ of length $W$ into model space $(z_\text{hist}, v_\text{hist})$. Starting from $h_0 = 0$, advances state through $W$ teacher-forcing steps:
  $$
  h_{k+1} = (1 - \alpha) h_k + \alpha \tanh\big(W_\text{res} h_k + W_\text{in} [z_k;\, v_k;\, 1]\big),
  $$
  returning the primed state vector $h_W \in \mathbb{R}^{N_\text{res}}$.

- **`rollout(state, u_future)`**:
  Given primed state $h_0 = \text{state}$ and future control sequence $u_\text{future}$ of length $N$:
  For step $j = 0, \dots, N-1$:
  1. Compute model-space readout prediction:
     $$
     \hat{z}_{j} = W_\text{out} \begin{bmatrix} h_j \\ 1 \end{bmatrix} \in \mathbb{R}^{d_y}
     $$
  2. Advance reservoir state with self-fed prediction:
     $$
     h_{j+1} = (1 - \alpha) h_j + \alpha \tanh\big(W_\text{res} h_j + W_\text{in} [\hat{z}_j;\, v_{j};\, 1]\big)
     $$
  3. Reconstruct raw EEG predictions $\hat{y}_{1:N}$ via `y_pipeline.inverse_transform(preds_z)`.

### 5.3 CasADi symbolic integration

For Model Predictive Control (MPC), [`ESNSymbolicModel`](../src/neuro/esn_predictor_casadi.py) exposes symbolic CasADi functions:

- Converts sparse $W_\text{res}$ into a CasADi sparse `DM` matrix using coordinate triplets.
- Converts dense $W_\text{in}$ and $W_\text{out}$ into CasADi `DM` matrices.
- `f_step(h, u)`: CasADi `Function("F_step_esn", [h, u], [h_next])` performing one free-running reservoir step under raw control $u$.
- `f_out(h)`: CasADi `Function("F_out_esn", [h], [y])` decoding state $h$ to raw EEG prediction $y$.

---

## 6. Normal equations & Ridge regression optimization

### 6.1 State matrix harvesting

Instead of storing giant concatenated state matrices in memory, [`harvest_normal_equations`](../src/neuro/esn.py) continuously accumulates the normal equation statistics across all training trajectories:

- State covariance matrix sum $G \in \mathbb{R}^{(N_\text{res}+1) \times (N_\text{res}+1)}$:
  $$
  G = \sum_{\text{traj}} \sum_{t=W}^{T-1} \begin{bmatrix} h_t \\ 1 \end{bmatrix} \begin{bmatrix} h_t \\ 1 \end{bmatrix}^\top
  $$
- Cross-covariance matrix sum $P \in \mathbb{R}^{(N_\text{res}+1) \times d_y}$:
  $$
  P = \sum_{\text{traj}} \sum_{t=W}^{T-1} \begin{bmatrix} h_t \\ 1 \end{bmatrix} z_t^\top
  $$

This accumulation runs in $O(T_\text{total} \cdot N_\text{res}^2)$ time and requires constant $O(N_\text{res}^2 + N_\text{res} d_y)$ memory.

### 6.2 Solving ridge regression

[`solve_ridge`](../src/neuro/esn.py) computes the optimal linear readout matrix $W_\text{out} \in \mathbb{R}^{d_y \times (N_\text{res}+1)}$ in closed form:

$$
W_\text{out}^\top = \big(G + \lambda I_\text{unreg}\big)^{-1} P
$$

where $\lambda = \texttt{ridge\_lambda}$ is the $L_2$ regularization parameter, and $I_\text{unreg} = \operatorname{diag}(1, 1, \dots, 1, 0) \in \mathbb{R}^{(N_\text{res}+1) \times (N_\text{res}+1)}$ is the identity matrix with the **last entry zeroed out** so the bias term is **unregularized**.

The linear system is solved via `np.linalg.solve(G + reg, P)`, yielding $W_\text{out} \in \mathbb{R}^{d_y \times (N_\text{res}+1)}$.

---

## 7. Evaluation & metrics

Carrying notation through in **raw EEG units**:

| symbol | meaning |
| ------ | ------- |
| $y_t \in \mathbb{R}^C$ | raw measured EEG sample at step $t$ |
| $Y_{b,i,c}$ | ground-truth raw EEG for validation window $b$, horizon step $i$, channel $c$ |
| $\hat{Y}_{b,i,c}$ | free-running ESN rollout prediction starting from window $b$'s primed history |

Evaluation is performed on held-out validation trajectories using [`evaluate_rollouts`](../src/neuro/artifacts.py):

1. Validation trajectories are windowed on a stride-25 grid.
2. For each window $b$, past history of length $W = \texttt{washout}$ is passed to `art.prime(y_hist, u_hist)` to obtain state $h_b$.
3. `art.rollout(h_b, u_future)` computes the free-running forecast $\hat{Y}_b \in \mathbb{R}^{N \times C}$.
4. **Normalized Mean Squared Error (NMSE)** is computed using the energy of the true signal over the index set:

$$
\text{NMSE}_i = \frac{\sum_{b,c} \big(Y_{b,i,c} - \hat{Y}_{b,i,c}\big)^2}{\sum_{b,c} Y_{b,i,c}^2},
\qquad
\text{NMSE} = \frac{\sum_i \sum_{b,c} \big(Y_{b,i,c} - \hat{Y}_{b,i,c}\big)^2}{\sum_i \sum_{b,c} Y_{b,i,c}^2}.
$$

An $\text{NMSE} = 1.0$ corresponds to a zero-output baseline.

---

## 8. Artifacts & serialization

[`train_esn.py`](../scripts/train_esn.py) writes the trained model into the designated artifact directory (e.g. `artifacts/esn_8s/`):

| file                  | contents                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `model.json`          | Metadata dictionary (`dt`, `downsample`, `horizon`, `reservoir_size`, `leak_rate`, `spectral_radius`, `washout`, `input_scaling`, `density`, `noise_sigma`, `ridge_lambda`, `seed`, pipeline steps) |
| `model.scalers.npz`   | Pipeline transformation parameters (`y` and `u` standardizers + PCA basis/mean)            |
| `model.weights.npz`   | Weight matrices `W_in`, `W_out`, and CSR sparse components `W_res.data`, `W_res.indices`, `W_res.indptr`, `W_res.shape` |
| `training_stats.json` | Execution timings (`harvest_seconds`, `fit_seconds`), pooled `nmse_rollout`, and per-step `nmse_rollout_per_step` |
| `config.yaml`         | Copy of resolved configuration YAML for full provenance                                     |

The artifact is loaded via [`ESNArtifact.load(artifact_path)`](../src/neuro/esn.py).

---

## 9. Configuration reference

Configurations are declared in YAML files and validated using Pydantic ([`ESNPredictorConfig`](../src/neuro/config.py)).

### `simulation`

| key          | type   | default | meaning                                                |
| ------------ | ------ | ------- | ------------------------------------------------------ |
| `dt`         | float  | `1e-4`  | Raw simulation step (s)                                |
| `downsample` | int    | `100`   | Decimation factor $d$                                  |
| `n_steps`    | int    | `null`  | Raw steps loaded per trajectory (`null` $\Rightarrow$ all) |
| `data_path`  | string | required| Directory containing `.npz` trajectory files           |

### `model`

| key               | type   | default | meaning                                                 |
| ----------------- | ------ | ------- | ------------------------------------------------------- |
| `reservoir_size`  | int    | `500`   | Number of reservoir units $N_\text{res}$               |
| `spectral_radius` | float  | `0.9`   | Target spectral radius $\rho$ for $W_\text{res}$        |
| `leak_rate`       | float  | `0.1`   | Reservoir leakage rate $\alpha \in (0, 1]$              |
| `density`         | float  | `0.1`   | Sparsity density $d_\text{res} \in (0, 1]$ of $W_\text{res}$ |
| `input_scaling`   | float  | `0.1`   | Input weight scaling $\gamma$ for $W_\text{in}$         |
| `washout`         | int    | `100`   | Washout step count $W$                                  |
| `ridge_lambda`    | float  | `1e-3`  | $L_2$ regularization $\lambda$ for readout solve         |
| `noise_sigma`     | float  | `0.0`   | Std dev $\sigma$ of Gaussian noise added during harvesting |
| `horizon`         | int    | `50`    | Direct forecasting rollout horizon $N$                  |
| `latent_dim`      | int    | `null`  | PCA component count $k$ (`null` $\Rightarrow$ full channels) |

### `training`

| key              | type   | default    | meaning                                            |
| ---------------- | ------ | ---------- | -------------------------------------------------- |
| `train_split`    | float  | `0.8`      | Fraction of data files used for training           |
| `seed`           | int    | `69`       | PRNG seed for reservoir generation and noise       |
| `scaler`         | string | `"standard"`| Scaling method (`"standard"` or `"robust"`)        |
| `global_scaling` | bool   | `false`    | Single pooled scale vs per-channel scaling         |

### `sweep`

| key               | type       | default                         | meaning                                        |
| ----------------- | ---------- | ------------------------------- | ---------------------------------------------- |
| `reservoir_sizes` | list[int]  | `[100, 250, 500, 1000]`         | Outer grid reservoir sizes $N_\text{res}$      |
| `lambdas`         | list[float]| `[1e-5, 1e-4, ..., 10.0]`       | Inner grid ridge regularization candidate list |
| `n_trials`        | int        | `50`                            | Number of Optuna hyperparameter trials per $N_\text{res}$ |
| `model`           | dict       | `{}`                            | Optuna search space specification              |

### Shipped preset (`configs/nn_predictor/esn_8s.yaml`)

```yaml
simulation:
  dt: 1.0e-4
  downsample: 200
  data_path: data/experiment_excited_roast_8s/train

model:
  reservoir_size: 500
  spectral_radius: 0.9
  leak_rate: 0.1
  density: 0.1
  input_scaling: 0.1
  washout: 100
  ridge_lambda: 1.0e-3
  noise_sigma: 0.0
  horizon: 50

training:
  train_split: 0.8
  seed: 69
  scaler: robust
  global_scaling: true

artifact: artifacts/esn_8s

sweep:
  reservoir_sizes: [250, 500, 1000, 2000]
  n_trials: 50
```

---

## 10. Hyperparameter sweeps

[`scripts/sweep_esn.py`](../scripts/sweep_esn.py) performs a two-tier hyperparameter search:

1. **Outer Grid:** Iterates through specified reservoir sizes $N_\text{res} \in [250, 500, 1000, 2000]$.
2. **Inner Optuna Search:** For a given $N_\text{res}$, samples continuous/discrete reservoir hyperparameters:
   - `spectral_radius` $\rho \in [0.1, 1.5]$
   - `leak_rate` $\alpha \in [0.01, 1.0]$
   - `density` $d_\text{res} \in [0.01, 0.5]$
   - `noise_sigma` $\sigma \in [0.0, 0.5]$
3. **Inner Regularization Grid:** For each Optuna trial, normal equations $G, P$ are harvested once, and an inner loop evaluates validation rollout NMSE across candidate ridge values $\lambda \in [10^{-5}, 10^{-4}, \dots, 10.0]$.
4. **CasADi Complexity Profiling:** Instantiates `ESNSymbolicModel` for the winning trial at each $N_\text{res}$ and records `f_step.n_nodes()` to quantify the symbolic expression graph size for MPC solver planning.
5. **Output CSV:** Results are recorded in `esn_sweep_results.csv` containing columns for $N_\text{res}$, best validation NMSE, best $\lambda$, optimal parameters, timing statistics, and CasADi node count.

---

## 11. Notes & gotchas (Comparison to NN predictor)

- **Memory vs. Window Lag:** Unlike the MLP, which relies on an explicit finite history window ($n_y, n_u$), the ESN stores dynamic history implicitly in its state vector $h_t \in \mathbb{R}^{N_\text{res}}$.
- **One-Shot Closed Form vs. Iterative Gradient Descent:** ESN training evaluates state dynamics in a single pass and computes readout weights $W_\text{out}$ via a direct linear matrix solve ($O(N_\text{res}^3)$), taking seconds to train compared to multi-epoch Optax AdamW backpropagation.
- **Bias Term Unregularized:** In `solve_ridge`, $I_\text{unreg}[-1, -1] = 0.0$ ensures the constant bias offset is not shrunk toward zero, preventing systematic state offsets.
- **Washout Priming Requirement:** During deployment or evaluation, an ESN requires $W = \texttt{washout}$ initial steps of consecutive history to warm up state $h_t$ before issuing valid rollout predictions.
- **Determinism & Seeding:** Reservoir generation ($W_\text{res}, W_\text{in}$) is strictly deterministic given `training.seed`.
