# Hankel-DMDc Predictor

## Dynamic Mode Decomposition with Control for Observables

This document specifies the Hankel-DMDc (delay-embedded Dynamic Mode Decomposition with Control) Predictor for STFT log-power Observables and Raw EEG Waveforms.

---

### 1. Mathematical Formulation

#### 1.1 Hankel State & Control Embeddings

To model linear transitions of the STFT log-power Observable while respecting the non-Markovian memory induced by window overlap ($(K_{\text{width}} - 1) \cdot \text{hop} + N_{\text{segment}}$ samples) and delayed Jansen-Rit neural dynamics, the state and control history are stacked into delay-embedded Hankel vectors:

$$\mathbf{z}_k = \begin{bmatrix} \mathbf{o}_k \\ \mathbf{o}_{k-1} \\ \vdots \\ \mathbf{o}_{k-n_y+1} \end{bmatrix} \in \mathbb{R}^{n_y d_o}, \quad \mathbf{v}_k = \begin{bmatrix} \mathbf{u}_k \\ \mathbf{u}_{k-1} \\ \vdots \\ \mathbf{u}_{k-n_u+1} \end{bmatrix} \in \mathbb{R}^{n_u m}$$

The combined feature vector is $\mathbf{x}_k = [\mathbf{z}_k; \mathbf{v}_k] \in \mathbb{R}^f$ where $f = n_y d_o + n_u m$.

#### 1.2 Mean-Centered Truncated SVD

Given $N$ training snapshot pairs $\mathbf{X} \in \mathbb{R}^{N \times f}$ and $\mathbf{Y} \in \mathbb{R}^{N \times d_o}$ (where $\mathbf{Y}$ holds target Frames $\mathbf{o}_{k+1}$ or residual deltas $\Delta \mathbf{o}_{k+1} = \mathbf{o}_{k+1} - \mathbf{o}_k$), we compute empirical feature and target means:

$$\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{X}_{i, :}, \quad \bar{\mathbf{y}} = \frac{1}{N} \sum_{i=1}^N \mathbf{Y}_{i, :}$$

The centered snapshot matrix $\mathbf{\tilde{X}} = \mathbf{X} - \mathbf{1} \bar{\mathbf{x}}^T$ is decomposed via thin SVD:

$$\mathbf{\tilde{X}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

#### 1.3 Rank Selection and Tikhonov Regularization

The truncation rank $r \le \min(N, f)$ is determined by:

1. **Explicit rank $r$**: $r = \min(\text{dmd\_rank}, \text{rank}(\mathbf{\tilde{X}}))$.
2. **Cumulative singular-value energy threshold**: Smallest $r$ such that $\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^K \sigma_i^2} \ge \text{dmd\_energy}$ (defaulting to $0.99$).

The inverted singular values with optional Tikhonov damping $\lambda \ge 0$ are:

$$\sigma_{i, \text{inv}} = \frac{\sigma_i}{\sigma_i^2 + \lambda}$$

The linear weight matrix $\mathbf{W} \in \mathbb{R}^{d_o \times f}$ and affine bias vector $\mathbf{b} \in \mathbb{R}^{d_o}$ are:

$$\mathbf{W} = (\mathbf{\tilde{Y}}^T \mathbf{U}_r) \operatorname{diag}(\mathbf{\sigma}_{r, \text{inv}}) \mathbf{V}_r^T$$
$$\mathbf{b} = \bar{\mathbf{y}} - \mathbf{W} \bar{\mathbf{x}}$$

---

### 2. Runtime & Inference Seam

The fitted parameters $[\mathbf{W}, \mathbf{b}]$ are installed into a `depth=0` `AutoregressiveMLP` module and serialized into standard `.npz` exchange checkpoints. At inference time, `ObservableMLPModel` (JAX/Equinox) loads the linear weights and runs the discrete state transition during receding-horizon Trajopt MPC with zero runtime overhead.

---

### 3. Extended DMD (EDMD) Trade-Off Analysis

Extended DMD (EDMD) lifts state measurements through a nonlinear dictionary $\mathbf{\psi}(\mathbf{o}_k) \in \mathbb{R}^D$ to compute finite-dimensional approximations of the linear Koopman operator.

#### Why Hankel-DMDc is Preferred for STFT Observables

1. **Curse of Dimensionality**: The STFT log-power Frame already has dimension $d_o = n_{\text{channels}} \times n_{\text{values}} \approx 240$. Even quadratic polynomial lifting yields $D = \frac{d_o(d_o+1)}{2} \approx 28,920$ features, causing SVD computation, memory footprint, and snapshot matrix storage to explode.
2. **STFT Frame Memory**: Standard EDMD dictionary lifting assumes memoryless state Observables. However, STFT Frames inherently aggregate historical signal Segments. Takens' delay embedding via Hankel-DMDc directly addresses the temporal memory of the STFT without blowing up the feature space.
3. **Reconstruction & Decode**: EDMD requires an explicit projection map $\mathbf{C} \mathbf{\psi}(\mathbf{o}) \approx \mathbf{o}$ to decode Observable states for MPC Cost evaluation, whereas Hankel-DMDc operates directly in Observable coordinates.
