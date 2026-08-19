# Direct Observable Prediction for Closed-Loop Neurostimulation

Exploration of directly forecasting sensor-space observables (such as `eeg_ms` power envelopes or
hopped STFT spectral power) over the control horizon instead of rolling out raw EEG waveforms
autoregressively.

Source of truth:

- Symbolic MPC formulation: [`src/neuro/control/nlp.py`](../src/neuro/control/nlp.py)
- MPC controller implementation: [`src/neuro/control/nonlinear_mpc.py`](../src/neuro/control/nonlinear_mpc.py)
- Current autoregressive predictor: [`src/neuro/predictor/module.py`](../src/neuro/predictor/module.py)
- CasADi symbolic model wrapper: [`src/neuro/nn_predictor_casadi.py`](../src/neuro/nn_predictor_casadi.py)
- Observable losses: [`src/neuro/predictor/losses.py`](../src/neuro/predictor/losses.py)
- Spectral objectives: [`docs/spectral_objectives.md`](spectral_objectives.md)
- Domain vocabulary: [`CONTEXT.md`](../CONTEXT.md)

---

## 1. Core Paradigm Shift

### Current Architecture (Autoregressive Raw EEG)

1. The surrogate predictor ingests past standardized EEG and control history $(y_{\text{past}}, u_{\text{past}})$.
2. It autoregressively predicts single-step standardized EEG $\hat{y}_{t+1} = f(\hat{y}_t, u_t)$ rolled out $H$ steps into the future.
3. The controller evaluates the cost objective (e.g. EEG power or spectral hinge $J_{\text{PSD}}$ via a symbolic DFT) on the predicted raw EEG trajectory $\hat{y}_{1:H}$.

```text
[Raw EEG & u history]
       │ (State Absorption)
       ▼
[Autoregressive Predictor] ──(Rollout H steps)──► [Predicted Raw EEG: (H, C)]
                                                         │
                                                         ▼
                                            [_spectral_hinge_cost (CasADi DFT)]
                                                         │
                                                         ▼
                                                    [Cost J_PSD]
```

### Proposed Architecture (Direct Observable Forecasting)

1. The surrogate predictor directly forecasts the sequence of observables over the horizon $\hat{\mathbf{z}} \in \mathbb{R}^{M \times C \times F}$ (e.g. hopped STFT power frames or windowed `eeg_ms`) given history and the candidate control sequence $\mathbf{u} = [u_0, \dots, u_{H-1}]$.
2. This mapping can be a direct multi-step / non-autoregressive feedforward function:
   $$\hat{\mathbf{z}} = g(x_0, \mathbf{u})$$
3. The controller evaluates the cost directly on $\hat{\mathbf{z}}$, without intermediate trajectory unrolling or symbolic DFT computations.

```text
[History State x_0]
       │
       ▼
[Direct Observable Predictor: g(x_0, u)] ──► [Predicted Observable: (M, C, F)]
                                                         │
                                                         ▼
                                              [Direct Stage / Hinge Cost]
                                                         │
                                                         ▼
                                                    [Cost J_PSD]
```

---

## 2. Literature Precedents

1. **Envelope / Band-Power Predictive Control in Neurostimulation & DBS:**
   - In closed-loop deep brain stimulation (DBS) and cortical stimulation (e.g., *Santaniello et al., Fleming et al., Little et al., Tinkhauser et al.*), control loops almost exclusively track and predict **spectral band power** (e.g. beta-band power envelope $\sim 13\text{--}30\text{ Hz}$) or energy/line-length envelopes rather than raw oscillating LFP/EEG waveforms.
   - Raw oscillatory voltages fluctuate rapidly ($>50\text{ Hz}$) with high phase sensitivity, whereas power/amplitude envelopes evolve on a much slower, quasi-stationary timescale ($\sim 1\text{--}10\text{ Hz}$).

2. **Direct Multi-Step (DMS) vs. Recursive Forecasting:**
   - In time-series analysis and system identification (*Chevillon 2007*, *Marcellino et al. 2006*), predicting the entire horizon $\hat{\mathbf{z}} = g(x_0, \mathbf{u})$ simultaneously is the standard Direct Multi-Step approach. It avoids the compounding single-step autoregressive errors and autoregressive feedback divergence.

3. **Observable / Feature-Space MPC & Koopman Operator Control:**
   - In data-driven MPC (*Korda & Mezić 2018*, *Peitz et al.*), dynamical systems are lifted into a space of nonlinear observables (such as energies, harmonic modes, or powers) where dynamics can be stepped forward or directly optimized without simulating the underlying microscopic state.

---

## 3. Theoretical and Biophysical Hazards

### 3.1 The Phase Ambiguity & Cross-Term Dilemma ($2 y^T \Delta y$)

When an external electrical current $u$ is delivered through scalp electrodes, the resulting electric field perturbs the neural membrane potential by $\Delta y$.

In raw EEG space, the perturbed signal is additive:
$$y_{\text{perturbed}} = y + \Delta y$$

The resulting signal power contains an explicit cross-term:
$$\|y + \Delta y\|^2 = \|y\|^2 + \underbrace{2 y^T \Delta y}_{\text{phase-dependent interference}} + \|\Delta y\|^2$$

- **The issue:** If the model only observes past power $\|y\|^2$, but **lacks the instantaneous phase of $y$**, it cannot determine whether an instantaneous control pulse $u$ will constructively interfere (increase power) or destructively interfere (quench power).
- **When direct power prediction holds:** Direct observable prediction is valid when stimulation acts on **slow excitability / gain dynamics** (e.g., shifting neural population firing thresholds and quenching seizure recruitment over several oscillation cycles) rather than cycle-by-cycle phase cancellation.

### 3.2 Non-Markovianity of the Observable Space

- A history of raw EEG $[y_{t-n_y}, \dots, y_t]$ carries phase, instantaneous velocity, and higher derivatives.
- A history of scalar power $[z_{t-n_z}, \dots, z_t]$ alone discards phase, which can make the observable state non-Markovian unless:
  - The input history state still ingests the raw EEG window $[y_{t-n_y}, \dots, y_t]$ during priming/absorption, but outputs future observables $\hat{z}_{1:M}$, or
  - The lookback buffer of observables is sufficiently wide to capture slow envelope dynamics.

### 3.3 Horizon-Wide Causal Structure

In an autoregressive model, causality is guaranteed by step-by-step unrolling. In a direct horizon network $\hat{z}_{1:M} = g(x_0, [u_0, \dots, u_{H-1}])$, the network architecture must enforce causal structure (e.g. causal convolutions or masked lower-triangular weight matrices) so that observable frame $m$ does not depend on future controls $u_k$ where $t_k > t_m$.

---

## 4. Application to This Codebase

### 4.1 Elimination of the CasADi DFT Graph

In [`src/neuro/control/nlp.py`](../src/neuro/control/nlp.py), the MPC spectral cost currently computes:

```python
# Slices segments from predicted raw EEG and multiplies by DFT cosine/sine matrices
power = ((y_tapered @ ca.MX(dft_cos)) ** 2 + (y_tapered @ ca.MX(dft_sin)) ** 2) * scale_row
hinge = ca.fmax(0.0, ca.log(power[:, 1:] + LOG_FLOOR) - log_ref)
```

CasADi has no native FFT, so this constructs large symbolic matrix multiplication graphs across all segments $M$ and channels $C=62$.

With direct observable prediction:

- The predictor outputs $\hat{P}_{m, c, f}$ directly.
- The NLP cost simplifies to a trivial elementwise hinge:
  $$J_{\text{PSD}} = \frac{1}{M \cdot C \cdot F'} \sum_{m, c, f} \left[\max\left(0, \log(\hat{P}_{m, c, f} + \varepsilon) - \log P^{\text{ref}}_{c, f}\right)\right]^2$$
- This completely removes the dense DFT graph from CasADi.

### 4.2 State Absorption & Priming

The controller interface remains clean:

- `model.absorb(state, y_raw, u_last)` can continue to buffer incoming raw EEG measurements and control inputs.
- When `model.is_ready(state)` is satisfied, `model.predict_horizon(state, u_seq)` evaluates the direct forward map.

---

## 5. Comparison: Autoregressive Raw Model vs. Direct Observable Model

| Dimension | Autoregressive Raw EEG Model | Direct Observable Model |
| :--- | :--- | :--- |
| **Solver Graph Complexity** | $H$ sequential layer evaluations + multiple shooting defect constraints + symbolic DFT matrices | Single feedforward network evaluation + elementwise hinge |
| **Solve Time & Latency** | High (large NLP with intermediate state roots or deep unrolled graph) | Low (controls-only NLP with simple box bounds $-u_{\max} \le u \le u_{\max}$ and $\sum u = 0$) |
| **NLP Decision Variables** | Controls $\mathbf{u} \in \mathbb{R}^{H \times m}$ and state shooting roots $\phi \in \mathbb{R}^{S \times n_x}$ | Controls $\mathbf{u} \in \mathbb{R}^{H \times m}$ only (plus L1 slacks if active) |
| **Gradient Propagation** | Backpropagation through $H$ recurrent unrolls (susceptible to vanishing/exploding gradients) | Direct analytic Jacobian $\frac{\partial \hat{\mathbf{z}}}{\partial \mathbf{u}}$ from the forward network |
| **Objective Alignment** | Predictor minimizes EEG MSE + auxiliary STFT loss; MPC scores spectral hinge | Zero representation mismatch: the predictor directly forecasts what the MPC penalizes |
| **Temporal Granularity** | Step-by-step ($dt = 0.02\text{ s}$, 50 steps) | Hopped frame-by-frame ($M$ frames, e.g. 3–5 frames over the horizon) |

---

## 6. Verification and Implementation Roadmap

Before modifying the controller or rewriting CasADi interfaces, verify the hypothesis offline:

1. **Predictability Probe:**
   Train a direct feedforward network on existing training trajectories:
   $$[y_{\text{past}} \in \mathbb{R}^{n_y \times 62}, u_{\text{past}} \in \mathbb{R}^{n_u \times 10}, u_{\text{future}} \in \mathbb{R}^{H \times 10}] \longrightarrow \hat{P}_{\text{future}} \in \mathbb{R}^{M \times 62 \times F}$$
   Score test-set loss against the target STFT frames.

2. **Control Sensitivity Analysis ($\nabla_u \hat{P}$):**
   Verify that varying $u_{\text{future}}$ yields smooth, monotonic, and biophysically sensible suppression in predicted spectral power $\hat{P}$, confirming that the absence of phase in the target does not induce ill-conditioned gradients.

3. **Closed-Loop Benchmark:**
   Evaluate whether the direct observable model running with fast SQP/IPOPT solves matches or exceeds the seizure suppression efficacy of the full autoregressive raw-signal MPC.
