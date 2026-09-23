# Observable Predictor Latency and Action Sensitivity: Diagnostic Analysis and Proposals

**Date:** September 2026
**Context:** Follow-up to `artifacts/observable_mlp_retraining_comparison/report.md` and `artifacts/observable_mlp_seed_diagnosis/report.md`.
**Scope:** Addressing the two identified bottlenecks in Observable closed-loop control:

1. Underestimating the stimulation derivative $\frac{\partial \mathbf{y}}{\partial \mathbf{u}}$ by $8.7\times\text{--}20.7\times$ (action blindness).
2. Incurring $60\text{--}100\,\text{ms}$ STFT windowing and filter lag (delayed recruitment detection).

---

## 1. Executive Summary

Retraining Observable MLPs through full curriculum completion restored autoregressive numerical stability but did not eliminate seed-specific recruitment failures. The failure mechanism is dynamical rather than numerical:

- **Action Sensitivity Deficit:** Because unweighted MSE training against autonomous neural activity downweights the small incremental effect of stimulation, the model underestimates $\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{u}}$ by up to $20\times$. The MPC solver penalizes effort heavily relative to the perceived spectral benefit and under-stimulates near recruitment thresholds.
- **Temporal Detection Lag:** Symmetrically windowed STFT Segments and temporal Frame Kernels pool up to $1.12\,\text{s}$ of trailing history. Epileptiform synchronization spreads across coupled regions within $50\text{--}100\,\text{ms}$, before the smoothed Observable Frame crosses the threshold.

Below are six concrete proposals addressing both bottlenecks while preserving the existing residual Predictor formulation.

---

## 2. Proposal Breakdown

### 2.1 MPC Effort Calibration (Immediate Sensitivity Compensation)

- **Target:** Action-sensitivity underestimation ($\frac{\partial \mathbf{y}}{\partial \mathbf{u}}$ deficit).
- **Mechanism:** In the MPC objective:
  $$\mathcal{J}(\mathbf{u}_{0:H-1}) = \sum_{k=1}^H \mathcal{C}_\text{spectral}(\hat{\mathbf{y}}_k) + w_\text{quad} \sum_{k=0}^{H-1} \|\mathbf{u}_k\|_2^2 + w_\text{sparse} \sum_{k=0}^{H-1} \|\mathbf{u}_k\|_1$$
  the necessary optimality condition balances the marginal spectral benefit against marginal effort:
  $$\nabla_\mathbf{u} \mathcal{J} = \left(\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{u}}\right)^\top \nabla_{\hat{\mathbf{y}}} \mathcal{C}_\text{spectral} + 2 w_\text{quad} \mathbf{u} + w_\text{sparse} \operatorname{sign}(\mathbf{u}) = \mathbf{0}$$
  Because $\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{u}}$ is underestimated by a factor of $\alpha \approx 10$, the solver views stimulation as $10\times$ less effective than it physically is, allowing the effort penalties to prematurely suppress counter-stimulation.
- **Proposed Action:**
  1. Calibrate $w_\text{quad}$ and $w_\text{sparse}$ downward by $10\times\text{--}50\times$ (e.g., $w_\text{quad}$ from $1.0$ to $0.05\text{--}0.02$) on failing seeds (7001, 7004).
  2. Alternatively, incorporate an empirical sensitivity multiplier $\alpha$ into the linearized Predictor step inside the OSQP/qpOASES solver formulation ($\Delta \hat{\mathbf{y}} = \mathbf{A}\Delta \mathbf{y} + \alpha \mathbf{B} \Delta \mathbf{u}$).
- **Touchpoints:** `configs/simulation/*.yaml` under `controller.cost.weights`.
- **Advantage:** Requires **zero model retraining**; can be verified immediately on existing checkpoints.

---

### 2.2 Asymmetric STFT Windowing (Causal Group-Delay Reduction)

- **Target:** $60\text{--}100\,\text{ms}$ STFT windowing lag.
- **Mechanism:** In `src/neuro/spectral.py`, `compute_log_power_frames` applies a symmetric Hann window across each Segment of length $N = n_\text{segment}$:
  $$w_\text{hann}[n] = \frac{1}{2}\left(1 - \cos\frac{2\pi n}{N}\right), \quad n = 0, \dots, N-1$$
  Its center of energy is at $n = N/2$, imposing an inherent group delay of:
  $$\tau_g = \frac{N}{2 f_s} \approx 250\text{--}500\,\text{ms}$$
  When followed by a temporal Frame Kernel of width $K$, the total sample support reaches $0.98\text{--}1.12\,\text{s}$.
- **Proposed Action:**
  Replace the symmetric Segment window with an asymmetric causal taper that concentrates weight on the most recent samples ($n \to N-1$):
  $$w_\text{causal}[n] = \exp\left(-\lambda \frac{N - 1 - n}{N}\right) \cdot \sin\left(\frac{\pi n}{2(N - 1)}\right)$$
- **Impact:** Reduces effective filter group delay by $50\%\text{--}70\%$ (to $<30\,\text{ms}$ at the hop rate), allowing emerging epileptiform bursts to register in the Observable Frame with minimal latency while preserving identical output dimensions.
- **Touchpoints:** `src/neuro/spectral.py` (`compute_log_power_frames`), `src/neuro/config.py` (`StftGeometry`).

---

### 2.3 Training Data Input Holding-Periods & Transition Reweighting

- **Target:** Action-sensitivity underestimation ($\frac{\partial \mathbf{y}}{\partial \mathbf{u}}$ deficit).
- **Mechanism:** In `data/experiment_excited`, excitation currents are held constant for randomly drawn durations:
  $$\Delta t_\text{hold} \in [10, 100, 200, 400, 1000]\,\text{ms}$$
  During a $1000\,\text{ms}$ hold (16.7 hops at 60 ms), $\Delta \mathbf{u} = \mathbf{0}$. Over $80\%$ of the transitions in the training set exhibit constant control inputs.
  Standard MSE loss $\frac{1}{T}\sum_t \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|^2$ is dominated by autonomous relaxation. The network easily discovers that setting input weights $W_u \approx 0$ yields low aggregate loss.
- **Proposed Action:**
  1. **Regenerate Excitation Datasets with Rapid Switching:** Shift the hold distribution from long holds to rapid multi-rate transitions: holds in $[10, 20, 40, 60, 100]\,\text{ms}$, vastly increasing the density of dynamic step responses.
  2. **Transition-Weighted Loss:** In `src/neuro/predictor/train.py`, weight the per-sample loss by the magnitude of the control step:
     $$w_k = 1.0 + \beta \|\mathbf{u}_k - \mathbf{u}_{k-1}\|_2$$
     forcing gradient descent to penalize errors during the critical transient response window.
- **Touchpoints:** `configs/simulation/experiment_excited.yaml`, `src/neuro/predictor/train.py`.

---

### 2.4 Shorter Segments and Higher Hop Rates (Fast Observable Grid)

- **Target:** $60\text{--}100\,\text{ms}$ STFT windowing lag.
- **Mechanism:**
  Current canonical configurations:
  - `champion_band3_12`: $n_\text{segment}=50$ ($1.0\,\text{s}$), $n_\text{hop}=5$ ($100\,\text{ms}$ knot).
  - `fast_boxcar_kw9`: $n_\text{segment}=25$ ($0.5\,\text{s}$), $n_\text{hop}=3$ ($60\,\text{ms}$ knot).
  A 60–100 ms knot period means the controller only acts every 600–1000 plant steps ($dt = 0.0001$).
- **Proposed Action:**
  Transition to a high-rate Observable grid:
  - $n_\text{segment} = 10\text{--}15$ samples ($200\text{--}300\,\text{ms}$ at 50 Hz).
  - $n_\text{hop} = 1\text{--}2$ samples ($20\text{--}40\,\text{ms}$ decision interval).
  - Kernel width $K = 1$ (no multi-hop smoothing).
- **Spectral Resolution Trade-Off:**
  At $f_s = 50\,\text{Hz}$ and $N = 15$, bin resolution is $\Delta f = \frac{50}{15} \approx 3.33\,\text{Hz}$. The seizure band (3–12 Hz) retains 3 frequency bins (3.3, 6.7, 10.0 Hz), which adequately resolves theta and alpha power while reducing knot latency by $2.5\times\text{--}3\times$.
- **Touchpoints:** `StftGeometry` in model/simulation configs, `scripts/build_healthy_psd.py`.

---

### 2.5 Retaining the Residual Formulation $\mathbf{y}_{k+1} = \mathbf{y}_k + f(\mathbf{Y}_k, \mathbf{U}_k)$

- **Decision:** Preserve the established residual architecture over control-affine alternatives.
- **Rationale:**
  1. `ObservableMLPModel` and `ObservableCNNModel` already implement:
     $$\hat{\mathbf{y}}_{k+1} = \mathbf{y}_k + f(\mathbf{Y}_k, \mathbf{U}_k)$$
     when `residual: true` (standard across all trained checkpoints).
  2. The model already models increments $\Delta \mathbf{y}_k$ rather than absolute state.
  3. A control-affine reformulation $\hat{\mathbf{y}}_{k+1} = f(\mathbf{Y}_k) + \mathbf{B}(\mathbf{Y}_k)\mathbf{u}_k$ would require rewriting JAX/PyTorch model classes, serialization schemas, and validation routines without solving the core data-weighting problem.
  4. Coupling the residual architecture with **Proposals 2.1 and 2.3** (effort calibration + short-hold excitation data) directly resolves action sensitivity without architectural churn.

---

### 2.6 Instantaneous Teager-Kaiser Energy Operator (TKEO; arXiv:2511.17164)

- **Reference:** Chourdaki, Avramidis, Garoufis, Zlatintsi, & Maragos, *"Teager-Kaiser Energy Methods For EEG Feature Extraction In Biomedical Applications"*, arXiv:2511.17164 (Nov 2025).
- **Mathematical Definition:**
  For discrete signal $x[n]$, the non-linear Teager-Kaiser Energy Operator is defined as:
  $$\Psi[x[n]] = x^2[n] - x[n-1] x[n+1]$$
  For an oscillating AM-FM signal $x[n] = a[n]\cos(\phi[n])$ with instantaneous amplitude $a[n]$ and instantaneous frequency $\Omega[n] = \frac{d\phi}{dn}$:
  $$\Psi[x[n]] \approx a^2[n] \Omega^2[n]$$
  It measures the instantaneous physical energy required to generate the oscillation.
- **Why TKEO is Suited for Closed-Loop Seizure Suppression:**
  1. **Zero Group Delay:** Unlike Fourier transforms that require a multi-cycle segment, discrete TKEO requires only **three consecutive samples** ($x[n-1], x[n], x[n+1]$). At $f_s = 50\,\text{Hz}$, this corresponds to a lookback of only $40\,\text{ms}$; at plant rate ($10\,\text{kHz}$), it is $<0.3\,\text{ms}$.
  2. **Simultaneous Amplitude and Frequency Amplification:** During seizure onset, both amplitude $a[n]$ and synchronization frequency $\Omega[n]$ surge. TKEO amplifies this product quadratically, creating a sharp, instantaneous onset spike before STFT power accumulates.
  3. **Energy Separation Algorithm (ESA):** As demonstrated by Chourdaki et al., passing bandpassed EEG (e.g. 3–12 Hz) through TKEO allows the Energy Separation Algorithm to extract instantaneous amplitude envelope and frequency with near-zero latency:
     $$a[n] \approx \frac{2\Psi[x[n]]}{\sqrt{\Psi[x[n+1] - x[n-1]]}}, \qquad \Omega[n] \approx \arcsin\left(\sqrt{\frac{\Psi[x[n+1] - x[n-1]]}{4\Psi[x[n]]}}\right)$$
- **Closed-Loop Integration Options:**
  - **Option A: Dual-Rate Objective Hinge (`TkeoHingeCost`):** Add a lightweight instantaneous TKEO penalty to the MPC cost alongside the STFT Frame cost. If the instantaneous TKEO crosses threshold, the optimizer applies counter-stimulation immediately, suppressing regional spread before the STFT window completes.
  - **Option B: Hybrid Observable Vector:** Concatenate instantaneous channel TKEO energy directly into the Observable vector $\mathbf{x}_\text{obs} = [\text{STFT Frame}, \Psi(\mathbf{y})]$, providing the Predictor with zero-lag onset awareness.

---

## 3. Comparison Matrix & Implementation Roadmap

| Proposal | Primary Benefit | Implementation Effort | Requires Retraining? |
| :--- | :--- | :---: | :---: |
| **2.1 MPC Effort Calibration** | Immediate restoration of counter-stimulation | Minimal | **No** |
| **2.2 Asymmetric STFT Taper** | $50\%\text{--}70\%$ group-delay reduction | Low | Yes (References & Models) |
| **2.3 Holding-Period Reweighting** | Cures $\frac{\partial \mathbf{y}}{\partial \mathbf{u}}$ underestimation at the source | Medium | Yes (Dataset & Training) |
| **2.4 Shorter Segments (Fast Knots)** | $2.5\times\text{--}3\times$ faster decision rate ($20\text{--}40\,\text{ms}$) | Medium | Yes (Full pipeline) |
| **2.5 Retain Residual Formulation** | Zero architectural churn | None | — |
| **2.6 Instantaneous TKEO** | Near-zero-latency ($<1\,\text{ms}$) onset detection | Medium | Optional (Cost or Feature) |

### Recommended Phased Roadmap

1. **Phase 1 (Immediate Zero-Retraining Calibration):**
   Evaluate Proposal 2.1 (effort weight scaling $w_\text{quad} \in [0.1, 0.02]$) across the 5 seeds on the retrained checkpoints to verify whether compensating for the $10\times$ sensitivity gap eliminates the failing seeds (7001, 7004).
2. **Phase 2 (Zero-Lag Onset Triggering):**
   Prototype Proposal 2.6 (TKEO instantaneous hinge cost) in simulation to arrest initial epileptiform bursts during the 60–100 ms STFT blind window.
3. **Phase 3 (Next Model Generation):**
   Combine Proposal 2.2 (asymmetric causal windowing), Proposal 2.3 (short-hold excitation dataset), and Proposal 2.4 (20–40 ms knot period) to train a fast, action-sensitive Observable Predictor generation.
