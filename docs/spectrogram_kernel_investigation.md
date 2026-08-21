# Spectrogram loss kernel continuum and closed-loop control

This document reports the investigation into time-resolved spectrogram losses for neural predictor training, why the raw short-time Fourier transform (STFT) loss degraded closed-loop seizure control, and how intermediate frame smoothing kernels resolve the trade-off between time resolution and controller authority.

Source files:

- Loss implementation: [`StftLoss`, `frame_kernel`, `smooth_frames`](../src/neuro/predictor/losses.py)
- Autoregressive predictor: [`AutoregressiveMLP`](../src/neuro/predictor/module.py)
- MPC formulation: [`MPCNlp`, `_spectral_hinge_cost`](../src/neuro/control/nlp.py)
- Offline probe scripts: [`probe_stft_geometry.py`](../scripts/probe_stft_geometry.py), [`probe_gradient_dynamics.py`](../scratch/kernel_investigation/probe_gradient_dynamics.py)
- Prior analysis notes: [`spectrogram_loss_guide.md`](spectrogram_loss_guide.md), [`stft_closed_loop_investigation.md`](stft_closed_loop_investigation.md)

---

## 1. Context and problem statement

The neural predictor models multi-channel EEG dynamics to forecast seizure progression inside a model predictive controller (MPC). Originally, predictor training used a Welch power spectral density loss (`PSDLoss`). Welch computes periodograms across a 1-second rollout and averages over the time axis, collapsing the entire window to a single frame ($M=1$).

To capture transient spectral dynamics, a hopped STFT loss (`StftLoss`, $W=25$, $H=12$, $M=3$) was introduced. Offline ranking on held-out ensembles selected this geometry because it achieved higher spectral rank separation (AUC = 0.885) than Welch (AUC = 0.793).

When tested in closed-loop simulations with the Jansen-Rit plant, the STFT predictor performed substantially worse:

| Evaluation Metric | Welch Endpoint ($W=50, H=50$) | Raw STFT ($W=25, H=12$) |
| :--- | ---: | ---: |
| Offline rank separation (AUC) | 0.793 | **0.885** |
| 1-second rollout NMSE | **0.498** | 0.501 |
| Seizure burden | **0.052** | 0.217 |
| Seeds suppressed | **3 / 3** | 1 / 3 |
| Mean delivered charge ($\mu\text{C}$) | **23.3** | 78.4 |
| Total rollout sensitivity $\|\partial \text{rollout}/\partial u\|$ | **6.81** | 2.12 |

The predictor that separated spectra best produced an ineffective controller, increasing seizure burden by a factor of 4 while consuming 3.4x more charge.

---

## 2. Methodology

We evaluated candidate loss configurations along the continuum between raw STFT and Welch using four complementary measurements.

### 2.1 Loss formulation and pre-log smoothing

Given standardized rollout predictions $\hat{y} \in \mathbb{R}^{B \times T \times C}$, signals are mapped to raw units $x = \hat{y} \cdot \sigma_y + \mu_y$. Hopped periodograms are computed per channel:

$$\hat{P}(b, c, m, f) = \frac{|\text{rfft}(x_{b,c,m} \odot w_{\text{Hann}})|^2}{f_s \sum w_{\text{Hann}}^2}$$

where $m \in \{0, \dots, M_{\text{in}}-1\}$ is the frame index and $f$ is the frequency bin.

A smoothing kernel $v \in \mathbb{R}^{K_w}$ is applied along the frame axis $m$ *before* the log:

$$\tilde{P}(b, c, m, f) = \sum_{k=0}^{K_w-1} v_k \hat{P}(b, c, m+k, f)$$

The loss is the mean squared log-residual:

$$\mathcal{L} = \frac{1}{B C M_{\text{out}} F} \sum_{b,c,m,f} \left(\log(\tilde{P}(b, c, m, f) + \epsilon) - \log(\tilde{P}_{\text{true}}(b, c, m, f) + \epsilon)\right)^2$$

where $\epsilon = 10^{-6}\text{ mV}^2/\text{Hz}$ (`LOG_FLOOR`) and $M_{\text{out}} = M_{\text{in}} - K_w + 1$.

Effective degrees of freedom per cell are defined by:

$$K_{\text{eff}} = \frac{(\sum_k v_k)^2}{\sum_k v_k^2}$$

### 2.2 Theoretical estimator floor

A single periodogram cell is $\chi_2^2$-distributed around the true underlying spectrum. For two independent realizations, the variance of their difference in log-power is $2\psi'(1) = 3.29\text{ nats}^2$. Pooling $K_{\text{eff}}$ frames before the log reduces the variance and theoretical floor to:

$$\text{Floor}_{\text{theory}} = 2 \psi'(K_{\text{eff}})$$

We measure the empirical floor on branched ensemble pairs differing only in noise realizations.

### 2.3 Autoregressive input sensitivity probe

In an autoregressive predictor $y_{t+1} = f(y_t \dots y_{t-n_y+1}, u_t \dots u_{t-n_u+1})$, control inputs enter layer 0 through weight matrix $W_u$, while past outputs enter through $W_y$.

We compute the exact Jacobian of predictions at lookahead step $k$ with respect to control applied at step $t$:

$$J_k = \frac{\partial \hat{y}_{t+k}}{\partial u_t} \in \mathbb{R}^{C \times N_u}$$

and the total rollout sensitivity over horizon $H=50$:

$$J_{\text{rollout}} = \left( \sum_{k=1}^H \|J_k\|_F^2 \right)^{1/2}$$

### 2.4 Online MPC optimization landscape

The controller solves an optimal control problem in CasADi with IPOPT over horizon $H=75$:

$$\min_u J(u) = \frac{w_u}{H} \sum_{k=0}^{H-1} \|u_k\|^2 + w_{\text{psd}} \cdot \text{Hinge}(\hat{y}(u))$$

We evaluate the symbolic gradient $\nabla_u J(0)$ at seizure onset and trace the convergence path of IPOPT.

---

## 3. Why raw STFT failed in the closed loop

Our investigation identified a three-stage failure mechanism.

### 3.1 Loss gradient mechanics in short windows

In a short window ($W=25$ samples = 0.5 s), frequency resolution is coarse ($\Delta f = 2.0\text{ Hz}$) and the Hann main lobe is $4.0\text{ Hz}$ wide.

In training trajectories, stimulation pulses create local high-frequency transients. In log-power space, relative error dominates in low-power high-frequency bins ($>12\text{ Hz}$, power $10^{-2}$ to $10^{-3}\text{ mV}^2/\text{Hz}$):

| Frequency | True Power ($\text{mV}^2/\text{Hz}$) | Mean Squared Log Residual (STFT $W=25$) | Mean Squared Log Residual (Welch $W=50$) |
| ---: | ---: | ---: | ---: |
| 0 Hz | 45.6 | 3.19 | 2.72 |
| 4 Hz | 4.03 | 7.50 | 9.13 |
| 8 Hz | 7.41 | 11.68 | 13.76 |
| 10 Hz | 8.23 | 12.53 | 14.05 |
| 14 Hz | 0.48 | 15.01 | 14.94 |
| 18 Hz | 0.23 | 12.33 | 10.00 |
| 22 Hz | 0.017 | 7.97 | 6.27 |

Because relative errors in background bins are large, backpropagation generates high prediction gradients ($\|dL/d\hat{y}\| = 49.5$ for STFT versus $31.4$ for Welch).

To minimize this short-window log-power variance, the optimizer attenuates the input weights $W_u$ relative to autonomous autoregressive weights $W_y$. The parameter norm ratio $\|W_u\| / \|W_y\|$ falls from $0.1456$ to $0.1093$.

### 3.2 Autoregressive sensitivity collapse

Because predictions feed back into the input register, this input attenuation compounds across the lookahead horizon:

| Lookahead Step $k$ | Lookahead Time | Welch $\|\partial \hat{y}_{t+k}/\partial u_t\|$ | Raw STFT $\|\partial \hat{y}_{t+k}/\partial u_t\|$ | Ratio (STFT / Welch) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02 s | 0.5386 | 0.3119 | 0.58x |
| 3 | 0.06 s | 0.9028 | 0.2825 | **0.31x** |
| 10 | 0.20 s | 0.7453 | 0.2668 | **0.36x** |
| 30 | 0.60 s | 0.3814 | 0.1189 | **0.31x** |
| 50 | 1.00 s | 0.0352 | 0.0051 | **0.14x** |
| **Total rollout** | **1.00 s** | **6.8075** | **2.1249** | **0.31x** |

The STFT model learned to treat external stimulation as an attenuated disturbance and relied on its autonomous state transition to predict spectral shape.

### 3.3 Online MPC gradient collapse and spatial steering breakdown

The MPC objective gradient with respect to control action $u$ is:

$$\nabla_u J = \frac{2 w_u}{H} u + w_{\text{psd}} \left( \frac{\partial \text{Hinge}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial u} \right)$$

Evaluating the CasADi NLP graph at seizure onset:

1. **Gradient magnitude:** $\|\nabla_u J(0)\|$ drops from $237.88$ (Welch) to $34.16$ (STFT), a 7.0x collapse.
2. **Perceived cost reduction:** IPOPT finds an optimal control trajectory with perceived cost reduction $\Delta J = -189.7$ for STFT, compared to $\Delta J = -1538.0$ for Welch (8.1x smaller).
3. **Loss of dipole coordination:** Commanded controls from the Welch predictor form a structured bipolar source (cross-channel correlations of $-0.93$ and $-0.94$). Commanded controls from the STFT predictor lose this coupling (correlations of $-0.26$ and $-0.28$).
4. **Saturation trap:** Because the model perceives stimulation as weak, IPOPT drives controls to hard bounds ($u_{\text{max}} = 2.0$) with erratic spatial vectors. The uncoordinated currents fail to suppress the plant, causing persistent seizure activity and forcing the controller to continuously deliver high charge.

---

## 4. The intermediate kernel continuum

We evaluated 18 intermediate kernel smoothers across geometries to determine if pre-log frame pooling can prevent input sensitivity collapse while retaining multi-frame time resolution ($M_{\text{out}} \ge 2$).

### 4.1 Estimator floor, noise, and spectral separation

| Candidate | Geometry | Kernel | Width | $M_{\text{out}}$ | $K_{\text{eff}}$ | Emp Floor ($\text{nats}^2$) | Theo Floor | Signal | Offline AUC |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Welch 50 | W50 H50 s50 | boxcar | 1 | 1 | 1.00 | 4.64 | 3.29 | 32.47 | 0.793 |
| Welch 75 | W75 H75 s75 | boxcar | 1 | 1 | 1.00 | 4.66 | 3.29 | 46.82 | 0.793 |
| W25 H12 s50 raw | W25 H12 s50 | boxcar | 1 | 3 | 1.00 | 4.45 | 3.29 | 43.68 | 0.885 |
| **W25 H12 s50 box2** | W25 H12 s50 | boxcar | 2 | 2 | 2.00 | 2.55 | 1.29 | 35.79 | 0.838 |
| W25 H12 s50 box3 | W25 H12 s50 | boxcar | 3 | 1 | 3.00 | 2.06 | 0.79 | 23.98 | 0.820 |
| W25 H12 s50 hann3 | W25 H12 s50 | hann | 3 | 1 | 2.67 | 2.12 | 0.91 | 25.42 | 0.832 |
| W25 H6 s50 raw | W25 H6 s50 | boxcar | 1 | 5 | 1.00 | 4.45 | 3.29 | 42.95 | 0.875 |
| W25 H6 s50 box2 | W25 H6 s50 | boxcar | 2 | 4 | 2.00 | 2.93 | 1.29 | 39.63 | 0.859 |
| **W25 H6 s50 box3** | W25 H6 s50 | boxcar | 3 | 3 | 3.00 | 2.15 | 0.79 | 34.61 | 0.834 |
| **W25 H6 s50 hann3** | W25 H6 s50 | hann | 3 | 3 | 2.67 | 2.22 | 0.91 | 35.46 | 0.835 |
| W25 H6 s50 box4 | W25 H6 s50 | boxcar | 4 | 2 | 4.00 | 1.94 | 0.57 | 29.56 | 0.832 |
| W37 H6 s50 raw | W37 H6 s50 | boxcar | 1 | 3 | 1.00 | 4.67 | 3.29 | 41.52 | 0.840 |
| **W37 H6 s50 box2** | W37 H6 s50 | boxcar | 2 | 2 | 2.00 | 2.76 | 1.29 | 34.58 | 0.814 |
| W50 H12 s75 raw | W50 H12 s75 | boxcar | 1 | 3 | 1.00 | 4.84 | 3.29 | 61.19 | 0.822 |
| W50 H12 s75 box2 | W50 H12 s75 | boxcar | 2 | 2 | 2.00 | 2.87 | 1.29 | 48.97 | 0.803 |

Smoothing along frames monotonically lowers the estimator floor towards the theoretical limit. Offline rank separation (AUC) decreases as kernel width increases, confirming that temporal smoothing trades offline discriminability for estimator variance reduction.

### 4.2 Loss gradient magnitudes and noise suppression

| Candidate | $M_{\text{out}}$ | $K_{\text{eff}}$ | Loss ($\text{nats}^2$) | $\|dL/d\hat{y}\|$ | $\|\nabla_{W_u} L\|$ | $\|\nabla_{W_y} L\|$ | In-Band Loss | Out-of-Band Noise |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Welch 50 | 1 | 1.00 | 9.21 | 28.41 | 13.78 | 80.42 | — | — |
| W25 H12 s50 raw | 3 | 1.00 | 8.48 | 32.40 | 8.90 | 40.78 | 34.37 | 58.14 |
| **W25 H12 s50 box2** | 2 | 2.00 | 6.46 | **8.93** | 8.59 | 32.12 | 27.02 | **48.62** |
| W25 H6 s50 raw | 5 | 1.00 | 10.45 | 41.79 | 9.42 | 39.86 | 33.96 | 57.02 |
| **W25 H6 s50 box3** | 3 | 3.00 | 7.11 | **13.63** | 9.07 | 40.32 | 27.19 | **47.91** |
| **W25 H6 s50 hann3** | 3 | 2.67 | 8.95 | **26.14** | 10.56 | 41.56 | 27.78 | **48.90** |
| **W37 H6 s50 box2** | 2 | 2.00 | 8.80 | **16.81** | 10.81 | 36.15 | 25.10 | **44.30** |

Applying a width-2 boxcar kernel to $W=25, H=12$ reduces the prediction gradient norm $\|dL/d\hat{y}\|$ by a factor of 3.6 (from $32.40$ to $8.93$) and suppresses out-of-band noise from $58.14$ to $48.62\text{ nats}^2$.

### 4.3 Input sensitivity retention

We probed the input Jacobian across lookahead steps for models trained on intermediate kernel losses:

| Candidate | $M_{\text{out}}$ | $K_{\text{eff}}$ | Direct $\|W_u\|$ | Total $\|J_{\text{rollout}}\|$ | $k=1$ (0.02s) | $k=10$ (0.20s) | $k=50$ (1.00s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Welch 50** | 1 | 1.00 | 1.353 | **18.41** | 0.069 | 0.114 | **1.084** |
| **W25 H12 s50 raw** | 3 | 1.00 | 1.352 | **5.82** | 0.067 | 0.115 | **0.206** |
| **W25 H12 s50 box2** | 2 | 2.00 | 1.356 | **12.83** | 0.069 | 0.118 | **0.869** |
| **W25 H6 s50 box4** | 2 | 4.00 | 1.353 | **13.58** | 0.069 | 0.104 | **0.720** |
| **W37 H6 s50 box2** | 2 | 2.00 | 1.356 | **18.73** | 0.070 | 0.109 | **1.236** |

Raw STFT suffers severe long-range sensitivity decay ($0.206$ at step 50). Intermediate kernels restore input sensitivity:

- `W25_H12_s50_box2` recovers total sensitivity to **12.83** and step-50 sensitivity to **0.869** (retaining 80% of Welch's sensitivity while maintaining $M_{\text{out}} = 2$).
- `W37_H6_s50_box2` reaches total sensitivity **18.73**, matching Welch's $18.41$ while providing $1.35\text{ Hz}$ frequency resolution and $M_{\text{out}} = 2$ frames.

---

## 5. Synthesis and trade-off landscape

```text
Time Resolution (M_out)        Estimator Variance & Floor        Input Sensitivity (||J_roll||)
-----------------------        --------------------------        ------------------------------
Raw STFT (M=3..5)              High Noise (Floor = 4.45)         Collapsed (5.82, k=50: 0.21)
  │                              │                                 │
  ▼                              ▼                                 ▼
Intermediate Kernels (M=2..3)  Low Noise  (Floor = 2.15..2.55)   Restored  (12.83..18.73)
  │                              │                                 │
  ▼                              ▼                                 ▼
Welch Endpoint (M=1)           Averaged   (Floor = 4.64)         Preserved (18.41, k=50: 1.08)
```

The investigation clarifies three core properties of the loss space:

1. **Offline separation is an incomplete objective for control.** Offline AUC measures how cleanly a loss scores wrong spectra, but ignores whether the model achieves this by dropping sensitivity to control inputs. Evaluating the input Jacobian norm $\|\partial \text{rollout}/\partial u\|$ is required for candidate screening.
2. **Intermediate kernels break the trade-off.** Pre-log frame smoothing suppresses relative noise in background frequency bins without discarding intra-span time resolution.
3. **Optimal candidate operating points:**
   - **`W25_H12_s50_box2` ($W=25, H=12, K_{\text{eff}}=2.0, M_{\text{out}}=2$):** Preserves 2-frame time resolution at $2.0\text{ Hz}$ resolution, raises AUC to $0.838$ over Welch's $0.793$, and maintains strong rollout input sensitivity ($12.83$).
   - **`W37_H6_s50_box2` ($W=37, H=6, K_{\text{eff}}=2.0, M_{\text{out}}=2$):** Provides finer $1.35\text{ Hz}$ frequency resolution, matches Welch's input Jacobian norm ($18.73$), and preserves 2 time frames.

---

## 6. Recommendations

1. **Keep live training on Welch until retrained.** The live config [`nonlinear_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_mse02_psd.yaml) remains on `n_segment = n_hop = n_span = 50`.
2. **Promote `W25_H12_box2` and `W37_H6_box2` for closed-loop validation.** Retrain predictors on these two intermediate kernel configurations and run full closed-loop suppression sweeps across seeds.
3. **Include `eeg_ms` in multi-term training.** Because `EegMsLoss` explicitly scores broadband energy courses, pairing an intermediate time-resolved STFT term with `eeg_ms` provides an energy anchor that prevents input sensitivity decay.
4. **Monitor input sensitivity during training.** Compute $\|\partial \text{rollout}/\partial u\|$ during validation and reject checkpoints where rollout sensitivity falls below $6.5$.
