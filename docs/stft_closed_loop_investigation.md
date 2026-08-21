# Why STFT loss degraded closed-loop seizure control

Offline rank separation picked `n_segment = 25` and `n_hop = 12` as the best spectrogram loss geometry, scoring 0.885 against 0.793 for the single-frame Welch endpoint. When deployed in the closed-loop MPC, that predictor failed. Seizure burden jumped from 0.052 to 0.217, suppressed seeds dropped from 3/3 to 1/3, and delivered charge rose 3.4x.

This note records the mechanism behind that failure.

Source files:

- Training loss implementation: [`StftLoss`](../src/neuro/predictor/losses.py)
- Autoregressive model: [`AutoregressiveMLP`](../src/neuro/predictor/module.py)
- MPC formulation and spectral hinge cost: [`_spectral_hinge_cost`, `MPCNlp`](../src/neuro/control/nlp.py)
- Simulation config: [`mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml)
- Prior loss geometry guide: [`spectrogram_loss_guide.md`](spectrogram_loss_guide.md)

---

## 1. Summary of findings

The failure is caused by an interaction between short-window log-power penalties, autoregressive input attenuation, and the online MPC objective.

1. **Short FFT windows penalize control transients in low-power bins.** A 25-sample window (0.5 s at 50 Hz) has coarse 2 Hz resolution and 4 Hz main-lobe spectral leakage. Control inputs in the training data create high-frequency local transients. In log-power space, errors in low-power high-frequency bins carry large relative gradients ($dL/d\hat{y} = 49.5$ for STFT versus $31.4$ for Welch).
2. **Backpropagation attenuates input weights.** To minimize short-window log-spectral residuals, the optimizer reduces the input weight norm $W_u$ relative to autoregressive weights $W_y$. The ratio $\|W_u\| / \|W_y\|$ falls from 0.1456 to 0.1093.
3. **Attenuation compounds across lookahead steps.** In an autoregressive model, reducing direct input sensitivity shrinks the rollout Jacobian over time. Total rollout sensitivity $\|\partial \text{rollout}/\partial u\|$ drops 3.2x (from 6.81 to 2.12).
4. **The MPC control gradient collapses 7x.** At seizure onset, the MPC objective gradient $\|\nabla_u J\|$ drops from 237.88 to 34.16. The optimizer perceives an 8.1x smaller cost reduction from stimulation.
5. **Spatial steering breaks down and saturates.** Welch drives a coordinated dipole between electrode 0 and electrodes 1 and 2 (cross-channel correlations of -0.93 and -0.94). STFT loses this coordination (correlations of -0.26 and -0.28) and drives inputs to saturation ($u_{\text{max}} = 2.0$). The incoherent stimulation fails to desynchronize the plant, leaving the loop stuck in a high-charge, high-burden state.

---

## 2. Loss gradient mechanics during training

`StftLoss` scores log power per frame:

$$\mathcal{L} = \frac{1}{B C M F} \sum_{b,c,m,f} \left(\log(\hat{P}_{b,c,m,f} + \epsilon) - \log(P_{b,c,m,f} + \epsilon)\right)^2$$

With `n_segment = 25`, `n_hop = 12`, a 1-second rollout ($T=50$) splits into $M=3$ frames. The Welch endpoint (`n_segment = 50`, `n_hop = 50`) uses $M=1$ frame.

### 2.1 Per-bin residuals

In training data from `data/experiment_excited_roast/train`, true power in high-frequency bins ($>12$ Hz) is small ($10^{-2}$ to $10^{-3} \text{ mV}^2/\text{Hz}$). Because the loss operates after the log, relative errors dominate:

| Frequency | True power ($\text{mV}^2/\text{Hz}$) | Mean squared log residual (STFT $W=25$) | Mean squared log residual (Welch $W=50$) |
| ---: | ---: | ---: | ---: |
| 0 Hz | 45.6 | 3.19 | 2.72 |
| 4 Hz | 4.03 | 7.50 | 9.13 |
| 8 Hz | 7.41 | 11.68 | 13.76 |
| 10 Hz | 8.23 | 12.53 | 14.05 |
| 14 Hz | 0.48 | 15.01 | 14.94 |
| 18 Hz | 0.23 | 12.33 | 10.00 |
| 22 Hz | 0.017 | 7.97 | 6.27 |

When stimulation $u$ is applied in training trajectories, it introduces high-frequency energy. In a 25-sample window, any mismatch in timing or phase between predicted and true response produces large log-power errors in those background bins.

### 2.2 Gradient backpropagation to model weights

Measuring loss gradients on a batch of 64 training trajectories:

| Loss term | Loss value | $\|dL/d\hat{y}\|$ | Total $\|dL/d\theta\|$ | $\|\nabla_{W_u} L\|$ | $\|\nabla_{W_y} L\|$ | Ratio $W_u/W_y$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curriculum MSE | 0.652 | 0.0036 | 19.56 | 0.21 | 6.87 | 0.030 |
| Welch STFT ($W=50, H=50$) | 10.46 | 31.42 | 183.24 | 14.99 | 102.70 | 0.146 |
| STFT ($W=25, H=12$) | 10.67 | 49.47 | 203.22 | 10.07 | 37.51 | 0.268 |

The STFT loss creates higher prediction gradient norm ($49.47$ vs $31.42$). The resulting model has smaller input weights:

- **Welch model:** $\|W_u\| = 1.3214$, $\|W_y\| = 9.0724$, ratio = **0.1456**
- **STFT model:** $\|W_u\| = 1.0659$, $\|W_y\| = 9.7521$, ratio = **0.1093**

The STFT model lowers its training loss by attenuating its response to $u$, ensuring predicted rollouts stay smooth and avoid short-window log-power penalties.

---

## 3. Autoregressive compounding of input sensitivity

In `AutoregressiveMLP`, past outputs feed back into the input vector. A smaller $W_u$ causes the sensitivity to stimulation to decay faster over the rollout:

$$\frac{\partial \hat{y}_{t+k}}{\partial u_t}$$

Evaluating this Jacobian across test trajectories:

| Lookahead step $k$ | Lookahead time | Welch $\|\partial \hat{y}_{t+k} / \partial u_t\|$ | STFT $\|\partial \hat{y}_{t+k} / \partial u_t\|$ | Ratio (STFT / Welch) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02 s | 0.5386 | 0.3119 | 0.58x |
| 3 | 0.06 s | 0.9028 | 0.2825 | **0.31x** |
| 5 | 0.10 s | 0.7423 | 0.2954 | 0.40x |
| 10 | 0.20 s | 0.7453 | 0.2668 | **0.36x** |
| 20 | 0.40 s | 0.4676 | 0.2067 | 0.44x |
| 30 | 0.60 s | 0.3814 | 0.1189 | **0.31x** |
| 50 | 1.00 s | 0.0352 | 0.0051 | **0.14x** |
| **Total rollout** | **1.00 s** | **6.8075** | **2.1249** | **0.31x** |

At step $k=3$, sensitivity is down by a factor of 3.2. By step $k=50$, it is down by a factor of 7.0. The STFT model relies almost entirely on its internal autoregressive history to predict spectral shape.

---

## 4. Online MPC gradient collapse and optimization failure

The MPC controller solves an NLP over horizon $H=75$ steps:

$$J(u) = \frac{w_u}{H} \sum_{k=0}^{H-1} \|u_k\|^2 + w_{\text{psd}} \cdot \text{Hinge}(\hat{y}(u))$$

where $w_u = 10.0$, $w_{\text{psd}} = 1000.0$, and $\text{Hinge}$ scores excess power above `data/healthy_psd.npz`.

The gradient with respect to control sequence $u$ is:

$$\nabla_u J = \frac{2 w_u}{H} u + w_{\text{psd}} \left( \frac{\partial \text{Hinge}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial u} \right)$$

Evaluating the CasADi NLP graph at a seizure state ($u = 0$):

| Metric | Welch predictor | STFT predictor |
| :--- | ---: | ---: |
| Initial cost $J(u=0)$ | 7432.75 | 4250.77 |
| Initial gradient $\|\nabla_u J\|$ | 237.88 | 34.16 |
| Gradient ratio (STFT / Welch) | 1.00 | **0.14x (7.0x smaller)** |
| Optimal cost after IPOPT | 5894.72 | 4061.11 |
| Predicted cost reduction $\Delta J$ | **-1538.03** | **-189.65 (8.1x smaller)** |

### 4.1 Loss of spatial coordination

Because the STFT gradient is weak and noisy across frames, IPOPT converges to a different, ineffective spatial pattern.

Cross-channel correlation matrix of commanded controls $u(t)$:

```text
Welch:
[[ 1.000, -0.931, -0.942],
 [-0.931,  1.000,  0.753],
 [-0.942,  0.753,  1.000]]

STFT:
[[ 1.000, -0.277, -0.854],
 [-0.277,  1.000, -0.263],
 [-0.854, -0.263,  1.000]]
```

Welch coordinates electrode 0 against electrodes 1 and 2 as a structured dipole source. STFT loses the coupling between channels 1 and 2 (-0.263) and drives erratic currents.

### 4.2 Control saturation and closed-loop trap

Because the STFT model perceives stimulation as having low authority, the optimizer pushes $u$ to its hard limits ($u_{\text{max}} = 2.0$) trying to achieve small spectral changes.

In the real plant:

- Uncoordinated saturated stimulation fails to desynchronize the neural population.
- The seizure persists.
- The controller continues to see high spectral error and commands saturated current.
- Delivered charge increases 3.4x while seizure burden increases 4x.

---

## 5. Conclusions and implications

1. **Offline spectral separation does not measure control authority.** A loss can separate spectral shapes accurately by relying solely on autoregressive continuation while ignoring inputs. Offline metrics must evaluate input sensitivity $\partial \hat{y} / \partial u$ alongside spectral error.
2. **Frequency-domain loss terms need energy anchors.** Pure log-spectral loss terms over short windows distort broadband energy and input Jacobians. Pairing time-resolved spectral terms with broadband power terms (such as `EegMsLoss`) is required to preserve input coupling.
3. **Keep the live config on Welch.** Until a composite loss preserves the input Jacobian norm ($\|\partial \text{rollout}/\partial u\| \ge 6.5$), the single-frame Welch endpoint (`n_segment = 50, n_hop = 50`) remains the correct training objective for closed-loop control.
