# Hyperparameter sweep strategy

This document lists the hyperparameter sweep experiments for surrogate Predictor models and closed-loop Model Predictive Control (MPC).

Training and offline evaluation horizons are at most $1.0\,\text{s}$. The MPC Control Horizon is fixed at $1.0\,\text{s}$ through experiment 2.3; experiment 2.4 may test longer lookahead. The Waveform curriculum MSE span is at most $0.25\,\text{s}$; a spectral loss may score the remainder of the $1.0\,\text{s}$ training rollout. The $700*$ Plant seeds are reserved for closed-loop evaluation and are distinct from training-data generation seeds. Historical champion configs are references, not baselines for this new sweep.

## Baseline and calibration before the sweeps

Freeze one explicit baseline per Predictor family before varying its hyperparameters. These are starting values for the new code and data, not claims that the historical champions remain optimal:

| Family | Initial baseline |
| --- | --- |
| Observable MLP | $f_s=50\,\text{Hz}$; $n_\text{seg}=25$, $n_\text{hop}=3$, $1$--$25\,\text{Hz}$, linear kernel $K=1$, asymmetric Hann; residual, $n_y=1$, $n_u=9$, depth 1, width 256, tanh. |
| Observable CNN | Same Observable geometry; residual, $n_y=5$, $n_u=9$, depth 2, 16 filters, $3\times3$ kernels, tanh. |
| Observable DMDc | Same Observable geometry; $n_y=1$, $n_u=9$, energy cutoff $0.99$, ridge parameter $10^{-6}$. |
| Waveform MLP | $f_s=100\,\text{Hz}$; residual, $n_y=15$, $n_u=10$, depth 1, width 256, softplus; curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$ with 50-sample segments, 25-sample hops and $1$--$25\,\text{Hz}$ band. |

The $100\,\text{Hz}$ Waveform rate is provisional until the rate decision; it represents $0.25\,\text{s}$ exactly. For neural baselines use AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$, batch size 256, 150 epochs and patience 35; curriculum expansion begins at epoch 10 and completes by epoch 80. Use a fixed training/validation split and preprocessing within each family. Record the resolved baselines in local experiment configs before starting a sweep. For every changed Observable geometry, recompute the minimum valid control history with `observable.min_past_controls()`.

The [gradient calibration experiment](../artifacts/gradient_weight_calibration/report.md) on the current code and separate 20 ms excitation recordings sets one pilot ratio for each joint objective:

1. **Training loss:** Fix $w_\text{mse}=1$ and $w_\text{stft}=0.01$. The measured unweighted parameter-gradient ratio changes with training stage, reaching 0.013 after 512 and 0.0054 after 1024 short MSE-only updates on separate probe recordings. The fixed value is a rounded pilot choice; gradient norms do not stay equal throughout training.
2. **MPC Cost:** Fix $w_y=1$ and $w_\text{hinge}=1000$ for a combined Cost. The unweighted tracking-to-hinge output-gradient ratio was about 1000 across seizure phases, using a 100 Hz healthy reference and the same future EEG for both terms. This measures Cost scaling with respect to predicted EEG, not the gradient through a trained Predictor to Control Current. Calibrate the overall state-cost-to-effort scale separately in the closed-loop effort experiment; matching the two output gradients does not establish the desired Delivered Charge.

Keep these relative weights fixed in the subsequent comparisons. Judge their actual value by closed-loop Seizure Burden, Delivered Charge, and solver behavior rather than running a weight-ratio sweep.

For the new Observable MLP baseline, use $w_\text{hinge}=10$, $w_u=1$, and $w_{u,l1}=0$. This is the tested operating point from `observable_dataset_sensitivity_comparison`; use it as a starting point for the newly trained Predictor, not as a claim that the optimum transfers to changed Observable geometry or training data. The Waveform output-gradient calibration above does not apply to an Observable MLP.

For the new Waveform MLP baseline, use $w_u=10$ and $w_{u,l1}=0$. This retains the historical Waveform effort scale as a starting value. Calibrate Waveform effort separately after training its new baseline checkpoint; the Observable MLP's $w_u=1$ does not transfer across the different state Costs and Predictor sensitivities.

---

## Metric decoupling principle

Hyperparameters are divided into two tiers based on their evaluation metric:

* **Tier 1 (Offline validation loss):** Used when the training loss composition, target state representation, and time step are strictly held constant. When the optimization objective does not vary across trials, validation loss measures dynamical forecasting accuracy, model capacity, and optimization efficiency.
* **Tier 2 (Closed-loop Plant score):** Used when altering the training loss composition, STFT geometry, temporal resolution, or controller cost formulation. Changing loss component weights (such as adding an STFT penalty to MSE) or altering Frame definitions changes the loss scale, entropy, or dimension, making offline validation loss mathematically incomparable across trials. These configurations must be evaluated by real-time Plant control performance: Seizure Burden, Delivered Charge, and Propagation Zone containment.

---

## Tier 1: Offline validation loss experiments

Tier 1 experiments run through Optuna (`scripts/sweep_nn_predictor.py`). All trials must satisfy three qualification gates before checkpoint export:

1. **Schedule eligibility:** Checkpoints are selected only after scheduled losses reach eligibility (`curr_end: 80` or later).
2. **Recursive rollout:** Validation loss is scored over a $1.0\,\text{s}$ recursive rollout.
3. **Stability filter:** Prune or reject candidates with amplitude growth ratio $> 3.0$ or terminal NMSE $> 1.25$.

### 1.1 Observable MLP optimization

* **Purpose:** Determine network capacity and optimizer parameters for one-step Observable Frame prediction under a fixed representation.
* **Target metric:** Multi-step `val_log_mse` over $1.0\,\text{s}$ recursive rollout.
* **Fixed configuration:** Fixed Observable geometry ($n_\text{seg}=25, n_\text{hop}=3$, `band_hz: [1.0, 25.0]`, $K=1$, asymmetric Hann), residual connection enabled.
* **Sub-experiments:**
  * **Sub-experiment A (Model capacity):**
    * `model.hidden_size`: $[128, 256, 512, 768, 1024]$
    * `model.depth`: $[1, 2, 3]$
    * `model.activation`: `[tanh, softplus]`
  * **Sub-experiment B (Optimization dynamics):**
    * `training.learning_rate`: Log-uniform $[1.0 \times 10^{-4}, 3.0 \times 10^{-3}]$
    * `training.weight_decay`: Log-uniform $[1.0 \times 10^{-6}, 1.0 \times 10^{-3}]$
    * `training.batch_size`: $[128, 256, 512]$

### 1.2 Observable CNN optimization

* **Purpose:** Exploit spatio-temporal structure across EEG channels and frequency bins under a fixed representation.
* **Target metric:** Multi-step `val_log_mse` across the full frame Control Horizon.
* **Fixed configuration:** Fixed Observable geometry, residual connection enabled.
* **Sub-experiments:**
  * **Sub-experiment A (Convolutional structure):**
    * Channel filters (`hidden_size`): $[16, 32, 64, 128]$
    * Depth: $[1, 2, 3]$
    * Time kernel size ($K_t$): $[3, 5]$
    * Frequency kernel size ($K_f$): $[3, 5]$
  * **Sub-experiment B (Optimizer tuning):**
    * `training.learning_rate`: Log-uniform $[1.0 \times 10^{-4}, 3.0 \times 10^{-3}]$
    * `training.weight_decay`: Log-uniform $[1.0 \times 10^{-6}, 1.0 \times 10^{-3}]$

### 1.3 Waveform MLP optimization

* **Purpose:** Tune network capacity and optimizer parameters for high-rate raw EEG prediction under a fixed joint loss.
* **Target metric:** Recursive rollout NMSE over $1.0\,\text{s}$.
* **Fixed configuration:** Provisional sampling rate $f_s = 100\,\text{Hz}$, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, calibrated fixed loss weights, residual connection enabled.
* **Sub-experiments:**
  * **Sub-experiment A (Model capacity):**
    * `model.hidden_size`: $[64, 128, 256, 512]$
    * `model.depth`: $[1, 2, 3, 4]$
    * `model.activation`: `[tanh, softplus]`
  * **Sub-experiment B (Optimizer tuning):**
    * `training.learning_rate`: Log-uniform $[1.0 \times 10^{-4}, 2.0 \times 10^{-3}]$
    * `training.weight_decay`: Log-uniform $[1.0 \times 10^{-6}, 1.0 \times 10^{-3}]$

### 1.4 Hankel-DMDc subspace optimization

* **Purpose:** Identify linear state-space operators using dynamic mode decomposition with control.
* **Target metric:** Multi-step linear recursive rollout `val_log_mse` over $1.0\,\text{s}$.
* **Fixed configuration:** Evaluated per knot interval ($n_\text{hop} \in \{3, 5\}$).
* **Sub-experiments:**
  * **Sub-experiment A (Embedding orders):**
    * Output delay order ($H_y$): $[1, 2, 4, 8]$
    * Control delay order ($H_u$): $[1, 2, 4, 8]$
  * **Sub-experiment B (SVD truncation):**
    * Energy cutoff: $[0.80, 0.85, 0.90, 0.95, 0.99, 0.999]$
  * **Sub-experiment C (Ridge regularization):**
    * `training.dmd_lambda`: $[0, 10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}]$, with the selected embedding and energy cutoff.

---

## Tier 2: Closed-loop plant score experiments

Tier 2 experiments evaluate the full closed-loop control system against the nonlinear Jansen-Rit Plant.

* **Simulation protocol:** Canonical closed-loop evaluation seeds `[7000, 7001, 7002, 7004, 7005]`. Predictor training data is generated with different seeds; these five seeds never enter Predictor fitting.
* **Duration:** $12.0\,\text{s}$ per run.
* **Primary metric:** Seizure Burden (time-averaged network fraction of seizing regions).
* **Secondary metrics:** Delivered Charge ($\int_0^T \|\mathbf{u}(t)\|_1 dt$), final seizing node count, solver solve time, and IPOPT success rate.

### 2.1 Controller effort weight calibration

* **Purpose:** Check whether the previously effective Observable MLP effort setting still balances suppression and Delivered Charge with the newly trained Predictor. This requires zero model retraining.
* **Target metric:** Seizure Burden and Delivered Charge on seeds 7001 and 7004.
* **Fixed configuration:** Newly trained baseline Observable MLP checkpoint, with $w_\text{hinge}=10$, $w_u=1$, and $w_{u,l1}=0$ as the reference arm; do not reuse a historical champion checkpoint.
* **Sub-experiments:**
  * **Sub-experiment A (Quadratic control weight $w_u$):**
    * Candidate values: $[0.5, 1.0, 2.0]$ with $w_\text{hinge}=10$ and $w_{u,l1}=0$; retain $w_u=1$ as the reference arm.
  * **Sub-experiment B (Sparse control weight $w_{l1}$):**
    * Candidate values: $[0, 0.01, 0.1]$ at the selected $w_u$, only if a sparse-control comparison is still wanted.

### 2.2 Observable STFT geometry exploration

* **Purpose:** Determine the optimal Observable representation balancing detection latency, frequency resolution, and estimator variance. Rather than testing all 72 combinatorial options, the search runs in four decoupled sub-experiments.
* **Target metric:** Seizure Burden, Delivered Charge, and recruitment detection latency across all canonical seeds.
* **Sub-experiments:**
  * **Sub-experiment A (Dynamical backbone grid):**
    * Explores the $2 \times 2$ matrix of segment length and knot interval.
    * Fixed settings: `band_hz: [1.0, 25.0]`, linear kernel $K=1$, asymmetric Hann.
    * Candidates:
      1. $n_\text{seg}=25, n_\text{hop}=3$ ($60\,\text{ms}$ knot, $0.5\,\text{s}$ segment, $2\,\text{Hz}$ resolution)
      2. $n_\text{seg}=50, n_\text{hop}=3$ ($60\,\text{ms}$ knot, $1.0\,\text{s}$ segment, $1\,\text{Hz}$ resolution)
      3. $n_\text{seg}=25, n_\text{hop}=5$ ($100\,\text{ms}$ knot, $0.5\,\text{s}$ segment, $2\,\text{Hz}$ resolution)
      4. $n_\text{seg}=50, n_\text{hop}=5$ ($100\,\text{ms}$ knot, $1.0\,\text{s}$ segment, $1\,\text{Hz}$ resolution)
  * **Sub-experiment B (Spectral band and DC screening):**
    * Evaluates spectral coverage and DC inclusion on the winning backbone from Sub-experiment A.
    * Fixed settings: Winning $(n_\text{seg}, n_\text{hop})$, linear kernel $K=1$, asymmetric Hann.
    * Candidates:
      1. $0\text{--}25\,\text{Hz}$ (Broadband with DC): Tests whether static baseline offset provides actionable stabilization information.
      2. $1\text{--}25\,\text{Hz}$ (Broadband AC): Drops DC offset, scores full spectrum up to Nyquist.
      3. $3\text{--}12\,\text{Hz}$ (Seizure band): Targets pathological theta and alpha synchronization directly.
  * **Sub-experiment C (Linear frame kernel width):**
    * Evaluates temporal smoothing along the Frame sequence on the winning backbone and band.
    * Fixed settings: Winning $(n_\text{seg}, n_\text{hop})$, winning `band_hz`, asymmetric Hann.
    * Candidates:
      1. $K = 1$: Unsmoothed periodogram. Minimum delay, highest variance.
      2. $K = 4$: Linear ramp weights $[1, 2, 3, 4] / 10$. Moderate smoothing.
      3. $K = 7$: Linear ramp weights $[1, \dots, 7] / 28$. Strong smoothing.
  * **Sub-experiment D (Window taper causal delay ablation):**
    * Tests the group delay benefit of causal windowing on the top overall geometry.
    * Fixed settings: Winning $(n_\text{seg}, n_\text{hop})$, winning `band_hz`, winning $K$.
    * Candidates:
      1. Asymmetric Hann (`asymmetric_window: true`): Energy concentrated at newest samples.
      2. Symmetric Hann (`asymmetric_window: false`): Standard Hann window with energy center at $N/2$.

### 2.3 Waveform Loss and Cost formulation

* **Purpose:** Determine the optimal training loss composition and receding-horizon cost function for Waveform models, testing the alignment between time-domain tracking and spectral excess penalties and evaluating whether joint supervision provides a closed-loop advantage over pure single-objective control.
* **Target metric:** Seizure Burden, Delivered Charge, Propagation Zone containment, and IPOPT solve time per decision step across canonical seeds.
* **Fixed configuration:** Baseline Waveform MLP architecture and sampling rate from the calibration section, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, and Waveform effort baseline $w_u=10$, $w_{u,l1}=0$. The joint loss uses $w_\text{mse}=1$, $w_\text{stft}=0.01$; the joint Cost uses $w_y=1$, $w_\text{hinge}=1000$.
* **Sub-experiments:**
  * **Pre-experiment (Waveform effort check):** With one newly trained baseline Waveform checkpoint and $w_{u,l1}=0$, compare $w_u=[3,10,30]$ under the fixed combined Cost. Use the selected effort weight unchanged across the loss-Cost arms below; report Seizure Burden and Delivered Charge together.
  * **Sub-experiment A (Loss-Cost alignment):**
    * Evaluates the fundamental correspondence between training objective and controller objective:
      * Arm 1 (Pure time-domain): Pure MSE Loss ($w_\text{mse} = 1.0, w_\text{stft} = 0.0$) paired with Raw Voltage Tracking Cost ($w_\text{hinge} = 0.0$).
      * Arm 2 (Pure spectral): Pure STFT Loss ($w_\text{mse} = 0.0, w_\text{stft}=0.01$) paired with Spectral Frame Hinge Cost ($w_y = 0.0, w_\text{hinge}=1000$).
      * Arm 3 (Joint loss under tracking): Calibrated MSE + STFT Loss paired with Raw Voltage Tracking Cost.
      * Arm 4 (Joint loss under hinge): Calibrated MSE + STFT Loss paired with Spectral Frame Hinge Cost.
      * Arm 5 (Joint loss under combined cost): Calibrated MSE + STFT Loss paired with the calibrated Tracking + Hinge Cost.
  * **Sub-experiment B (Terminal state weighting):**
    * Depending on which cost objective is active (voltage tracking vs. spectral hinge), evaluate uniform stage weighting against boosted terminal penalties to stabilize finite-horizon boundary behavior:
      * Voltage tracking terminal weight: $w_{y,\text{terminal}} \in [1 \times, 2 \times, 5 \times, 10 \times] w_y$
      * Spectral hinge terminal weight: $w_{\text{hinge},\text{terminal}} \in [1 \times, 2 \times, 5 \times, 10 \times] w_\text{hinge}$

### 2.4 Control Horizon lookahead sweep

* **Purpose:** Determine the optimal planning lookahead for the receding-horizon controller, trading off long-range trajectory foresight against IPOPT solve time per decision step.
* **Target metric:** Seizure Burden, solve time, and IPOPT iteration count.
* **Fixed configuration:** Champion Predictor checkpoint and calibrated cost weights.
* **Sub-experiments:**
  * Lookahead durations: $[0.6\,\text{s}, 0.8\,\text{s}, 1.0\,\text{s}, 1.2\,\text{s}, 1.5\,\text{s}]$
  * Corresponding steps depend on the knot interval ($H = T / \Delta t$).

### 2.5 Predictor family champion comparison

* **Purpose:** Benchmark the final champion model from each Predictor family under identical closed-loop Plant conditions to determine the top overall architecture.
* **Target metric:** Seizure Burden, Delivered Charge, and Propagation Zone containment across all canonical seeds.
* **Arms:**
  1. `uncontrolled`: No stimulation baseline.
  2. `threshold`: Static heuristic stimulation triggered on regional amplitude.
  3. `obs_mlp_champion`: Top Observable MLP model.
  4. `obs_cnn_champion`: Top Observable CNN model.
  5. `obs_dmd_champion`: Top Hankel-DMDc model.
  6. `waveform_mlp_champion`: Top Waveform MLP model.
