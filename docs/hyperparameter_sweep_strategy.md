# Hyperparameter sweep strategy

This document lists the hyperparameter sweep experiments for surrogate Predictor models and closed-loop Model Predictive Control (MPC).

Training and offline evaluation horizons are at most $1.0\,\text{s}$. The MPC Control Horizon is fixed at $1.0\,\text{s}$ through experiment 2.3; experiment 2.4 may test longer lookahead. The Waveform curriculum MSE span is at most $0.25\,\text{s}$; a spectral loss may score the remainder of the $1.0\,\text{s}$ training rollout. The $700*$ Plant seeds are reserved for closed-loop evaluation and are distinct from training-data generation seeds. Historical champion configs are references, not baselines for this new sweep.

Since training happens mostly on GPU and closed-loop runs mostly on CPU experiments can be pipelined. Once a predictors Tier 1 experiment is done it's Tier 2 experiments can already start while other predictors are training. This has to be carefully managed as to not exhaust the computers resources.

## Baseline and calibration before the sweeps

Freeze one explicit baseline per Predictor family before varying its hyperparameters. These are starting values for the new code and data, not claims that the historical champions remain optimal:

| Family | Initial baseline |
| --- | --- |
| Observable MLP | $f_s=50\,\text{Hz}$; $n_\text{seg}=25$, $n_\text{hop}=3$, $1$--$25\,\text{Hz}$, linear kernel $K=1$, asymmetric Hann; residual, $n_y=1$, $n_u=9$, depth 1, width 256, tanh. |
| Observable CNN | Same Observable geometry; residual, $n_y=5$, $n_u=9$, depth 2, 16 initial filters, $3\times3$ kernels, tanh; `tapering_ratio: 1.0`, `head_depth: 1`, `head_width: trunk`; batch size 512, learning rate $2\times10^{-4}$; retrain with activation after every convolution. |
| Observable DMDc | Same Observable geometry; $n_y=1$, $n_u=9$, energy cutoff $0.99$, ridge parameter $10^{-6}$. |
| Waveform MLP | $f_s=100\,\text{Hz}$; residual, $n_y=15$, $n_u=10$, depth 1, width 256, softplus; curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$ with 50-sample segments, 25-sample hops and $1$--$25\,\text{Hz}$ band. |
| Waveform CNN | $f_s=100\,\text{Hz}$; residual, $n_y=15$, $n_u=10$, depth 2, 32 initial filters, $3$ temporal kernel, tanh; `tapering_ratio: 1.0`, `head_depth: 1`, `head_width: trunk`; batch size 512, learning rate $2\times10^{-4}$; retrain with activation after every convolution; curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$ with 50-sample segments, 25-sample hops and $1$--$25\,\text{Hz}$ band. |

The $100\,\text{Hz}$ Waveform rate is provisional until the rate decision; it represents $0.25\,\text{s}$ exactly. For neural baselines use AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$, batch size 256, 150 epochs and patience 35; curriculum expansion begins at epoch 10 and completes by epoch 80. For both CNN baselines, resource pilots, and architecture sweeps, use batch size 512 and learning rate $2\times10^{-4}$: double the learning rate alongside the increase from batch size 256 to 512. This is a starting scaling choice to validate in the pilots; keep these settings fixed across architecture candidates. Use a fixed training/validation split and preprocessing within each family. Record the resolved baselines in local experiment configs before starting a sweep. For every changed Observable geometry, recompute the minimum valid control history with `observable.min_past_controls()`.

The [gradient calibration experiment](../artifacts/gradient_weight_calibration/report.md) on the current code and separate 20 ms excitation recordings sets one pilot ratio for each joint objective:

1. **Training loss:** Fix $w_\text{mse}=1$ and $w_\text{stft}=0.01$. The measured unweighted parameter-gradient ratio changes with training stage, reaching 0.013 after 512 and 0.0054 after 1024 short MSE-only updates on separate probe recordings. The fixed value is a rounded pilot choice; gradient norms do not stay equal throughout training.
2. **MPC Cost:** Fix $w_y=1$ and $w_\text{hinge}=1000$ for a combined Cost. The unweighted tracking-to-hinge output-gradient ratio was about 1000 across seizure phases, using a 100 Hz healthy reference and the same future EEG for both terms. This measures Cost scaling with respect to predicted EEG, not the gradient through a trained Predictor to Control Current. Calibrate the overall state-cost-to-effort scale separately in the closed-loop effort experiment; matching the two output gradients does not establish the desired Delivered Charge.

Keep these relative weights fixed in the subsequent comparisons. Judge their actual value by closed-loop Seizure Burden, Delivered Charge, and solver behavior rather than running a weight-ratio sweep.

For the new Observable MLP baseline, use $w_\text{hinge}=10$, $w_u=1$, and $w_{u,l1}=0$. This is the tested operating point from `observable_dataset_sensitivity_comparison`; use it as a starting point for the newly trained Predictor, not as a claim that the optimum transfers to changed Observable geometry or training data. The Waveform output-gradient calibration above does not apply to an Observable MLP.

For the new Waveform MLP baseline, use $w_u=10$ and $w_{u,l1}=0$. This retains the historical Waveform effort scale as a starting value. Calibrate Waveform effort separately after training its new baseline checkpoint; the Observable MLP's $w_u=1$ does not transfer across the different state Costs and Predictor sensitivities. Waveform CNN adopts the same initial effort baseline ($w_u=10, w_{u,l1}=0$) under the continuous waveform controller.

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

### 1.2 Observable CNN architecture rerun

* **Status:** Required fresh sweep. The initial and expanded old-architecture sweeps and standalone champion run are archived in [legacy CNN sweeps](../artifacts/legacy_cnn_sweeps/report.md). Their results remain historical evidence, not completed trials or champions for the updated architecture.
* **Purpose:** Evaluate causal time-frequency feature extraction, tapering, dense-head capacity, and direct versus residual prediction under a fixed Observable representation.
* **Target metric:** Multi-step `val_log_mse` over the $1.0\,\text{s}$ recursive rollout.
* **Fixed configuration:** Baseline Observable geometry from the calibration section, $n_u=9$, AdamW learning rate $2\times10^{-4}$, weight decay $10^{-5}$, batch size 512, and the baseline training schedule and split.
* **Sub-experiment A (Architecture):**
  * `model.n_y`: Integer range $[3,12]$; effective history is bounded by the receptive field below.
  * `model.hidden_size`: $[16,32,64,128]$, the first convolution's filter count. Restore 128 with tapering; its earlier exclusion was a runtime constraint, not evidence of poor forecasting performance.
  * `model.depth`: $[1,2]$; reconsider depth 3 with tapering, subject to resource checks.
  * `model.kernel_size`: $[2,3,4,5]$.
  * `model.frequency_kernel_size`: $[1,2,3,4,5,7]$.
  * `model.activation`: `[tanh, softplus]`.
  * `model.tapering_ratio`: Float range $[0.5,1.0]$.
  * `model.head_depth`: $[0,1,2]$ hidden dense layers.
  * `model.head_width`: `[trunk, geometric_mean]`.
  * `model.residual`: `[true, false]`; predict $\Delta y$ or $y$, respectively, and score both on reconstructed Observable Frames.
* **Sub-experiment B (Optimizer tuning):** Freeze a qualified architecture, then compare learning rate log-uniform $[10^{-4},3\times10^{-3}]$ and weight decay log-uniform $[10^{-6},10^{-3}]$ in a separate fresh study. The old skipped optimizer stage is not a completed experiment.

#### CNN rerun protocol shared with Section 1.4

* **Sampler and trial budget:** Run 25 architecture trials per family in fresh studies, using `TPESampler(n_startup_trials=15, multivariate=True, seed=training.seed)`. This gives 15 random startup trials followed by 10 TPE-guided trials when all trials finish successfully. Startup accounting includes completed and pruned trials, but excludes failures; failed trials can therefore reduce the number of guided trials within the 25-trial budget. The sampler seed follows the fixed training seed. This runner change does not implement conditional sampling or enqueue anchor configurations.
* **Architecture definition:** Every hidden convolution is followed by the chosen activation, including the final convolution. Select newest-time features, retain frequency positions for Observable Frames, concatenate flattened Control Current history, apply `[dense + activation]` $\times$ `head_depth`, and finish with a linear output. `residual: true` adds the latest measurement; `false` predicts the next measurement directly. Convolutions use stride one, without dilation or internal residual blocks.
* **Tapering and head widths:** $H_1=\texttt{hidden_size}$ and $H_{l+1}=\lfloor H_l r\rfloor$; only positive channel widths are eligible. `trunk` uses $H_L$ for each hidden dense layer; `geometric_mean` uses $\lfloor\sqrt{D_{in}D_{out}}\rfloor$. Head width has no effect with `head_depth: 0`; tapering has no effect with `depth: 1`. Avoid trials that differ only in inactive settings.
* **Linear-head ablation:** `head_depth: 0` imposes additive, fixed one-step Control Current sensitivity. Hidden head layers permit measurement-dependent control interactions. Both choices retain a nonlinear convolutional trunk.
* **Effective history:** Record $R_t=1+\texttt{depth}(\texttt{kernel_size}-1)$ and $n_{effective}=\min(n_y,R_t)$ for every trial. Older measurements beyond $R_t$ cannot affect the newest-time features. For otherwise fixed settings, canonicalize histories exceeding $R_t$ to one representative value rather than repeat equivalent models. For example, the Waveform baseline with $n_y=15$, depth 2, and kernel 3 uses only five samples of measurement history. This update does not introduce dilation.
* **Fresh execution and storage:** Retrain explicit baselines and all candidates from scratch. Do not resume archived SQLite studies, reuse old checkpoints, or skip combinations because they ran under the old architecture. Use new experiment folders under `artifacts/cnn_architecture_rerun/observable/` and `artifacts/cnn_architecture_rerun/waveform/`, with separate `architecture/` and `optimizer/` studies. Keep self-contained baseline and trial configs in those folders.
* **Resource and qualification checks:** Pilot the baseline and representative widest/deepest candidates before launching the 25-trial study, including width-128 tapered stacks such as $128\to64\to32$ at depth 3 and ratio 0.5. Width 128 is eligible in both families; assess runtime for the full architecture rather than excluding the initial width alone. Tapering does not shrink the first convolution and has no effect at depth 1. Record parameter count, actual layer widths, memory use, training time, and controller derivative/solve cost for finalists. Retain the $\le3\,\text{h}$ per-trial target; tapering alone does not establish runtime or recursive stability. Apply the Tier 1 schedule, rollout, and stability gates. Conditional deduplication and resource limits must be implemented in the run configuration or runner before launch; this document does not imply that the current runner already enforces them.
* **Reproducibility:** Save the source revision and any uncommitted patch, dependency versions, resolved configs, training-data identity and split, seeds, and activation-after-every-convolution convention in each new report. Old checkpoints may load in current code but produce different predictions; reproduce archived results only with the historical architecture. Freeze newly qualified champions for Tier 2 work.

### 1.3 Waveform MLP optimization

* **Purpose:** Tune network capacity and optimizer parameters for high-rate raw EEG prediction under a fixed joint loss.
* **Target metric:** Recursive rollout NMSE over $1.0\,\text{s}$.
* **Fixed configuration:** Provisional sampling rate $f_s = 100\,\text{Hz}$, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, calibrated fixed loss weights, residual connection enabled.
* **Sub-experiments:**
  * **Sub-experiment A (Model capacity):**
    * `model.hidden_size`: $[64, 128, 256]$ (pruned $512$ to maintain $\le 3\,\text{h}$ training budget)
    * `model.depth`: $[1, 2, 3]$ (pruned $4$ to maintain $\le 3\,\text{h}$ training budget)
    * `model.activation`: `[tanh, softplus]`
  * **Sub-experiment B (Optimizer tuning):**
    * `training.learning_rate`: Log-uniform $[1.0 \times 10^{-4}, 2.0 \times 10^{-3}]$
    * `training.weight_decay`: Log-uniform $[1.0 \times 10^{-6}, 1.0 \times 10^{-3}]$
  * **Sub-experiment C (Autoregressive history length extension):**
    * *Status:* Prospective extension. The initial capacity sweep froze $n_y = 15$ ($150\,\text{ms}$ at $100\,\text{Hz}$). If raw waveform state observability requires differing history context, evaluate $n_y \in [5, 10, 15, 20, 25]$ (spans $50\,\text{ms}$ to $250\,\text{ms}$) on the frozen capacity and optimizer champion.

### 1.4 Waveform CNN architecture rerun

* **Status:** Required fresh sweep. The old execution stopped during Trial 0 and produced no completed trial or champion. Its configs and study are archived in [legacy CNN sweeps](../artifacts/legacy_cnn_sweeps/report.md). The former Section 1.6 described this same interrupted campaign and is consolidated here.
* **Purpose:** Evaluate convolutional capacity, effective temporal history, tapering, dense-head structure, activation, and direct versus residual prediction under the calibrated fixed joint Loss. Follow the shared CNN protocol in Section 1.2.
* **Target metric:** Recursive rollout NMSE over $1.0\,\text{s}$ (`rollout_nmse`).
* **Fixed configuration:** Provisional $f_s=100\,\text{Hz}$, $n_u=10$, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, weights $w_\text{mse}=1$, $w_\text{stft}=0.01$, AdamW learning rate $2\times10^{-4}$, weight decay $10^{-5}$, batch size 512, and the baseline training schedule and split. Use the same batch size for every architecture candidate; freeze any throughput-driven change before launching the study.
* **Sub-experiment A (Architecture):**
  * `model.n_y`: Integer range $[5,15]$; deduplicate histories exceeding the receptive field as described in Section 1.2.
  * `model.hidden_size`: $[16,32,64,128]$ initial filters. Restore 128 with tapering; the interrupted old campaign did not establish a performance reason to exclude it.
  * `model.depth`: $[1,2]$, subject to the pilot resource checks.
  * `model.kernel_size`: $[3,5,7, 9, 11, 13, 15]$.
  * `model.activation`: `[tanh, softplus]`.
  * `model.tapering_ratio`: Float range $[0.5, 1.0]$.
  * `model.head_depth`: $[0,1,2]$.
  * `model.head_width`: `[trunk, geometric_mean]`.
  * `model.residual`: `[true, false]`; score both on reconstructed EEG with the same Loss.
* **Sub-experiment B (Optimizer tuning):** Freeze a qualified structural champion and use a separate fresh study for learning rate log-uniform $[10^{-4},2\times10^{-3}]$ and weight decay log-uniform $[10^{-6},10^{-3}]$.

### 1.5 Hankel-DMDc subspace optimization

* **Purpose:** Identify linear state-space operators using dynamic mode decomposition with control.
* **Target metric:** Multi-step linear recursive rollout `val_log_mse` over $1.0\,\text{s}$.
* **Fixed configuration:** Evaluated per knot interval ($n_\text{hop} \in \{3, 5\}$).
* **Sub-experiments:**
  * **Sub-experiment A (Embedding orders):**
    * Output delay order ($H_y$): $[1, 2, 4, 8]$
    * Control delay order ($H_u$): $[1, 2, 4, 8]$
  * **Sub-experiment B (SVD truncation):**
    * Energy cutoff: $[0.80, 0.85, 0.90, 0.95, 0.99, 0.999, 1.0]$
  * **Sub-experiment C (Ridge regularization):**
    * `training.dmd_lambda`: $[0, 10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}]$, with the selected embedding and energy cutoff.

### 1.6 Historical Waveform CNN campaign (archived)

The old structure/capacity campaign was interrupted with no completed trials. Its execution record is preserved in [the archived Waveform report](../artifacts/legacy_cnn_sweeps/sweep_waveform_cnn_structure/report.md). Section 1.4 now defines the single replacement Waveform CNN campaign; Section 1.6 is retained only to resolve historical references.

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

### 2.2 Observable MLP STFT geometry exploration

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

### 2.3 Observable CNN STFT geometry exploration

* **Purpose:** Determine whether changing the Observable representation improves closed-loop control with the tuned Observable CNN, while balancing detection latency, frequency resolution, and estimator variance.
* **Prerequisite:** Complete the updated-architecture Observable CNN rerun in Section 1.2 at the baseline geometry. Historical results in `artifacts/observable_cnn_stft_geometry_exploration/` used the old architecture and do not qualify the updated Predictor. Preserve those outputs and use a fresh directory for a geometry rerun. Freeze its selected architecture, optimizer, training schedule, and other non-geometry settings as the CNN geometry-sweep baseline. Calibrate the Observable CNN controller effort weight on its newly trained baseline checkpoint before the geometry sweep; do not assume the Observable MLP effort setting transfers to the CNN.
* **Target metric:** Seizure Burden, Delivered Charge, recruitment detection latency, solver solve time, and IPOPT success rate across all canonical seeds. Offline validation loss is a qualification and stability check, not the cross-geometry ranking metric.
* **Training protocol:** Retrain the frozen CNN configuration from scratch for every candidate geometry using the same training data, training/validation split, preprocessing procedure, and training-data seeds. Regenerate geometry-dependent standardizers, checkpoints, healthy references, and controller artifacts for each candidate. Keep the CNN hyperparameters and calibrated controller cost weights fixed across candidates. For every geometry, recompute the minimum valid control history with `observable.min_past_controls()`.
* **Simulation protocol:** Use the Tier 2 canonical seeds `[7000, 7001, 7002, 7004, 7005]`, Plant duration $12.0\,\text{s}$, and otherwise identical Plant and controller settings. Set controller `dt` and horizon steps to preserve each candidate's knot interval and the $1.0\,\text{s}$ Control Horizon: `n_hop=3` uses $dt=0.06\,\text{s}$ and 17 steps; `n_hop=5` uses $dt=0.10\,\text{s}$ and 10 steps.
* **Sub-experiments:** Use the same four-stage, decoupled candidate sequence as the Observable MLP exploration in Section 2.2. Advance each stage using the selected candidate from the preceding stage:
  * **Sub-experiment A (Dynamical backbone grid):** Four combinations of `n_segment` $\in \{25,50\}$ and `n_hop` $\in \{3,5\}$; fix `band_hz: [1.0, 25.0]`, linear kernel $K=1$, and asymmetric Hann.
  * **Sub-experiment B (Spectral band and DC screening):** On the winning backbone from A, compare `band_hz: [0.0, 25.0]`, `[1.0, 25.0]`, and `[3.0, 12.0]`; fix $K=1$ and asymmetric Hann.
  * **Sub-experiment C (Linear frame kernel width):** On the winning backbone and band from B, compare $K \in \{1,4,7\}$ with linear ramp weights and asymmetric Hann.
  * **Sub-experiment D (Window taper causal delay ablation):** On the winning backbone, band, and $K$ from C, compare asymmetric and symmetric Hann.
* **Selection and reporting:** Apply the same predeclared selection rule at each stage, considering Seizure Burden and Delivered Charge together and using recruitment detection latency as the latency measure. Report all candidate results, seed-level values, aggregation method, solver outcomes, and the selected geometry. Do not select by offline loss alone or compare raw validation losses across different geometries.

### 2.4 Waveform Loss and Cost formulation

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

### 2.5 Control Horizon lookahead sweep

* **Purpose:** Determine the optimal planning lookahead for the receding-horizon controller, trading off long-range trajectory foresight against IPOPT solve time per decision step.
* **Target metric:** Seizure Burden, solve time, and IPOPT iteration count.
* **Fixed configuration:** Champion Predictor checkpoint and calibrated cost weights.
* **Sub-experiments:**
  * Lookahead durations: $[0.6\,\text{s}, 0.8\,\text{s}, 1.0\,\text{s}, 1.2\,\text{s}, 1.5\,\text{s}]$
  * Corresponding steps depend on the knot interval ($H = T / \Delta t$).

### 2.6 Predictor family champion comparison

* **Purpose:** Benchmark the final champion model from each Predictor family under identical closed-loop Plant conditions to determine the top overall architecture.
* **Target metric:** Seizure Burden, Delivered Charge, and Propagation Zone containment across all canonical seeds.
* **CNN eligibility:** Both CNN arms must use qualified, newly trained champions from Sections 1.2 and 1.4. Archived old-architecture champions must not populate these arms under the updated runtime.
* **Arms:**
  1. `uncontrolled`: No stimulation baseline.
  2. `threshold`: Static heuristic stimulation triggered on regional amplitude.
  3. `obs_mlp_champion`: Top Observable MLP model.
  4. `obs_cnn_champion`: Top Observable CNN model.
  5. `obs_dmd_champion`: Top Hankel-DMDc model.
  6. `waveform_mlp_champion`: Top Waveform MLP model.
  7. `waveform_cnn_champion`: Top Waveform CNN model.
