# Hyperparameter sweep strategy

This document lists the hyperparameter sweep experiments for surrogate Predictor models and closed-loop Model Predictive Control (MPC).

Training and offline evaluation horizons are at most $1.0\,\text{s}$. The MPC Control Horizon is fixed at $1.0\,\text{s}$ through experiment 2.3; experiment 2.4 may test longer lookahead. The Waveform curriculum MSE span is at most $0.25\,\text{s}$; a spectral loss may score the remainder of the $1.0\,\text{s}$ training rollout. The $700*$ Plant seeds are reserved for closed-loop evaluation and are distinct from training-data generation seeds. Historical champion configs are references, not baselines for this new sweep.

Since training happens mostly on GPU and closed-loop runs mostly on CPU experiments can be pipelined. Once a predictors Tier 1 experiment is done it's Tier 2 experiments can already start while other predictors are training. This has to be carefully managed as to not exhaust the computers resources.

## Baseline and calibration before the sweeps

Freeze one explicit baseline per Predictor family before varying its hyperparameters. These are starting values for the new code and data, not claims that the historical champions remain optimal:

| Family | Initial baseline |
| --- | --- |
| Observable MLP | $f_s=50\,\text{Hz}$; $n_\text{seg}=25$, $n_\text{hop}=3$, $1$--$25\,\text{Hz}$, linear kernel $K=1$, asymmetric Hann; residual, $n_y=1$, $n_u=9$, depth 1, width 256, tanh. |
| Observable CNN | Same Observable geometry; direct output, $n_y=9$, $n_u=9$, depth 2, 32 initial filters tapered to 16, $3\times3$ kernels, tanh; `tapering_ratio: 0.5`, `head_depth: 1`, `head_width: trunk`; valid temporal convolutions and all-position flattening; batch size 512, learning rate $2\times10^{-4}$. |
| Observable DMDc | Same Observable geometry; $n_y=1$, $n_u=9$, energy cutoff $0.99$, ridge parameter $10^{-6}$. |
| Waveform MLP | $f_s=100\,\text{Hz}$; residual, $n_y=15$, $n_u=10$, depth 1, width 256, softplus; curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$ with 50-sample segments, 25-sample hops and $1$--$25\,\text{Hz}$ band. |
| Waveform CNN | $f_s=100\,\text{Hz}$; direct output, $n_y=100$, $n_u=10$, depth 2, 32 initial filters tapered to 16, temporal kernel 5, tanh; valid convolutions and all-position flattening, `tapering_ratio: 0.5`, `head_depth: 1`, `head_width: trunk`; batch size 512, learning rate $2\times10^{-4}$; curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$ with 50-sample segments, 25-sample hops and $1$--$25\,\text{Hz}$ band. |

The $100\,\text{Hz}$ Waveform rate is provisional until the rate decision; it represents $0.25\,\text{s}$ exactly. For neural baselines use AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$, batch size 256, 150 epochs and patience 35; curriculum expansion begins at epoch 10 and completes by epoch 80. For both CNN baselines, resource pilots, and architecture sweeps, use batch size 512 and learning rate $2\times10^{-4}$: double the learning rate alongside the increase from batch size 256 to 512. This is a starting scaling choice to validate in the pilots; keep these settings fixed across architecture candidates. Use a fixed training/validation split and preprocessing within each family. Record the resolved baselines in local experiment configs before starting a sweep. For every changed Observable geometry, recompute the minimum valid control history with `observable.min_past_controls()`.

### Standardization and Loss units

All new neural Predictor runs standardize EEG or Observable histories, Control Current histories and future controls, and output targets using statistics fitted only on the training split. Mean/std scaling is fixed in code: each channel, or channel-frequency coordinate for an Observable, has its own center and scale. The `scaler` and `global_scaling` configuration options have been removed. The MLP/CNN predicts and recurses in standardized coordinates.

The dataset audit supports retaining mean/std scaling rather than switching to median/IQR: some Observable coordinates have IQR/std ratios near 0.12. IQR scaling would expand their observed extremes to roughly 70–77 scaled units, compared with 12–13 under mean/std scaling. This is a conditioning-based choice, not a measured training-performance advantage; robust statistics do not necessarily produce smaller neural-network inputs.

Every Loss is evaluated in original output units. Curriculum MSE multiplies standardized residuals by the fitted output scale before squaring; the center cancels. For Observable Predictors this recovers log-power, not EEG voltage. Waveform STFT and EEG mean-square-power Losses recover raw EEG before computing power. Loss units are mandatory; the former `unstandardized` configuration option has been removed.

The [dataset scaling audit](../artifacts/output_scaling_audit/report.md) records input/output ranges, coordinate distributions, training/validation differences, and the inverse-variance weighting that standardized MSE would impose. Dataset statistics establish objective weighting, not which objective achieves better closed-loop control.

Freeze this Loss-unit convention before starting fresh sweeps. Historical trials and checkpoints retain their recorded objective; do not mix their scores with the new runs or resume their studies. Offline `val_log_mse` already uses recovered log-power, and Waveform `rollout_nmse` is an evaluation metric independent of the training MSE unit convention.

The [extended shared uncertainty calibration](../artifacts/shared_uncertainty_10000/report.md) defines one frozen training objective for both Waveform families. Paired MLP/CNN pilots contribute equally to one shared pair of learned Loss scales, using matched calibration learning rates and batch sizes. The common ratio is pooled across three initialization seeds and frozen before architecture comparisons; shared-family pooling and freezing adapt the paper's method.

1. **Training Loss:** Fix $w_\text{mse}=1$ and use the same $w_\text{stft}=r_\text{shared}=15$ for both Waveform MLP and CNN. Three paired CPU pilots ran 10,000 adaptive updates after 1,024 MSE-only warm-up updates. Their final 1,000-update geometric medians were 15.560, 15.577 and 15.563; the pooled estimate was 15.564. Use 1:15 as the rounded common objective in new resolved experiment configs. The earlier 1:20 choice rounded the short-run estimate of 16.97 to one significant digit and is superseded; its [baseline snapshots](../artifacts/shared_uncertainty_calibration/report.md) remain historical. Do not learn or recalibrate Loss weights separately for architecture candidates. Calibration used the cached 20-recording subset of the 20 ms excitation dataset; it does not establish the best ratio on the full datasets or demonstrate that fixed 1:15 outperforms fixed 1:20. It does not replace full-curriculum baseline qualification, recursive stability checks or closed-loop evaluation.
2. **MPC Cost:** Retain $w_y=1$ and $w_\text{hinge}=1000$ for a combined Cost from the [historical output-gradient calibration](../artifacts/gradient_weight_calibration/report.md). That probe measured tracking-to-hinge output gradients using a 100 Hz healthy reference and the same future EEG. This measures Cost scaling with respect to predicted EEG, not the gradient through a trained Predictor to Control Current. Calibrate the overall state-cost-to-effort scale separately in the closed-loop effort experiment; matching the two output gradients does not establish the desired Delivered Charge.

The former family-specific gradient calibration (MLP 1:1, CNN 1:4) is retained as historical evidence; it does not define the shared architecture-comparison objective. The still earlier $w_\text{stft}=0.01$ used global robust scaling and standardized MSE under the older Predictor/Loss implementation. Preserve those historical configs and studies. The new common training ratio does not change MPC Cost calibration. Judge its eventual value by closed-loop Seizure Burden, Delivered Charge and solver behavior; uncertainty calibration is not a proof of optimal control performance.

Existing trained Predictors retain their saved weights and standardizer buffers: changing this training ratio does not alter their inference or MPC Cost. Preserve completed experiments, resolved configs, checkpoints, Loss curves and study values under their original objectives. Evaluate historical checkpoints with common unweighted metrics when comparing predictions; do not compare or relabel weighted aggregate Losses across different weights, units or reductions. Train fresh baselines and candidates under the new convention in new experiment folders and studies. Any fine-tuning of an old checkpoint with 1:15 is a separate experiment with a changed objective, not a continuation of the old comparison. For checkpoints predating the CNN activation changes, reproduce historical results with their archived source revision; changing only the Loss weights does not repair architecture differences.

For the new Observable MLP baseline, use $w_\text{hinge}=10$, $w_u=1$, and $w_{u,l1}=0$. This is the tested operating point from `observable_dataset_sensitivity_comparison`; use it as a starting point for the newly trained Predictor, not as a claim that the optimum transfers to changed Observable geometry or training data. The Waveform output-gradient calibration above does not apply to an Observable MLP.

For the new Waveform MLP baseline, use $w_u=10$ and $w_{u,l1}=0$. This retains the historical Waveform effort scale as a starting value. Calibrate Waveform effort separately after training its new baseline checkpoint; the Observable MLP's $w_u=1$ does not transfer across the different state Costs and Predictor sensitivities. Waveform CNN adopts the same initial effort baseline ($w_u=10, w_{u,l1}=0$) under the continuous waveform controller.

### Rerun priorities after the training-convention change

Rerun experiments whose Predictor fitting or selected checkpoint depends on the changed Loss units, standardization or joint Loss ratio. Preserve the original runs as historical evidence. Use the following order; broad sweeps wait for baseline qualification.

| Priority | Experiments | Required work |
| --- | --- | --- |
| 1 | Waveform MLP/CNN baseline qualification | Train fresh baselines through the full curriculum with original-unit Losses, per-channel mean/std scaling and MSE:STFT = 1:15. Use matched training seeds, data splits and evaluation horizons across comparisons. Check unweighted forecast metrics and recursive stability; the short uncertainty pilots do not qualify full training. |
| 1 | Observable MLP/CNN baseline qualification | Retrain under original-unit MSE and per-coordinate mean/std scaling. The 1:15 ratio applies only to Waveform joint Losses. Use the updated CNN architecture; historical checkpoints do not qualify the new baseline. |
| 2 | Controller effort calibration: Section 2.1 and the Section 2.4 pre-experiment | Recheck effort weights on each newly qualified Predictor before using it in further closed-loop comparisons. An unchanged MPC Cost can produce different Control Currents with a retrained Predictor. |
| 3 | Waveform Loss-Cost alignment: Section 2.4A | Retrain the pure-MSE, pure-STFT and joint-Loss arms under the new conventions, then repeat the five controller comparisons with the prescribed common effort weight and canonical Plant seeds. The historical joint-Loss advantage must be rechecked under 1:15. |
| After qualification | Capacity and optimizer studies: Sections 1.1-1.4 | Use fresh studies when selecting champions for the new objective. Rerun Observable and Waveform MLP optimization; complete the already-required CNN architecture reruns. Do not carry old trial scores or champions into the new rankings. |
| After qualified Predictor selection and effort calibration | Observable geometry: Sections 2.2-2.3 | Retrain each compared geometry under the new convention and repeat its closed-loop evaluation before adopting its historical ranking. The CNN geometry rerun also depends on the updated-architecture study. |
| After the relevant earlier selections | Terminal weighting, Control Horizon and final champion comparison: Sections 2.4B, 2.5 and 2.6 | Repeat these comparisons on the newly selected Predictors and controller settings before reporting them as conclusions for the new pipeline. |

The existing excitation datasets and scaling audit do not need regeneration solely because of this convention change. The 1:15 ratio does not apply to DMDc; audit Section 1.5's fitting and preprocessing provenance, and rerun affected fits if their scaling or other fitting settings changed. Re-evaluating an old checkpoint under common metrics is useful as a historical reference, but does not replace retraining for a new-objective comparison.

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

### 1.2 Observable CNN re_rerun

* **Status:** Prepared in [re_rerun](../artifacts/re_rerun/report.md). All candidates use the new full-history readout and train from scratch; archived scores and checkpoints are historical evidence.
* **Purpose:** Evaluate time-frequency feature extraction, history length, tapering and calculated head capacity under fixed Observable geometry and direct prediction (`residual: false`).
* **Target metric:** One-second recursive Rollout `val_log_mse`.
* **Fixed configuration:** Baseline Observable geometry, $n_u=9$, temporal `kernel_size: 3`, AdamW learning rate $2\times10^{-4}$, weight decay $10^{-5}$, batch size 512, 150 epochs, patience 35, and the baseline split and curriculum schedule.
* **Architecture ranges:**
  * `model.n_y`: Integer $[5,17]$ Frames. Including each 0.5-second Segment and the 60 ms hop, the underlying EEG support is approximately 0.74–1.46 seconds.
  * `model.hidden_size`: Integer $[8,32]$ initial filters.
  * `model.depth`: Integer $[1,2]$ convolutions.
  * `model.frequency_kernel_size`: Categorical `[1,3,5]`.
  * `model.activation`: Categorical `[tanh, softplus]`.
  * `model.tapering_ratio`: Uniform float $[0.5,1.0]$, active only for depth greater than one.
  * `model.head_depth`: Integer $[0,1]$ hidden dense layers.
  * `model.head_width`: Categorical `[trunk, geometric_mean]`, active only for a hidden head.
* **Baseline:** History 9 Frames, two convolutions, filters $32\to16$, temporal/frequency kernels 3, tanh, one hidden head with `trunk` width, direct output.
* **Optimizer tuning:** After structural selection, use a separate fresh study for learning rate log-uniform $[10^{-4},3\times10^{-3}]$ and weight decay log-uniform $[10^{-6},10^{-3}]$.

#### CNN re_rerun protocol shared with Section 1.4

* **Sampler and budget:** 25 trials per family in fresh study directories, using `TPESampler(n_startup_trials=15, multivariate=True, seed=training.seed)`. Completed and pruned trials count toward startup; failed trials do not. Conditional dimensions may fall back to independent sampling under multivariate TPE.
* **Architecture:** Activate every convolution. Use valid temporal convolutions, unit stride, no dilation or temporal pooling, and same-size frequency padding for Observable Frames. Flatten all final positions in time/filter/frequency order (time/filter for Waveform), concatenate Control Current history, apply the selected hidden head and a linear direct output. Both families fix `residual: false`.
* **History support:** Record $R_t=1+L(k-1)$ and $T=n_y-L(k-1)$. Each local feature sees $R_t$ positions, but the head combines all $T$ positions and can depend on all supplied history. All combinations in the prepared ranges satisfy $T\ge1$; do not deduplicate histories using $R_t$.
* **Tapering and heads:** $H_1=\texttt{hidden_size}$ and $H_{l+1}=\lfloor H_l r\rfloor$. Tapering shrinks later layers and the flattened features, but not the first convolution. `trunk` uses $H_L$; `geometric_mean` uses $\lfloor\sqrt{D_{in}D_{out}}\rfloor$, including the entire flattened feature map and Control Current history in $D_{in}$. The runner skips taper sampling at depth 1 (resolved ratio 1.0) and width-rule sampling at head depth 0 (resolved `trunk`). Other coincident sampled architectures are not deduplicated.
* **Linear-head ablation:** Head depth 0 retains a nonlinear convolutional trunk, but gives additive, fixed one-step Control Current sensitivity. A hidden head permits measurement-dependent control interactions. A linear head can have more parameters than a narrow hidden head because it maps the entire feature map directly to every output.
* **Resource pilots:** Before launch, run each family's baseline, maximum flattened geometric-mean head, linear head, and deepest tapered trunk. Measure peak memory and training time at full Loss Span on the 4 GB GPU. Freeze any batch-size/learning-rate change consistently within the family before the study.
* **Runtime pruning:** Prepared configs set `training.max_training_hours: 7.0`. After three completed epochs, project total fit time as elapsed time plus remaining configured epochs times the mean duration of the latest two epochs. Exclude the first epoch from the rate estimate, retain its cost in elapsed time, and recheck each later epoch. Over-budget fits stop immediately at that epoch boundary and Optuna records PRUNED with timing evidence. The forecast includes epoch training and validation, assumes the full epoch budget, and is not a hard per-epoch timeout.
* **Qualification:** Preserve schedule eligibility and one-second recursive validation. Apply the amplitude-growth and terminal-NMSE gates separately before selecting champions; the sweep runner does not automatically enforce those stability gates. Measure controller derivative and solve costs for finalists.
* **Storage and reproducibility:** Self-contained baseline, pilot and architecture configs are under `artifacts/re_rerun/{observable,waveform}/`. Use fresh databases and checkpoints, and record the source revision, uncommitted patch, dependencies, data manifests and resolved configs. Historical newest-position checkpoints are explicitly rejected and require retraining. Optimizer tuning follows qualified structural selection.

### 1.3 Waveform MLP optimization

* **Purpose:** Tune network capacity and optimizer parameters for high-rate raw EEG prediction under a fixed joint loss.
* **Target metric:** Recursive rollout NMSE over $1.0\,\text{s}$.
* **Fixed configuration:** Provisional sampling rate $f_s = 100\,\text{Hz}$, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, fixed weights $w_\text{mse}=1$ and $w_\text{stft}=r_\text{shared}$, residual connection enabled.
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

### 1.4 Waveform CNN re_rerun

* **Status:** Prepared in [re_rerun](../artifacts/re_rerun/report.md), superseding the earlier `waveform_cnn_full_history` preparation. Follow the shared protocol in Section 1.2.
* **Architecture:** Valid temporal convolutions, activation after each convolution, all-position flattening, calculated-width head and direct EEG output (`residual: false`).
* **Fixed configuration:** 100 Hz, $n_u=10$, curriculum MSE Span $0.25\,\text{s}$ and STFT Span $1.0\,\text{s}$ with shared output-unit weights $w_\text{mse}=1$, $w_\text{stft}=r_\text{shared}$, one-second recursive `rollout_nmse`, AdamW learning rate $2\times10^{-4}$, weight decay $10^{-5}$, batch size 512, 150 epochs, patience 35, and the baseline schedule and split.
* **Architecture ranges:**
  * `model.n_y`: Integer $[25,150]$ samples, approximately 0.25–1.5 seconds.
  * `model.hidden_size`: Integer $[8,32]$ initial filters.
  * `model.depth`: Integer $[1,3]$ convolutions.
  * `model.kernel_size`: Integer $[3,9]$, including even sizes.
  * `model.activation`: Categorical `[tanh, softplus]`.
  * `model.tapering_ratio`: Uniform float $[0.5,1.0]$, active only for depth greater than one.
  * `model.head_depth`: Integer $[0,1]$ hidden dense layers.
  * `model.head_width`: Categorical `[trunk, geometric_mean]`, active only for a hidden head.
* **Baseline:** History 100 samples, two kernel-5 convolutions, filters $32\to16$, tanh, one hidden head with `trunk` width, direct output.
* **Optimizer tuning:** After structural selection, use a separate fresh study for learning rate log-uniform $[10^{-4},2\times10^{-3}]$ and weight decay log-uniform $[10^{-6},10^{-3}]$.

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
* **Fixed configuration:** Baseline Waveform MLP architecture and sampling rate from the calibration section, curriculum MSE span $0.25\,\text{s}$, STFT span $1.0\,\text{s}$, and Waveform effort baseline $w_u=10$, $w_{u,l1}=0$. The joint Loss uses $w_\text{mse}=1$, $w_\text{stft}=r_\text{shared}$; the joint Cost uses $w_y=1$, $w_\text{hinge}=1000$.
* **Sub-experiments:**
  * **Pre-experiment (Waveform effort check):** With one newly trained baseline Waveform checkpoint and $w_{u,l1}=0$, compare $w_u=[3,10,30]$ under the fixed combined Cost. Use the selected effort weight unchanged across the loss-Cost arms below; report Seizure Burden and Delivered Charge together.
  * **Sub-experiment A (Loss-Cost alignment):**
    * Evaluates the fundamental correspondence between training objective and controller objective:
      * Arm 1 (Pure time-domain): Pure MSE Loss ($w_\text{mse} = 1.0, w_\text{stft} = 0.0$) paired with Raw Voltage Tracking Cost ($w_\text{hinge} = 0.0$).
      * Arm 2 (Pure spectral): Pure STFT Loss ($w_\text{mse} = 0.0, w_\text{stft}=r_\text{shared}$) paired with Spectral Frame Hinge Cost ($w_y = 0.0, w_\text{hinge}=1000$).
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
