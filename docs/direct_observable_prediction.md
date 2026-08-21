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
- Deployed geometry: [`configs/simulation/mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml)
- Spectral objectives: [`docs/spectral_objectives.md`](spectral_objectives.md)
- Domain vocabulary: [`CONTEXT.md`](../CONTEXT.md)

Numbers quoted below are measured against the deployed configuration, not estimated:
$C = 62$ channels, $m = 3$ electrodes, $n_y = 15$, $n_u = 10$, hidden $128 \times 2$,
$f_s = 50\text{ Hz}$, horizon $H = 75$, envelope $L = 50$, $R = 25$, so $M = 2$ frames of
$F = 26$ bins with the DC bin unscored.

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

1. **Envelope / band-power feedback in neurostimulation and DBS.**
   Closed-loop DBS work (*Little et al. 2013*, *Tinkhauser et al. 2017*, *Fleming et al. 2020*)
   tracks beta-band power envelopes rather than raw LFP waveforms. The precedent is real but
   narrower than it first looks: those loops *threshold* or *PI-regulate* a band-power estimate,
   they do not *forecast* it over a horizon. *Santaniello et al. 2011* is the closest to a
   predictive scheme and it is still recursive. So the literature supports band power as the right
   **feedback signal**; it does not establish that band power is predictable **several frames ahead
   as a function of a candidate control sequence**, which is the claim this document depends on.

2. **Direct multi-step (DMS) vs. recursive forecasting.**
   The usual citation, *Marcellino, Stock & Watson 2006*, concludes the opposite of what a DMS
   proposal wants: iterated forecasts generally **beat** direct ones, and the iterated advantage
   *grows* with horizon. Direct wins in one specific regime — when the one-step model is
   misspecified, so recursion compounds a systematic bias rather than just noise. That regime
   plausibly applies here (a 148k-parameter MLP standing in for a delayed Jansen-Rit network on a
   connectome is definitely misspecified), but it has to be argued and measured, not assumed.
   *Chevillon 2007* surveys the same trade-off; it is not an endorsement of DMS.

3. **Observable-space MPC and Koopman control.**
   *Korda & Mezić 2018* and *Peitz et al.* lift the state into observables — but the lifted dynamics
   are then **stepped recursively**, typically linearly: $z_{k+1} = A z_k + B u_k$. Koopman MPC is
   therefore an argument for moving the *state* into observable space, not for replacing recursion
   with a horizon-wide feedforward map. It is a distinct third option, and arguably the better one
   here; see [section 7](#7-alternative-recursion-in-observable-space).

---

## 3. Problems

### 3.1 Input and output dimension

At the deployed geometry the direct map is
$g : \mathbb{R}^{960} \times \mathbb{R}^{225} \to \mathbb{R}^{3100}$:

| Quantity | Size | Composition |
| :--- | ---: | :--- |
| History state $x_0$ | 960 | $n_y C + n_u m = 15 \cdot 62 + 10 \cdot 3$ |
| Control block $\mathbf{u}$ | 225 | $H m = 75 \cdot 3$ |
| Observable target $\hat{\mathbf{z}}$ | 3100 | $M C (F - 1) = 2 \cdot 62 \cdot 25$ |

The output is the binding constraint. A 3100-wide head off a 128-unit hidden layer is
$128 \times 3100 \approx 397\text{k}$ parameters in the last layer alone — roughly **2.7x the entire
current predictor** (~148k). Total direct model: ~565k parameters.

The data budget does not scale with it. The training set is 200 trajectories of 8 s
([`data/experiment_excited_roast/train`](../data/experiment_excited_roast/train)). Sliding windows
give ~62k rows, but the target spans 1.5 s, so each trajectory holds only ~5 **non-overlapping**
targets: on the order of **1000 independent $(x_0, \mathbf{u}, \hat{\mathbf{z}})$ triples** against
565k parameters. The autoregressive model reuses one 148k-parameter map at all 75 steps and sees
~62k single-step examples — weight sharing across the horizon is doing real statistical work, and
the direct map throws it away. This is the same mechanism behind the iterated-beats-direct result
in section 2.2.

Mitigations worth costing before committing: predict a reduced observable (pooled bins, or `eeg_ms`
scalar power per channel, $M C = 124$ outputs instead of 3100); factor the head as
$\text{hidden} \to r \to M C (F{-}1)$ with $r \ll 128$; or share one per-frame head across frames.

### 3.2 Non-Markovianity of the observable space

- A history of raw EEG $[y_{t-n_y}, \dots, y_t]$ carries phase, instantaneous velocity, and higher derivatives.
- A history of scalar power $[z_{t-n_z}, \dots, z_t]$ alone discards phase, which can make the observable state non-Markovian unless:
  - the input history state still ingests the raw EEG window $[y_{t-n_y}, \dots, y_t]$ during priming/absorption, but outputs future observables $\hat{z}_{1:M}$, or
  - the lookback buffer of observables is sufficiently wide to capture slow envelope dynamics.

For the direct map this is the mild case: $x_0$ stays raw, so nothing is discarded on the input
side. It becomes the hard case only for the recursive variant in section 7, where $z$ must serve as
the state.

### 3.3 Horizon-wide causal structure

In an autoregressive model, causality is guaranteed by step-by-step unrolling. In a direct horizon
network $\hat{z}_{1:M} = g(x_0, [u_0, \dots, u_{H-1}])$ nothing structurally prevents frame $m$ from
depending on controls that land after it, so the architecture has to enforce the mask (causal
convolutions, or a block-lower-triangular head).

At the deployed geometry this matters less than it sounds. With $L = 50$, $R = 25$, $H = 75$ there
are $M = 2$ frames: frame 0 spans steps 0–49 and frame 1 spans 25–74, so between them they already
depend on nearly the whole control block. The masking constraint bites only for short segments and
many frames. Do not spend architecture complexity on it at $M = 2$; reconsider if the frame count
grows.

### 3.4 The horizon becomes a weight, not a config knob

The autoregressive model is trained at one horizon and deployed at another. That is not
hypothetical: [`mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml) runs
`horizon: 75` against `artifacts/nonlinear_mse02_psd`, whose `native_horizon` is **50**. The MPC
currently extrapolates the predictor 50% past its trained rollout depth, and `MPCController.horizon`
exists precisely to allow that.

A direct map freezes $H$, $L$, $R$ and $M$ into the output layer. Retuning the horizon, the envelope
geometry, or the frame hop then means retraining — and the envelope geometry is the single source of
truth for the cost (`PsdEnvelope` carries $L$ and $R$; `neuro.validation` already raises when the
YAML disagrees with the npz). Every sweep over `psd_window_s` / `psd_hop_s` / `horizon` becomes a
training sweep. This is the largest practical cost of the proposal.

### 3.5 Predict log-power, not power

Section 4.1's cost as originally written, $\log(\hat{P} + \varepsilon)$, is unsafe: a network with an
affine output layer emits negative values, and $\log$ of a negative argument hands IPOPT a NaN that
no line search recovers from. `LOG_FLOOR = 1e-8` protects a *measured* power, which is already
non-negative; it does not protect a *predicted* one.

The fix is free and improves alignment: have the network output
$\hat{\ell}_{m,c,f} = \log \hat{P}_{m,c,f}$ directly. Then

$$J_{\text{PSD}} = \frac{1}{M C F'} \sum_{m,c,f} \left[\max\left(0,\; \hat{\ell}_{m,c,f} - \log P^{\text{ref}}_{c,f}\right)\right]^2$$

with no floor, no positivity constraint, and no exponential in the graph. `StftLoss` already scores
in log space (`log_spectrogram`), so training target and network output live in the same space —
which is the "zero representation mismatch" the proposal is selling, and it only actually holds in
the log parameterisation.

### 3.6 The control-attributable signal is small

This is the load-bearing risk, and it is measurable today without building anything.

Stimulation does not enter as an additive overlay on the sensors — `EEGMeasurement` has no direct
feedthrough. Electrode currents project through $\boldsymbol{\gamma}$ into somatic drives inside the
pyramidal firing-rate sigmoid, so suppression works by shifting the regional operating point across
the bifurcation rather than by cancelling cycles. That predicts a smooth, monotone dose-response in
envelope power, which is the mechanism that would make $g$ well-conditioned.

Measured on the training set, that dose-response is present and monotone. Binning horizon windows by
mean TP9 current and taking the change in mean log-power from the trailing frame to the horizon
frames:

| Mean TP9 current over horizon (mA) | n | $\Delta$ mean log-power |
| :--- | ---: | ---: |
| $[-2.37, -0.51)$ | 3680 | $+0.263$ |
| $[-0.51, -0.15)$ | 3680 | $+0.265$ |
| $[-0.15, +0.13)$ | 3680 | $+0.219$ |
| $[+0.13, +0.47)$ | 3680 | $+0.175$ |
| $[+0.47, +2.39)$ | 3679 | $+0.058$ |

Anodal TP9 drive monotonically slows power growth, consistent with the threshold controller's
$+2\text{ mA}$ suppressive polarity. But the effect is small against ambient variance:
$\text{corr}(\bar{u}_{\text{TP9}}, \log P) = -0.17$, and the full $\pm 2.4\text{ mA}$ range moves
log-power by ~0.28 nats against a std of 0.53 — about **half a standard deviation across the entire
admissible current range**.

A ridge baseline makes the consequence concrete. Predicting the 3100-dim log-power target from $x_0$
alone versus from $(x_0, \mathbf{u})$, split by trajectory so no window leaks across:

| Predictor | held-out $R^2$ |
| :--- | ---: |
| $x_0$ only (control-blind) | 0.131 |
| $\mathbf{u}$ only | 0.006 |
| $x_0$ and $\mathbf{u}$ | 0.132 |

The 225-dim control block adds **0.000**. A linear direct map cannot find the effect at all: the
0.03 of variance it carries is smaller than what 225 extra regressors cost in overfit. This does not
prove a nonlinear net will fail — the effect is real and monotone, and the target here is *absolute*
log-power, dominated by where in the seizure the window happens to sit. It does mean the proposal's
whole value rests on extracting a ~0.3–0.5$\sigma$ effect from 225 inputs with ~1000 independent
samples, and that this must be gated on explicitly rather than assumed.

The control block itself is not the problem: over 75-step windows it has a participation ratio of
~25 and needs 116 principal components for 95% of its variance, so the RAS excitation does explore
the space. The problem is the signal-to-variance ratio of the target.

---

## 4. Application to This Codebase

### 4.1 Where the speed-up actually comes from

This document previously headlined "elimination of the CasADi DFT graph". Measured, that is the
small part. Timing the objective and its Jacobian at $H = 75$, single shooting, MX (no `expand`):

| Graph | $f$ | $\nabla f$ |
| :--- | ---: | ---: |
| Rollout + stagewise $y$ cost (`w_psd = 0`) | 16.5 ms | 67.7 ms |
| Rollout + DFT hinge (`w_psd > 0`) | 19.5 ms | 71.0 ms |
| Direct observable net (same width, 3100 outputs) | 2.1 ms | 6.6 ms |

The symbolic DFT costs **3.3 ms of a 71 ms Jacobian — under 5%**. The 75-step unrolled MLP is the
other 95%. Removing the DFT alone buys nothing worth a rewrite; removing the *rollout* buys roughly
**10x on the Jacobian**, which is the real claim and should be stated as such.

Two caveats on the 10x. It is a single objective-plus-Jacobian evaluation, not a solve: SQP iteration
counts and QP conditioning may differ, and the direct model's dense $225 \times 3100$ Jacobian could
shift cost into the QP subproblem. And the benchmarked net is an assumed architecture, not a trained
one — if section 3.1 forces a wider head or more depth to fit, the margin shrinks.

For reference, the current CasADi cost is

```python
# Slices segments from predicted raw EEG and multiplies by DFT cosine/sine matrices
power = ((y_tapered @ ca.MX(dft_cos)) ** 2 + (y_tapered @ ca.MX(dft_sin)) ** 2) * scale_row
hinge = ca.fmax(0.0, ca.log(power[:, 1:] + LOG_FLOOR) - log_ref)
```

which the direct model replaces with an elementwise hinge on $\hat{\ell}$ (section 3.5).

### 4.2 What actually has to change at the interface

"The controller interface remains clean" was too optimistic. `absorb` / `is_ready` / `initial_state`
carry over unchanged, but the rest of `SymbolicModel`
([`src/neuro/types.py`](../src/neuro/types.py)) is built around stepping:

- `f_step`, `f_out`, `step` and `output` have no meaning for a direct map. Either the protocol grows
  a `predict_observable(x0, u_seq)` member with the stepping members made optional, or direct models
  get a sibling protocol.
- `MPCNlp.build` is organised around `_rollout_cost` producing `y_nodes` per step. With no per-step
  outputs there is no stagewise `w_y * sumsqr(y)` term and no defect constraints, so
  `shooting_depth`, `phi_vars` and `get_phi` all go dead on this path. `w_u`, `w_u_l1` and
  `_sum_to_zero` are functions of $\mathbf{u}$ alone and survive untouched.
- `MPCController._solve` builds its warm start by rolling `self.model.f_step` forward to seed
  `phi_guess`. With no shooting roots the seed collapses to `u_guess` (plus L1 slacks) — simpler,
  but the code path has to be branched or removed.
- `_spectral_hinge_cost` validates that the envelope's channel count matches the model's. A direct
  model needs a stricter check: envelope $(L, R, f_s, C, F)$ must match what the network was
  *trained* to emit, not just what the config declares. Given section 3.4, this check is the one
  thing standing between a geometry edit and a silently wrong cost.

---

## 5. Comparison

Both columns assume the deployed setting: single shooting (`shooting_depth >= horizon`), which is
what every config in the repo uses and the only setting that solves reliably.

| Dimension | Autoregressive Raw EEG Model | Direct Observable Model |
| :--- | :--- | :--- |
| **Solver graph** | $H$ sequential MLP evaluations + symbolic DFT (95% / 5%) | Single feedforward evaluation + elementwise hinge |
| **Jacobian cost** | 71 ms measured at $H = 75$ | ~7 ms measured, same width |
| **NLP decision variables** | $\mathbf{u} \in \mathbb{R}^{H \times m}$; states condensed out, shooting roots only if $D < H$ | $\mathbf{u} \in \mathbb{R}^{H \times m}$ — identical |
| **Constraints** | $-u_{\max} \le u \le u_{\max}$, $\sum u = 0$, plus L1 slacks | Identical |
| **Objective alignment** | Predictor minimises EEG MSE + auxiliary STFT loss; MPC scores a spectral hinge | Predictor forecasts exactly what the MPC penalises — but only in the log parameterisation |
| **Horizon / geometry** | Config knob; currently deployed at $H = 75$ on a horizon-50 artifact | Frozen into the weights; a geometry sweep becomes a training sweep |
| **Parameters vs. independent samples** | ~148k shared across 75 steps, ~62k single-step examples | ~565k, ~1000 independent horizon targets |
| **Failure mode** | Rollout drift; compounding one-step bias | Control-blindness: $g$ ignores $\mathbf{u}$, leaving the solver no gradient to descend |

The decision-variable and constraint rows are unchanged between the two — the earlier version of this
table credited the direct model with eliminating shooting roots that single shooting had already
eliminated.

---

## 6. Verification roadmap

Ordered so the cheapest falsification runs first. Each stage has a kill criterion; if it trips, stop
rather than proceed.

1. **Control-attributable signal (done, section 3.6).**
   Linear ridge, split by trajectory. Result: control adds 0.000 held-out $R^2$ on absolute
   log-power, though a monotone dose-response of ~0.5$\sigma$ exists. Read as: the effect is real but
   a linear direct map cannot recover it, so the nonlinear probe is not obviously redundant.

2. **Control-blind baseline — the gate.**
   Train two nets with identical architecture and schedule: $g_{\text{full}}(x_0, \mathbf{u})$ and
   $g_{\text{blind}}(x_0, \mathbf{0})$.
   **Kill criterion:** if $g_{\text{full}}$ does not beat $g_{\text{blind}}$ on held-out trajectories
   by more than the seed-to-seed spread, the direct map is control-blind and the MPC built on it will
   output whatever the regulariser prefers. No amount of solver speed fixes this. This comparison,
   not "is test loss low", is the experiment that matters — a direct model can score well on
   $\hat{P}$ while being useless for control.

3. **Predictability probe.**
   $[y_{\text{past}} \in \mathbb{R}^{15 \times 62},\; u_{\text{past}} \in \mathbb{R}^{10 \times 3},\;
   u_{\text{future}} \in \mathbb{R}^{75 \times 3}] \longrightarrow \hat{\ell}_{\text{future}} \in
   \mathbb{R}^{2 \times 62 \times 25}$, scored against `StftLoss`'s log-spectrogram of the true
   future. Compare against the *existing* autoregressive artifact pushed through the same STFT — the
   honest baseline is the incumbent, not the training mean.

4. **Gradient sanity, $\nabla_u \hat{\ell}$.**
   Sweep a constant TP9 current from $-2$ to $+2\text{ mA}$ and check the predicted log-power
   decreases monotonically, reproducing the table in section 3.6. Check that
   $\|\nabla_u \hat{\ell}\|$ is not orders of magnitude below $\|\nabla_{x_0} \hat{\ell}\|$, since the
   solver only ever sees the former.

5. **Closed-loop benchmark.**
   Seizure burden against `mse02_psd_mpc_spectral.yaml` and `threshold_control.yaml` over the same
   seed set. Seizure state is scored on source-space LFP, which no observable model predicts, so
   offline observable accuracy cannot substitute for this run.

A data gap worth naming before stage 2: the training set has no **paired** trajectories sharing a
plant seed and differing only in $\mathbf{u}$. Such pairs would isolate $\partial P / \partial u$
directly, turning stage 1 from a variance-decomposition argument into a direct measurement. Whether
generating them is cheaper than the modelling effort they de-risk is an open call.

---

## 7. Alternative: recursion in observable space

Sections 3.1, 3.4 and 3.6 all trace back to the same choice — replacing recursion with a
horizon-wide map — and there is a middle option that keeps most of the speed-up without it:

$$z_{m+1} = A z_m + B(z_m)\, \bar{u}_m, \qquad \hat{\ell}_m = C z_m$$

with $z$ a small lifted observable state and $\bar{u}_m$ the control averaged over frame $m$. This is
what Koopman MPC (section 2.3) actually does.

- The rollout is over $M = 2$–3 frames in a state of dimension ~$10^1$, not 75 steps in dimension
  960, so the 95% of the Jacobian cost identified in section 4.1 still goes away.
- $H$, $L$ and $R$ stay config knobs: more frames just means more recursion steps. Section 3.4
  dissolves.
- Weight sharing across frames returns, so the parameter count drops by an order of magnitude and
  section 3.1's data-budget problem shrinks with it.
- If $B$ is state-independent the per-frame problem is bilinear in $(z, u)$ and the MPC is close to a
  QP.

The cost is that section 3.2 becomes the hard case: $z$ must carry enough phase and velocity
information to be genuinely Markovian, which is the standard Koopman lifting question and is not
free. Worth pricing against the direct map before committing, because it dominates the direct map on
three of the four objections raised here.

---

## 8. Summary of the critique

What survives scrutiny:

- Removing the 75-step unrolled rollout from the CasADi graph is worth ~10x on the Jacobian.
- Training and control objectives genuinely align, in log-power space.
- The suppression mechanism is excitability damping, not phase cancellation, so envelope power really
  does respond smoothly and monotonically to sustained current.

What does not:

- The DFT graph is under 5% of the cost; eliminating it was never the point.
- Single shooting already removed the shooting roots the comparison table credited to the proposal.
- The cited DMS literature concludes iterated beats direct, and Koopman is recursive.

What is unresolved and gates everything:

- Whether ~1000 independent horizon targets can teach a 565k-parameter map a 0.5$\sigma$ control
  effect that a linear probe cannot see at all (sections 3.1, 3.6, stage 2).
- Whether freezing the horizon and envelope geometry into the weights is acceptable, given the
  deployed config currently exploits the freedom to change both (section 3.4).
