# NN Predictor — Training Setup

A detailed, end-to-end description of how the neural-network EEG predictor is
trained. The predictor is a [PyTorch](https://pytorch.org/) MLP, optimised with `torch.optim.AdamW`,
that learns a **one-step-ahead** EEG map and is unrolled autoregressively to produce a
multi-step-ahead forecast conditioned on past EEG and past/future stimulation.

Source of truth:

- Training loop: [`src/neuro/predictor/train.py`](../src/neuro/predictor/train.py)
- Data loading & windowing: [`src/neuro/predictor/data.py`](../src/neuro/predictor/data.py)
- torch module: [`src/neuro/predictor/module.py`](../src/neuro/predictor/module.py)
- Losses: [`src/neuro/predictor/losses.py`](../src/neuro/predictor/losses.py)
- Artifact + inference: [`src/neuro/predictor/artifact.py`](../src/neuro/predictor/artifact.py)
- Shared metrics & artifact dispatch: [`src/neuro/artifacts.py`](../src/neuro/artifacts.py)
- Config schema: [`src/neuro/config.py`](../src/neuro/config.py)
- Entry points: [`scripts/run_nn_predictor.py`](../scripts/run_nn_predictor.py),
  [`scripts/sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py)
- Example configs: [`configs/nn_predictor/`](../configs/nn_predictor/)

> All computation runs in **float64**, requested explicitly: every `nn.Linear` is built with
> `dtype=torch.float64` and every array crossing into torch goes through
> `torch.as_tensor(..., dtype=torch.float64)`. The global default dtype is never changed, so
> importing `neuro` cannot alter dtype behaviour for anything else in the process.

---

## 1. Problem statement

Let

- $y_t \in \mathbb{R}^{C}$ — measured EEG at (downsampled) time step $t$, $C = n_\text{ch}$ channels
  (the Jansen–Rit sensor EEG has $C = 62$ channels; see the *uncalibrated units* note — EEG is in
  arbitrary units, not µV).
- $u_t \in \mathbb{R}^{m}$ — stimulation / control input at step $t$, $m = n_\text{controls}$ electrodes.

We want a direct **$N$-step-ahead** predictor. Given a history of $n_y$ past EEG samples, $n_u$ past
controls, and the $N$ future controls that *will* be applied, predict the next $N$ EEG samples:

$$
\hat{y}_{t+1:t+N} \;=\; F_\theta\!\big(\,\underbrace{y_{t-n_y+1:t}}_{\text{past EEG}},\;
\underbrace{u_{t-n_u:t-1}}_{\text{past control}},\;
\underbrace{u_{t:t+N-1}}_{\text{future control}}\,\big).
$$

$F_\theta$ is not a single monolithic network. It is the **autoregressive unrolling** of a shared
one-step model $f_\theta$:

$$
\hat{y}_{t+1} = f_\theta\big(y_{t-n_y+1:t},\; u_{t-n_u+1:t}\big),
$$

where $f_\theta$ is an MLP with weights $\theta$. Each rollout step shifts the newest predicted EEG
and the next scheduled control into the history windows and re-applies the *same* $f_\theta$
(Section 5). Training the rollout end-to-end therefore trains a single one-step model that is
consistent under its own feedback.

### 1.1 Indexing notation conventions

To avoid ambiguity between global time-series progression, dataset windowing, and algorithmic loops:

| index | range / domain | scope & meaning |
| ----- | -------------- | --------------- |
| $t$   | $0, \dots, T-1$ | Continuous time step index along a full trajectory recording |
| $k$   | $k_\text{start}, \dots, k_\text{end}-1$ | Anchor step index along a trajectory where a sliding window ends |
| $j$   | $0, \dots, N-1$ | Relative step index during sequential rollout unrolling (predicting $y_{k+j+1}$) |
| $i$   | $0, \dots, N-1$ | Relative horizon step index in algebraic loss and metric summations ($\text{NMSE}_i$) |
| $b$   | $0, \dots, B-1$ | Window / sample index within a batch or validation set |
| $c$   | $0, \dots, C-1$ | EEG channel index ($C = n_\text{ch} = 62$) |

---

## 2. Training data

### 2.1 Where it comes from

Each trajectory is an `.npz` file produced by the tES simulation experiments and stored in a
directory such as `data/experiment_excited/train/`. Files are discovered and sorted by
[`resolve_data_files`](../src/neuro/config.py); **every** `.npz` in the directory becomes a training
trajectory. Each file provides two arrays:

| key               | meaning                       | shape        |
| ----------------- | ----------------------------- | ------------ |
| `sensor_0.y_mea`  | measured EEG output           | $(T, C)$     |
| `controller.u`    | stimulation input             | $(T, m)$     |

The recordings are **persistently exciting** tES sequences (broadband stimulation), so the control
$u$ visits enough of the input space for the network to learn the input→output response rather than
just the autonomous dynamics. The stimulation obeys **Kirchhoff's current law** — each row of $u$
sums to zero across the $m$ electrodes (no net injected current), matching the constraint the MPC
controllers enforce. Data generated before this fix is physically invalid; if `data/experiment_excited/`
is empty, regenerate it with `uv run scripts/run_simulation.py configs/simulation/experiment_excited.yaml
--output-dir data/experiment_excited` and split the resulting `sim_*.npz` into `train/` (sim_000–021)
and `test/` (sim_022–024) before training (older wrong-stimulation datasets are archived under
`data/pre_kirchhoff_wrong_stim/`).

The datasets were regenerated on 2026-07-30 for the retuned plant (`K = 0.60`, `sigma = 280`,
`initial_state: rest` — see [seizure_spread_tuning.md](seizure_spread_tuning.md)). The trial layout
changed with it: **25 trials × 20 s** instead of 100 × 5 s, which is the same 500 s of simulated
data per dataset but long enough for each trial to cover the whole EZ → PZ → hemisphere
propagation rather than only its onset. Predictors fitted before that date were trained on a
different plant and must be refitted.

That intent was defeated in practice until 2026-08-01: every config still carried the old
`n_steps: 500`, which caps a trajectory at 5 s, so the extra 15 s of propagation was loaded and
then discarded. Every predictor fitted between 2026-07-30 and that date saw only each trial's
first 5 s — mostly pre-seizure. All configs now use `n_steps: null`
(see [tes_field_geometry.md](tes_field_geometry.md)).

The excitation was widened at the same time: `hold_ms` is now `[10, 50, 200]` ms rather than a
flat 10 ms, so the identification input covers the low-frequency band the MPC commands in and not
just the control rate. A noise-matched **zero-stimulation twin** of each dataset is generated
alongside it (`configs/simulation/experiment_baseline.yaml`), which makes the noise-free tES
response directly observable for diagnostics.

Which of these mattered was settled by ablation: the truncation is what decides whether the
controller can silence the focus at all, and the excitation bandwidth is what keeps the effect
lateralised. Both were needed — see [tes_field_geometry.md](tes_field_geometry.md).

### 2.2 Loading, filtering and downsampling

[`load_trajectory`](../src/neuro/predictor/data.py) reads at most `n_steps · downsample` raw samples
(or the entire trajectory if `n_steps` is omitted / `null`), low-passes the EEG, and then decimates by
taking every `downsample`-th sample:

$$
y^{\text{ds}}_k = \big(\mathcal{H}\,y^{\text{raw}}\big)_{k\cdot d}, \qquad
u^{\text{ds}}_k = u^{\text{raw}}_{k\cdot d}, \qquad k = 0,\dots,n_\text{steps}-1,
$$

with $d = \texttt{downsample}$. $\mathcal{H}$ is a **causal** Butterworth low-pass
([`lowpass_filter`](../src/neuro/filtering.py)), at `simulation.cutoff_hz` if that is set and
otherwise at the decimated Nyquist rate $1/(2 d \Delta t)$ ([`antialias_filter`](../src/neuro/filtering.py)).
It is causal rather than zero-phase because the same filter has to be reproducible sample-by-sample
online, inside the closed loop. The control is strided **unfiltered** — it is a commanded signal,
piecewise-constant at the control rate, so there is nothing to alias. The effective sample step is

$$
\Delta t_\text{real} = \Delta t \cdot d.
$$

Example (all shipped configs): $\Delta t = 10^{-4}\,\text{s}$, $d = 100 \Rightarrow \Delta t_\text{real}
= 10^{-2}\,\text{s}$ (100 Hz), and $n_\text{steps} = \texttt{null}$ ⇒ the whole **20 s per
trajectory** (2000 downsampled steps). This
$\Delta t_\text{real}$ is stored in the artifact and becomes the model's native step at inference.

---

## 3. Dataset construction (supervised windows)

[`build_dataset_for_trajectory`](../src/neuro/predictor/data.py) turns each trajectory into
input/target pairs with a **sliding window** (via `np.lib.stride_tricks.sliding_window_view`, so no
data is copied until the final gather).

For a valid center index $k$, define the anchor range

$$
k \in [\,k_\text{start},\, k_\text{end}\,), \qquad
k_\text{start} = \max(n_y - 1,\; n_u), \qquad
k_\text{end} = T - N,
$$

which guarantees the full history and full horizon exist. For each $k$ the feature and target are:

**Feature** $x_k \in \mathbb{R}^{\,n_y C + n_u m + N m}$, a concatenation of three flattened blocks

$$
x_k = \Big[\;
\underbrace{y_{k-n_y+1},\dots,y_{k}}_{n_y\ \text{steps (incl. }k)},\;\;
\underbrace{u_{k-n_u},\dots,u_{k-1}}_{n_u\ \text{past controls}},\;\;
\underbrace{u_{k},\dots,u_{k+N-1}}_{N\ \text{future controls}}
\;\Big].
$$

**Target** $Y_k \in \mathbb{R}^{\,N C}$, the next $N$ EEG samples

$$
Y_k = \big[\,y_{k+1},\, y_{k+2},\,\dots,\, y_{k+N}\,\big].
$$

Note the timing convention: predicting $y_{k+1}$ uses $u_{k}$ as the "current" control (the control
active over the step being predicted). This is exactly what the autoregressive rollout consumes at
each step.

Per trajectory this yields $k_\text{end} - k_\text{start}$ samples; trajectories are then concatenated
along the sample axis in [`prepare_datasets`](../src/neuro/predictor/data.py):

$$
X \in \mathbb{R}^{M \times (n_y C + n_u m + N m)}, \qquad
Y \in \mathbb{R}^{M \times N C}, \qquad
M = \sum_{\text{traj}} \big(T_\text{traj} - N - k_\text{start}\big).
$$

The control count is recovered from the feature width:

$$
m = \frac{\text{width}(X) - n_y C}{\,n_u + N\,}.
$$

`prepare_datasets` returns a [`Datasets`](../src/neuro/predictor/data.py) record holding the standardized
train and validation windows *and* the held-out trajectories whole (`val_trajs`), because free-run rollout
scoring (§8) and the comparison plot both need the un-windowed validation signals, not just their windows.

---

## 4. Preprocessing: split and scaling

### 4.1 Train / validation split

The split is applied **by trajectory file**, in name order and before any windowing
([`split_data_files`](../src/neuro/predictor/data.py)):

$$
n_\text{train} = \lfloor \texttt{train\_split} \cdot n_\text{files} \rfloor, \qquad
\text{train} = \text{files}_{0:n_\text{train}}, \qquad
\text{val} = \text{files}_{n_\text{train}:},
$$

clamped so each side keeps at least one of the $n_\text{files}$ files. Splitting whole trajectories rather than
the concatenated window index avoids leakage between the heavily-overlapping sliding windows, and it
leaves the validation trajectories intact so free-run rollouts can be scored on them (§8). The ESN
path splits identically — it calls the same helper — so the two model families are validated on the
same held-out files.

### 4.2 Transforms (Standardizers)

The EEG and control transforms are [`Standardizer`](../src/neuro/transforms.py)s, **fit on the continuous
training trajectories only** (so no validation statistics leak in) and reused for validation and at inference:

- **`y_std`**: [`Standardizer`](../src/neuro/transforms.py), fit on the full continuous training EEG channels.
- **`u_std`**: [`Standardizer`](../src/neuro/transforms.py), fit on the continuous training control signals.

The `Standardizer` supports:

- `scaler = "standard"` → subtract mean, divide by std.
- `scaler = "robust"` → subtract median, divide by IQR (robust to the large EEG bursts / bistable
  jumps). Used by all shipped configs.
- `global_scaling = true` → statistics pooled to **one** scalar shared across channels; `false` →
  **per-channel**. All shipped configs use global scaling.

Standardisation (per feature $j$): $\tilde{x} = (x - c_j)/s_j$, with $(c_j, s_j) = (\text{mean},
\text{std})$ for standard, $(\text{median}, \text{IQR})$ for robust.
In [`prepare_datasets`](../src/neuro/predictor/data.py), each continuous trajectory is transformed with
`y_std` and `u_std` before sliding windows are built. The model operates directly in standardized channel
space, and evaluation is done in raw EEG units via `y_std.inverse_transform` (Section 8).

The statistics are fitted on the **concatenated raw training trajectories**, so every sample counts
once. (Before the projection was removed they were fitted on the sliding windows instead, which
weighted each sample by roughly $n_y$ and dropped the trailing `horizon` samples; under
`scaler = "robust"` the median/IQR — and therefore every trained model — differ between the two.)

---

## 5. Model architecture

### 5.1 The one-step MLP $f_\theta$

An `nn.ModuleList` of `nn.Linear` layers inside
[`AutoregressiveMLP`](../src/neuro/predictor/module.py):

$$
f_\theta : \mathbb{R}^{\,n_y C + n_u m} \;\to\; \mathbb{R}^{C}.
$$

Widths are in **standardized channel space**: $C$ is the EEG channel count (§3.1), and `n_channels`
sizes every layer.

- **Input width** $= n_y C + n_u m$ — the one-step model sees only $n_y$ past EEG steps and $n_u$
  past controls. (The future controls in the feature vector are fed in one at a time by the rollout,
  not all at once.)
- **Output width** $= C$ — a single next-EEG vector in standardized channel space.
- **Layer sizes** $= [\,n_y C + n_u m,\ \underbrace{\texttt{hidden\_size}, \dots}_{\texttt{depth}},\ C\,]$.
  - `depth = 0` ⇒ a single affine layer $f_\theta(v) = Wv + b$ with **no** hidden layer and no
    activation — i.e. a **linear** predictor.
    The artifact reports this as `is_linear`, which is simply `len(layers) == 1`.
  - `depth = 1` ⇒ one hidden layer: $f_\theta(v) = W_2\,\sigma(W_1 v + b_1) + b_2$.
- **`activation`** $\sigma$ ∈ {`relu`, `tanh`, `softplus`}, applied after **every layer except the
  last**; the shipped configs use `softplus`, $\sigma(z) = \log(1 + e^{z})$. The literal type is
  enforced by the config schema, so a typo fails at config load rather than hours into a sweep at
  MPC construction time.

The layers are built with `dtype=torch.float64` and initialised by torch's default `nn.Linear`
initialiser under `torch.manual_seed(training.seed + seed_offset)`.

Everything the module sees is already **standardized** — the EEG block by `y_std`, the control blocks
by `u_std`, applied to whole trajectories before the windows are built (§3.1). The module itself
knows nothing about raw units.

### 5.2 Autoregressive rollout $F_\theta$

[`AutoregressiveMLP.forward`](../src/neuro/predictor/module.py) unrolls $f_\theta$ over the horizon
with a plain Python `for` loop (eager, no `scan`, no compile). Given a batch of feature vectors it
splits each into the history windows $Y^{(0)} = y_{k-n_y+1:k} \in \mathbb{R}^{n_y \times C}$,
$U^{(0)} = u_{k-n_u:k-1} \in \mathbb{R}^{n_u \times m}$, and the future-control sequence
$u_{k:k+N-1}$. For each rollout step $j = 0,\dots,N-1$ with incoming control $u_{k+j}$:

$$
\begin{aligned}
U^{(j+1)} &= \big[\,U^{(j)}_{1:},\; u_{k+j}\,\big] &&\text{(shift newest control in, drop oldest)}\\
\hat{y}_{k+j+1} &= f_\theta\big(\operatorname{vec}(Y^{(j)}),\; \operatorname{vec}(U^{(j+1)})\big) &&\text{(one-step prediction)}\\
Y^{(j+1)} &= \big[\,Y^{(j)}_{1:},\; \hat{y}_{k+j+1}\,\big] &&\text{(feed prediction back)}
\end{aligned}
$$

The rollout emits $\hat{Y} = [\hat{y}_{k+1},\dots,\hat{y}_{k+N}] \in \mathbb{R}^{NC}$. Crucially, the
**first** step ($j=0$) uses only ground-truth history — it is a pure one-step / teacher-forced
prediction — while later steps consume the model's own predictions, so errors can compound. This
distinction drives the curriculum loss below. Backpropagation runs through the whole unrolled loop,
so the BPTT depth equals the horizon $N$.

The batch dimension is native: `forward` takes `(B, n_y k + (n_u + N) m)` and returns `(B, N k)`.
There is no `vmap`.

### 5.3 The same rollout, three times over

The identical recursion exists in three places, and they are pinned to each other by test:

| where | used for | dtype/engine |
| ----- | -------- | ------------ |
| [`AutoregressiveMLP.forward`](../src/neuro/predictor/module.py) | training (differentiable) | torch, float64 |
| [`MLPArtifact.rollout`](../src/neuro/predictor/artifact.py) / `rollout_many` | evaluation, plotting, closed-loop simulation | NumPy, float64 |
| [`NNSymbolicModel`](../src/neuro/nn_predictor_casadi.py) `f_step` / `f_out` | the MPC's symbolic graph for IPOPT | CasADi |

`tests/test_predictor_module.py::test_torch_rollout_matches_casadi` pins torch against the CasADi
bridge to `1e-10` over `depth ∈ {0, 2}` and all three activations;
`test_prime_rollout_matches_the_training_window_at_the_same_index` pins `MLPArtifact.rollout` against
torch on the same $t_0$; `tests/test_batched_rollout.py` pins `rollout_many` against a loop over
`rollout` to `1e-12`.

Two convention traps are worth remembering. `NNSymbolicModel` standardizes raw $u$ *internally*,
while the torch module and `MLPArtifact.rollout`'s `state` expect the control already in model space.
And the three carry the recursion in **two state conventions**, which differ only in where the
control window sits:

| state built by | y-window ends | u-window ends | shift relative to the prediction |
| -------------- | ------------- | ------------- | -------------------------------- |
| `absorb` (MPC, fed the *previous* control) and the training feature row | $t$ | $t - 1$ | control shifts in **before** |
| `MLPArtifact.prime` | $t$ | $t$ | control shifts in **after** |

Both predict $\hat{y}_{t+1}$ from a $u$-window ending at $t$ — the rule of §1 — and the two are
interconvertible by lagging the u-window one step (see `_casadi_horizon_rollout` in
`tests/test_nn_predictor_casadi.py`, which does exactly that to compare them).

---

## 6. Loss function

Implemented in [`losses.py`](../src/neuro/predictor/losses.py). The training loss is a weighted sum
of independently configured loss terms:

$$
\boxed{\;\mathcal{L} = \sum_i w_i\,\mathcal{L}_i\;}
$$

The model rolls out directly in standardized channel space, producing $\hat{Y} \in \mathbb{R}^{B \times N \times C}$,
where the training rollout horizon $N = \max_i (\text{span\_steps}_i)$ is derived from the active losses.
The targets $Y \in \mathbb{R}^{B \times N \times C}$ are the standardized EEG channels (§4.2). Each loss term receives
$(\hat{Y}, Y, \text{ctx})$, where [`LossContext`](../src/neuro/predictor/losses.py) provides unit
recovery parameters ($y_\text{center}$, $y_\text{scale}$), sample rate $f_s$, and the current epoch
$e$ ($\text{epoch} = \text{null}$ during validation to evaluate terminal schedules).

### 6.1 Horizon-length curriculum MSE (`curriculum_mse`)

The MSE is scored over the first $L(e)$ rollout steps, where $L \le \text{span\_steps}$.
With per-step errors

$$
e_i = \frac{1}{BC}\sum_{b,c}\big(\hat{Y}_{b,i,c} - Y_{b,i,c}\big)^2,
$$

the curriculum MSE is their mean over that prefix:

$$
\boxed{\;\mathcal{L}_\text{MSE} = \frac{1}{L}\sum_{i<L} e_i\;}
$$

- $L = 1$ → pure one-step (teacher forcing): only the first step (ground-truth context, no feedback).
- $L = \text{span\_steps}$ → pure multi-step rollout: the objective we ultimately care about,
  exposing the model to its compounding errors.

$L$ is grown from 1 to $\text{span\_steps}$ over training epochs $[e_0, e_1]$ = `[curr_start, curr_end]`
(Section 7.3). Note that the full model horizon $N$ is always rolled out; the loss simply slices the
prefix $[:L]$ it scores, so the untrusted tail contributes no gradient.

### 6.2 Auxiliary spectral loss (`psd`)

Pushes the *statistics* of the rollout toward the data, not just point accuracy. Active when `psd` is
configured in `training.losses` and $e \ge \text{start\_epoch}$ (requires $\text{span\_steps} \ge 2$).

**PSD loss (log-spectral distance, Welch).** The PSD is a batch-of-snippets estimate over the loss's
span: per channel $c$, all $B$ windows are concatenated into one length-$B \cdot \text{span\_steps}$
series and Welch's method is applied with `nperseg = span_steps`, `noverlap = 0`:

$$
\mathcal{L}_\text{PSD} = \frac{1}{C\,F}\sum_{c,f}\Big(\log\big(\hat{P}_{c,f} + \varepsilon\big)
- \log\big(P_{c,f} + \varepsilon\big)\Big)^2, \qquad \varepsilon = 10^{-8}.
$$

[`welch_psd`](../src/neuro/predictor/losses.py) is a differentiable, bit-faithful replica of
`scipy.signal.welch(x, nperseg=span_steps, noverlap=0, axis=-1)` under that call's defaults —
`detrend="constant"`, a periodic Hann window, `scaling="density"`, one-sided folding, and mean
averaging over segments. It is pinned against SciPy to `1e-10` in
[`test_predictor_losses.py`](../tests/test_predictor_losses.py).

### 6.3 Metric-twin losses (`eeg_ms`, etc.)

Metric-twin losses align training objectives with evaluation metrics defined in
[`metrics.py`](../src/neuro/metrics.py).

- **EEG Mean Square (`eeg_ms`):** Evaluates the rolling power envelope in **raw EEG units**.
  The standardized tensors $\hat{Y}$ and $Y$ are converted to raw units via
  $\text{ctx.to\_raw}(\cdot) = (\cdot) \odot y_\text{scale} + y_\text{center}$, sliced to
  `[:span_steps]`, and unfolded into sliding windows of length $W = \operatorname{round}(\text{window\_s} \cdot f_s)$
  and hop $H = \operatorname{round}(\text{hop\_s} \cdot f_s)$. The mean square per window is computed,
  and scored via log-space MSE:

$$
\mathcal{L}_\text{ms} = \frac{1}{B\,C\,K}\sum_{b,c,k}\Big(\log\big(\hat{M}_{b,c,k} + \varepsilon\big)
- \log\big(M_{b,c,k} + \varepsilon\big)\Big)^2.
$$

`EegMsLoss` is bit-pinned against `METRICS["eeg_ms"]` to $< 10^{-12}$ across sample rates and hop
lengths in [`test_metric_losses.py`](../tests/test_metric_losses.py).

### 6.4 Total loss and component logging

The total loss $\mathcal{L} = \sum_i w_i \mathcal{L}_i$ is used for backpropagation. Unweighted
individual components $\mathcal{L}_i$ (and diagnostics like current rollout length $L$) are
recorded per epoch into `TrainingResult.train_components` and `val_components`, saved in
`training_stats.json`, and displayed in the training progress bar and loss curves.

---

## 7. Optimisation

### 7.1 Optimizer and schedule

`torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)` — Adam with
**decoupled** weight decay. Per parameter $\theta$, with $g_t = \nabla_\theta \mathcal{L}$, torch's
defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\hat\varepsilon = 10^{-8}$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, &
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\\
\hat m_t &= \tfrac{m_t}{1-\beta_1^t}, &
\hat v_t &= \tfrac{v_t}{1-\beta_2^t},\\
\theta_{t+1} &= \theta_t - \eta_t\Big(\tfrac{\hat m_t}{\sqrt{\hat v_t}+\hat\varepsilon} + \lambda\,\theta_t\Big).
\end{aligned}
$$

with decoupled decay $\lambda = \texttt{weight\_decay}$. The learning rate schedule
([`_lr_schedule`](../src/neuro/predictor/train.py)) steps once **per batch** over a budget of

$$
T_\text{total} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil \cdot (\texttt{epochs} - e_\text{first}),
\qquad
T_\text{warm} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil \cdot \texttt{warmup\_epochs},
$$

where $e_\text{first}$ is the first epoch actually run (§7.5). It is a `LinearLR` ramp and a
`CosineAnnealingLR` decay joined by a `SequentialLR` at $T_\text{warm}$:

$$
\eta_t = \begin{cases}
\texttt{learning\_rate} \cdot \tfrac{t + 1}{T_\text{warm}} & t < T_\text{warm}\\[4pt]
\tfrac{\texttt{learning\_rate}}{2}\Big(1 + \cos\pi\tfrac{t - T_\text{warm}}{T_\text{total} - T_\text{warm}}\Big) & t \ge T_\text{warm}
\end{cases}
$$

so $\eta$ climbs to the peak at $T_\text{warm}$ and anneals from there to $0$ at $T_\text{total}$.
`warmup_epochs = 0` (the default) collapses this to the bare cosine. The budget is computed for the
*full* `epochs`, so an early-stopped run simply never reaches the tail of the cosine.

The warm-up exists because the rollout is $N = \max_i(\text{span\_steps}_i)$ deep from the very
first batch (§6), so a randomly initialised model backpropagates through the whole horizon at epoch
0 regardless of where the curriculum starts. Taking that first, badly-conditioned gradient at the
peak learning rate is what the ramp avoids. $T_\text{warm}$ is clamped below $T_\text{total}$,
because a warm-started linear model (§7.4) skips ahead to `curr_start` and can be left with fewer
epochs than `warmup_epochs` asks for.

One training step is `loss.backward()` + `optimizer.step()` + `scheduler.step()`, in eager mode.
Nothing is JIT-compiled or `torch.compile`d.

### 7.2 Batching

`_shuffled_batches` in [`train.py`](../src/neuro/predictor/train.py) reshuffles the training sample
indices **every epoch** from a single `np.random.default_rng(training.seed + seed_offset)` created
once per run, then yields contiguous index slices of `batch_size`:

$$
\text{\#batches} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil,
$$

the final batch being smaller when $M_\text{train}$ is not a multiple of the batch size. There is no
`torch.utils.data.DataLoader`: the whole dataset is one resident float64 tensor on the target device,
so a batch is a single fancy-index gather. Validation is evaluated on the **whole** validation set in
a single `torch.no_grad()` call (not batched).

Because the shuffle RNG and `torch.manual_seed` are both driven by `training.seed + seed_offset`, a
run is reproducible from its config. `tests/test_predictor_train.py::test_same_seed_reproduces_and_offset_decorrelates`
pins this behaviour.

### 7.3 Curriculum schedule

Configured per loss term (e.g. `training.losses.curriculum_mse`). Let $e_0 = \texttt{curr\_start}$,
$e_1 = \texttt{curr\_end}$, and $N_\text{span} = \text{span\_steps}$. The trusted rollout length $L$
holds at 1 (teacher forcing) until $e_0$, ramps linearly $1 \to N_\text{span}$ between $e_0$ and $e_1$, and
holds at $N_\text{span}$ afterwards:

$$
L(e) = \operatorname{round}\big(1 + (N_\text{span} - 1)\cdot\operatorname{clip}(\tfrac{e - e_0}{\max(e_1 - e_0,\,1)},\,0,\,1)\big).
$$

The loss scores the first $L(e)$ steps of its span
([`CurriculumMSE.trusted_length`](../src/neuro/predictor/losses.py)). Epochs before $e_0$ train on pure teacher
forcing, epochs in $[e_0, e_1]$ grow the rollout $1 \to N_\text{span}$, and the remainder trains on
the full span objective.

### 7.4 Linear warm start

For `depth = 0` the single affine layer is initialised with the **exact** one-step least-squares
solution rather than randomly: `_warm_start_linear` solves
$\min_W \|[\,X_\text{1step}\;\mathbf{1}\,]W - Z_1\|^2$ with `np.linalg.lstsq` and copies the result
into `layer.weight` / `layer.bias`. $X_\text{1step}$ pairs the past-EEG block with the control window
shifted forward one step, matching what the rollout feeds the MLP at $j = 0$; $Z_1$ is the first
standardized target step.

A model that already solves the $L = 1$ problem has nothing to learn from the teacher-forcing phase,
so a warm-started run **skips straight to epoch `curriculum_mse.curr_start`** (if configured, else 0) — its loss
curves are $\texttt{epochs} - \texttt{curr\_start}$ entries long, not `epochs`. That surprises
readers of `training_stats.json`, where a linear preset's arrays are visibly shorter than a
nonlinear one's.

### 7.5 Training loop, validation, early stopping

[`_fit`](../src/neuro/predictor/train.py) runs the loop over epochs $e_\text{first}, \dots,
\texttt{epochs}-1$ (with $e_\text{first} = \texttt{curr\_start}$ for a warm-started linear model, else 0):

1. Compute the training loss with `ctx = LossContext(..., epoch=epoch)` on shuffled batches, take an
   optimizer step on each, and accumulate the batch losses and individual components.
2. **Validation loss** = `_batch_loss` on the full validation set with `ctx = LossContext(..., epoch=None)`
   (terminal schedule: all components active over their full spans), the quantity we ultimately minimise,
   comparable across epochs and rollout lengths.
3. **NaN guard**: if either the train or val loss is NaN, raise `ValueError("Loss is NaN…")`
   (the Optuna sweep catches this and prunes the trial).
4. **Best-model tracking**: `copy.deepcopy(model.state_dict())` whenever the validation loss
   improves. The deep copy is not incidental — a torch module is mutable, so keeping a reference
   would alias the live model and early stopping would silently return the *last* weights instead of
   the best. Pinned by
   `tests/test_predictor_train.py::test_returned_artifact_is_the_best_epoch_not_the_last`.
5. **Early stopping**: stop once `patience` consecutive epochs pass with no validation improvement.

On exit the best state dict is loaded back into the model, so `train` always returns the
lowest-validation-loss weights plus the per-epoch train/val loss and component histories.

---

## 8. Evaluation & artifacts

### 8.1 What `train()` returns

[`train(cfg, data_files, *, seed_offset=0)`](../src/neuro/predictor/train.py) performs **no I/O** —
it neither writes files nor imports `matplotlib`. It returns a
[`TrainingResult`](../src/neuro/predictor/train.py):

| field | is |
| ----- | -- |
| `artifact` | the best-epoch [`MLPArtifact`](../src/neuro/predictor/artifact.py): NumPy weights + fitted standardizers + native `dt` |
| `train_losses`, `val_losses` | per-epoch total loss, one entry per epoch actually run |
| `train_components`, `val_components` | per-epoch dict of unweighted loss components and diagnostics |
| `rollout` | a [`RolloutNMSE`](../src/neuro/artifacts.py) — free-run NMSE `pooled` and `per_step` evaluated over `eval_horizon_s` |
| `log_energy` | a [`LogEnergyError`](../src/neuro/artifacts.py) — free-run windowed-energy log-ratio error of §8.2.1, `pooled` and `per_position` |
| `val_trajs` | the held-out `(u, y)` trajectories, whole |
| `du_sensitivity` | the control-sensitivity scalar of §8.3 |

The caller decides what to persist and what to draw. `TrainingResult.save(artifact_dir)` writes
`model.npz` and `training_stats.json`; plotting lives in the run script (§8.5).

### 8.2 Free-run rollout NMSE

Carrying the notation of §1, §3 and §6 through this section, all in **raw EEG units**:

| symbol | is | indices |
| ------ | -- | ------- |
| $y_t \in \mathbb{R}^{C}$ | one measured EEG sample at time step $t$ (§1) | channel $c$ |
| $Y$ | the stacked validation targets (§3); $Y_{b,i,c}$ is horizon step $i$, channel $c$ of window $b$ | window $b$, step $i$, channel $c$ |
| $\hat{Y}$ | the model's prediction of $Y$ — the rollout $F_\theta$ of §5.2, run from the window's true history | as $Y$ |

Evaluation rollout horizon is configured explicitly via `training.eval_horizon_s`
($N_\text{eval} = \max(1, \operatorname{round}(\text{eval\_horizon\_s} \cdot f_s))$), independent of the
training rollout horizon $N = \max_i (\text{span\_steps}_i)$, and may exceed it.

Lowercase $y$ is always a single time step; uppercase $Y$ is always the stacked $N_\text{eval}$-step block
$[\,y_{k+1},\dots,y_{k+N_\text{eval}}\,]$ of §3, so $Y_{b,i,c} = y_{t_b + i,\,c}$ for the window starting at
$t_b$. A hat is a prediction. $F_\theta$ is autoregressive, so $\hat{Y}$ is already a free-running
rollout that feeds each predicted step back into its own history (§1, §5.2); there is no
teacher-forced evaluation anywhere in this section. Everything is scored in **raw EEG units**: the
model's standardized $\hat{z}$ is mapped back through `y_std.inverse_transform` (§3.1, §4.2).

Every reported NMSE — here and in the ESN path — divides a summed squared error by the **energy** of
the true signal over the same index set (the uncentred second moment, not the variance) through the
single definition in [`nmse`](../src/neuro/artifacts.py), so $1.0$ is always the score of the zero
predictor. [`evaluate_rollouts`](../src/neuro/artifacts.py) computes it over windows started on a
stride-25 grid, through the saved artifact's `prime_many`/`rollout_many` interface — the NumPy
rollout of §5.3, *not* the CasADi graph the MPC actually solves against, though the two are pinned to
each other by test — resolved per horizon step $i$ and then pooled:

$$
\text{NMSE}_i = \frac{\sum_{b,c}\big(Y_{b,i,c} - \hat{Y}_{b,i,c}\big)^2}{\sum_{b,c} Y_{b,i,c}^2},
\qquad
\text{NMSE} = \frac{\sum_{i}\sum_{b,c}\big(Y_{b,i,c} - \hat{Y}_{b,i,c}\big)^2}{\sum_{i}\sum_{b,c} Y_{b,i,c}^2}.
$$

Pooling divides summed error by summed energy — it is *not* the plain mean of the per-step curve but
its energy-weighted mean, so steps where the EEG is loud count for more. The per-step curve is the
informative half: it shows *where* along the horizon the model stops beating silence
($\text{NMSE}_i \to 1$), which the pooled scalar hides.

[`accumulate_rollout_errors`](../src/neuro/artifacts.py) stacks a trajectory's **whole** $t_0$ grid
and issues one batched `prime_many` + `rollout_many` call per trajectory rather than one pair per
window, which is worth 2–4× at the default `stride = 25` depending on model size (3.7× measured on
`nonlinear_full`'s dimensions). Both artifact families implement the batched pair.

The pooled NMSE remains the quantity `sweep_esn` minimises, and is available to the NN sweep as
`objective: rollout_nmse` — but it is **no longer the default objective**, for the reason §8.2.1
gives. Rolling out *past* the evaluation horizon $N_\text{eval}$ is a separate question, measured by
[`probe_rollout_horizon.py`](../scripts/probe_rollout_horizon.py), which reports the same
$\text{NMSE}_i$ alongside a power ratio $\sum \hat{Y}^2 / \sum Y^2$ — needed because a model that
decays to zero output scores $\text{NMSE}_i \to 1$, indistinguishable on that number alone from one
that is merely wrong.

### 8.2.1 Log-energy error — the functional the MPC actually costs

Waveform NMSE stops discriminating well before the horizons of interest. Its saturation value is
exactly $1.0$ (the zero predictor), the plant's phase is chaotic, and at a $\approx 9.3$ Hz peak the
waveform decorrelates within a couple hundred milliseconds — the same collapse
[`raw_series_grid`](../src/neuro/metrics.py) is built around. Past that point every candidate reads
$\approx 1.0$ and a sweep ranking on it is ranking sampler noise.

The deeper reason is that NMSE measures a quantity the controller discards. The MPC's stage cost is
`w_y * sumsqr(y_next) + w_u * sumsqr(u_curr)` ([`build_mpc_nlp`](../src/neuro/control.py)): it asks
the predictor for **energy over the horizon** and for how that energy moves with $u$, never for a
waveform. A phase-scrambled rollout carrying the right energy course yields the same optimal $u$.

[`evaluate_log_energy`](../src/neuro/artifacts.py) scores that functional directly. With
$E_{b,w} = \frac{1}{W C}\sum_{i \in w}\sum_c Y_{b,i,c}^2$ the cross-channel mean square of window
$b$ over trailing window position $w$ (length $W$ steps, spaced by the hop):

$$
D_w = \frac{1}{B}\sum_b \Big(\log(\hat{E}_{b,w} + \varepsilon) - \log(E_{b,w} + \varepsilon)\Big)^2,
\qquad D = \frac{1}{|w|}\sum_w D_w.
$$

Four choices, each load-bearing:

- **Log space**, because energy spans orders of magnitude between interictal and ictal and the
  controller responds to the ratio, not the difference.
- **Resolved per window $b$ before averaging.** Pooling numerator and denominator first — which is
  what the `pred_power` returned by `accumulate_rollout_errors` would give — lets over- and
  under-prediction cancel across windows, so a model right only *on average* would score perfectly.
- **Unbounded above**, so a prediction that decays to silence is scored as the failure it is rather
  than tying at $1.0$ with everything else. $\varepsilon = 10^{-12}$ mV⁴ floors the log of a
  collapsed prediction, not of a genuinely quiet one.
- **Window and hop follow the metrics layer's own `eeg_ms` convention** (`METRICS["eeg_ms"].window_s`
  and `DEFAULT_HOP_S`) rather than a config knob of their own, clamped where $N_\text{eval}$ is too
  short to hold one window.

Both scores are built on the shared batched-rollout generator
[`rollout_batches`](../src/neuro/artifacts.py) — extracted out of `accumulate_rollout_errors` so the
priming and free-run logic lives in one place — but they call it separately, so each traversal of the
validation windows happens once per score. Cheap next to training, and not worth fusing the two
accumulators for.

$D = 0$ is exact and lower is better. It is **not** an NMSE and does not share NMSE's scale; do not
compare the two numbers to each other.

### 8.3 `du_sensitivity` — is stimulation doing anything?

A predictor can score a good rollout NMSE by modelling the EEG's autonomous dynamics and ignoring the
stimulation entirely. Such a model is useless to the MPC, and the failure is invisible in every
metric above. `_du_sensitivity` makes it visible in seconds instead of after a closed-loop sweep: it
takes `torch.autograd.functional.jacobian` of the rollout with respect to the **future-control block
only** (history held fixed), in forward mode — the cheap direction here, since that block is far
narrower than the $N C$ rollout it drives — and reports the mean Frobenius norm over a fixed
subsample of 8 validation windows. A full Jacobian per window would dominate the training run.

**Read it within one config, never across configs.** The Jacobian is taken in **standardized space**
on both sides, so the number scales with whatever the `y` and `u` standardizers fitted on this
dataset. It is a "is the model responsive to stimulation at all" signal, not a comparable quantity.
The ESN path has no equivalent.

### 8.4 The artifact: one `.npz`, no framework

[`MLPArtifact`](../src/neuro/predictor/artifact.py) is a frozen dataclass of NumPy arrays. It
imports no deep-learning framework, and it is what everything downstream of training consumes.
`save` writes a single file (path convention: configs give a suffix-less stem such as
`artifacts/x/model`, and `save`/`load` apply `.with_suffix(".npz")`):

| npz key | contents |
| ------- | -------- |
| `meta` | 0-d unicode array holding `json.dumps(...)` of `model_type`, `activation`, `n_y`, `n_u`, `horizon`, `n_channels`, `n_controls`, `dt`, `downsample`, `n_layers` |
| `layer.<i>.weight`, `layer.<i>.bias` | the MLP stack in forward order, $(out, in)$ and $(out,)$ |
| `y_center`, `y_scale` | the EEG standardizer |
| `u_center`, `u_scale` | the control standardizer |

Storing `meta` as a 0-d `"<U"` array (not an object array) is deliberate: `np.load` reads it back
without `allow_pickle`, so loading an artifact never executes pickled code.
[`load_any_artifact`](../src/neuro/artifacts.py) dispatches on `meta["model_type"]`.

### 8.5 What lands in the artifact directory

`TrainingResult.save` writes two files; [`run_nn_predictor.py`](../scripts/run_nn_predictor.py) adds
the plots and a copy of the resolved config YAML for provenance.

| file | written by | contents |
| ---- | ---------- | -------- |
| `model.npz` | `save` | the artifact of §8.4 |
| `training_stats.json` | `save` | `train_loss[]`, `val_loss[]`, `train_components{}`, `val_components{}`, `nmse_rollout`, `nmse_rollout_per_step[]`, `du_sensitivity` |
| `loss_curve.png` | run script | total train vs. val loss and per-loss component curves per epoch |
| `comparison.png` | run script | free-run rollout fans vs. truth on the first held-out trajectory, up to 200 anchors, 4 channels |
| `<config>.yaml` | run script | copy of the resolved config, named after the source file |

Two changes here are worth flagging because they are visible to anyone reading old runs. The
teacher-forced `mse` key is **gone** — with `evaluate_model` deleted there is no teacher-forced
evaluation left in the pipeline, and `nmse_rollout` is the metric. And `comparison.png` now overlays
**free-run rollouts**, primed on true history and then running open-loop for `horizon` steps, where
it used to plot teacher-forced $N$-step predictions; the curves are correspondingly worse and
correspondingly more honest, since free-run is what the MPC does.

[`sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py) calls the same `save` into each

`trial_<n>/` and **does not plot**: a sweep produces one artifact per trial and nobody opens the
PNGs.

### 8.6 The torch-free seam

This is a load-bearing architectural property, not an implementation detail:

> `neuro.predictor.artifact`, `neuro.predictor.data`, `neuro.nn_predictor_casadi`, `neuro.control`
> and `neuro.artifacts` **never import torch**.

Training is the only thing in this repo that needs a deep-learning framework. Inference, the CasADi
bridge, the MPC and every closed-loop simulation run on NumPy and CasADi alone, reading the
framework-free artifact of §8.4. This keeps a `torch` import (and its multi-second cost, and
its threading defaults) out of the closed loop.

It is enforced, not merely intended:
`tests/test_predictor_module.py::test_control_path_never_imports_torch` imports each of those five
modules **in a subprocess** and asserts `'torch' not in sys.modules`. The subprocess is required —
by the time that test runs, pytest has long since imported torch via other test modules.

The split maps onto the package layout: `predictor/artifact.py` and `predictor/data.py` are
torch-free; `predictor/module.py`, `predictor/losses.py` and `predictor/train.py` are the torch side.

---

## 9. Configuration reference

All fields are validated strictly by pydantic ([`config.py`](../src/neuro/config.py)); unknown keys or
out-of-range values raise `ValidationError` rather than silently defaulting.

### `simulation`

| key           | default | meaning                                   |
| ------------- | ------- | ----------------------------------------- |
| `dt`          | `1e-4`  | raw simulation step (s)                   |
| `downsample`  | `1`     | decimation factor $d$                     |
| `n_steps`     | `null`  | downsampled steps loaded per trajectory (`null` ⇒ all) |
| `data_path`   | `null`  | directory of `.npz` trajectories          |
| `cutoff_hz`   | `null`  | explicit -3 dB low-pass cutoff before decimation (`null` ⇒ decimated Nyquist) |

### `model`

| key           | default | meaning                                                     |
| ------------- | ------- | ----------------------------------------------------------- |
| `n_y`         | `5`     | past EEG steps in history                                   |
| `n_u`         | `5`     | past control steps in history                               |
| `hidden_size` | `128`   | MLP width (ignored when `depth = 0`)                        |
| `depth`       | `2`     | hidden layers; `0` ⇒ linear model                           |
| `activation`  | `relu`  | `relu` / `tanh` / `softplus` (a `Literal`, so typos fail at load) |

### `training`

| key                          | default | meaning                                                       |
| ---------------------------- | ------- | ------------------------------------------------------------- |
| `epochs`                     | `100`   | max epochs                                                    |
| `warmup_epochs`              | `0`     | epochs of linear LR ramp before the cosine (must be `< epochs`) |
| `batch_size`                 | `128`   | SGD batch size                                                |
| `learning_rate` $\eta$       | `1e-3`  | AdamW peak LR, reached at `warmup_epochs` then cosine-annealed to 0 |
| `weight_decay` $\lambda$     | `1e-4`  | AdamW decoupled decay                                         |
| `train_split`                | `0.8`   | fraction used for training (tail held out for val)            |
| `seed`                       | `69`    | seed for weight init **and** the epoch shuffle (`+ seed_offset` in sweeps) |
| `patience`                   | `50`    | early-stopping patience (epochs)                              |
| `scaler`                     | `standard` | `standard` / `robust`                                      |
| `global_scaling`             | `false` | one shared scalar vs. per-channel scaling                     |
| `device`                     | `cpu`   | `cpu` / `cuda` — see the caveat below                         |
| `eval_horizon_s`             | *(required)* | evaluation rollout horizon in seconds                     |
| `losses`                     | *(required)* | composable loss terms (at least one must be active)           |

> **`device: cuda` is untested.** Only the CPU path has been exercised. The code moves the model and
> both resident dataset tensors onto `torch.device(training.device)` and is float64 throughout, which
> is a poor fit for consumer GPUs; treat CUDA as unvalidated rather than supported.

#### Loss specifications (`training.losses`)

The model's training rollout horizon is derived automatically as $N = \max_i (\text{span\_steps}_i)$,
where $\text{span\_steps} = \operatorname{round}(\text{span\_s} \cdot f_s)$. At least one active loss must have
`start_epoch = 0` (otherwise epoch 0 has no gradient, rejected at load).

- **`curriculum_mse`**:
  - `weight: float` *(required)*
  - `span_s: float` *(required)*
  - `curr_start: int` *(required)* — epoch where rollout starts expanding ($L = 1$ before it)
  - `curr_end: int` *(required)* — epoch where rollout reaches full span (held after; requires `curr_end >= curr_start`)
  - `start_epoch: int = 0` — epoch where this loss begins contributing to the gradient
- **`psd`**:
  - `weight: float` *(required)*
  - `span_s: float` *(required)* — requires $\operatorname{round}(\text{span\_s} \cdot f_s) \ge 2$
  - `start_epoch: int = 0`
- **`eeg_ms`** (and future metric twins):
  - `weight: float` *(required)*
  - `span_s: float` *(required)*
  - `window_s: float | null = null` — window duration; defaults to `METRICS[name].window_s` in `neuro.metrics` (0.1 s for `eeg_ms`; requires $1 \le \operatorname{round}(\text{window\_s} \cdot f_s) \le \text{span\_steps}$)
  - `hop_s: float | null = null` — window hop; defaults to `neuro.metrics.DEFAULT_HOP_S` (0.05 s) (requires $\operatorname{round}(\text{hop\_s} \cdot f_s) \ge 1$)
  - `start_epoch: int = 0`

**Units convention:** All time durations and horizons are specified in **seconds** (`span_s`,
`eval_horizon_s`, `window_s`, `hop_s`), converted to steps via $f_s = 1 / (\text{dt} \cdot \text{downsample})$.
All schedule checkpoints and gates are specified in integer **epochs** (`curr_start`, `curr_end`, `start_epoch`).

### Shipped presets

| preset | `epochs` | `patience` | `curriculum_mse.span_s` | extra loss | `curr_start` / `curr_end` |
| ------ | :------: | :--------: | :---------------------: | ---------- | :-----------------------: |
| [`nonlinear_full_8s.yaml`](../configs/nn_predictor/nonlinear_full_8s.yaml) | 300 | 100 | 1.0 | — | 100 / 250 |
| [`nonlinear_full_8s_eeg_ms.yaml`](../configs/nn_predictor/nonlinear_full_8s_eeg_ms.yaml) | 300 | 100 | 1.0 | `eeg_ms` (weight 0.5) | 100 / 250 |
| [`nonlinear_full_8s_no_curr.yaml`](../configs/nn_predictor/nonlinear_full_8s_no_curr.yaml) | 200 | 750 | 1.0 | — | 301 / 301 (never fires) |
| [`nonlinear_full_8s_mse02_eeg_ms.yaml`](../configs/nn_predictor/nonlinear_full_8s_mse02_eeg_ms.yaml) | 300 | 100 | 0.2 | `eeg_ms` (weight 0.08, from epoch 80) | 20 / 80 |
| [`nonlinear_full_8s_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_full_8s_mse02_psd.yaml) | 300 | 100 | 0.2 | `psd` (weight 1.0, from epoch 80) | 20 / 80 |

All five target the 50 Hz / 8 s ROAST dataset described in Section 2.2 and share `n_y=15`, `n_u=10`,
`hidden_size=64`, `depth=2`, `activation=softplus`, `batch_size=128`, `learning_rate=1e-5`,
`weight_decay=5e-4`, `train_split=0.8`, `seed=69`, `scaler=robust`, `global_scaling=true` and
`eval_horizon_s=1.0`.

The last two are a matched pair: the MSE is trusted only out to **0.2 s** and everything from there
to 1 s is shaped by the auxiliary term alone, so they isolate what each auxiliary loss contributes
past the point where the waveform is predictable. They are the only presets that set
`warmup_epochs` (10). Their weights are anchored on measured magnitudes: over a 1 s span at 50 Hz a
prediction that is merely a *different* EEG window scores `eeg_ms` ≈ 4.5 but `psd` ≈ 0.014, because
[`PSDLoss`](../src/neuro/predictor/losses.py) pools the batch into one spectrum per channel and so
compares batch-mean spectra rather than per-window ones. `eeg_ms` is therefore a per-window
objective that has to be scaled *down* to sit alongside the MSE, while `psd` acts as a barrier —
negligible when the spectrum matches, $O(10^2)$ when the rollout drifts to DC — and is weighted up.

---

## 10. Hyperparameter sweeps

[`scripts/sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py) wraps the same `train()` in an
**Optuna** study (`direction="minimize"`, persisted to a SQLite study DB in the sweep artifact
directory). A `sweep` config section declares per-field search spaces (`categorical` / `int` /
`float` / `loguniform`), including **dotted nested paths** such as `losses.curriculum_mse.curr_end` or
`losses.eeg_ms.weight`:

```yaml
sweep:
  artifact: "artifacts/sweep_losses"
  n_trials: 30
  objective: log_energy   # log_energy | val_loss | rollout_nmse | closed_loop
  training:
    learning_rate:
      type: loguniform
      low: 0.000005
      high: 0.0001
    losses.curriculum_mse.curr_end:
      type: int
      low: 50
      high: 200
```

Each trial samples overrides, expands dotted keys into nested dicts, deep-merges them onto the base
config, and re-validates the merged `model` / `training` sections. Sweep parameters on unconfigured
losses or overlapping with explicitly defined base values raise `ValidationError` at study setup.
Each trial uses `seed_offset = trial.number` to decorrelate otherwise identical runs and saves a full
artifact under `trial_<n>/`. A `ValueError` mentioning NaN is converted into `optuna.TrialPruned`.

### 10.1 Choosing the objective

`sweep.objective` selects what the study minimises. All four are **recorded as user attributes on
every trial regardless of which one is chosen**, so a finished study can be re-ranked on any of them
without re-running it.

| `objective` | is | cost |
| ----------- | -- | ---- |
| `log_energy` *(default)* | `result.log_energy.pooled` — the log-energy error of §8.2.1 | free; same rollouts as the NMSE |
| `val_loss` | `min(result.val_losses)` — the best-epoch validation loss of §7 | free; already computed |
| `rollout_nmse` | `result.rollout.pooled` — the incumbent waveform NMSE of §8.2 | free |
| `closed_loop` | the seizure-burden score from [`evaluate_closed_loop_suppression`](../src/neuro/closed_loop_eval.py) | one full simulation per seed per trial |

`log_energy` is the default because it is the only cheap option that scores the functional the
controller consumes (§8.2.1).

**`val_loss` is only comparable across trials while the loss composition is fixed.** It is the
honest choice when the sweep varies architecture and optimiser alone — which is what
[`sweep_nn_predictor.yaml`](../configs/nn_predictor/sweep_nn_predictor.yaml) does. The moment
`losses.*.weight` or `losses.*.span_s` enters the search space the number stops being comparable and
the objective actively selects small weights; use `log_energy` for those sweeps.

**`closed_loop` requires a `sweep.closed_loop` section** (enforced at config load) and is best used
as a *gate on finalists* rather than as the objective, for three reasons: seed variance with a
handful of seeds means ranking on seed luck; it confounds predictor quality with the MPC's own
`w_y`/`w_u` tuning, so a predictor that would win under different weights loses; and it costs
`len(seeds)` full simulations per trial. The recommended flow is to sweep on `log_energy`, then
re-run the top-$k$ trials with `objective: closed_loop`. Note that the closed-loop evaluation runs
whenever the section is present, whichever objective is selected, so its stats are recorded
alongside the others.

The score itself is the **seizure burden**: the fraction of regions seizing, averaged over the run
and over the seeds, plus `amplitude_weight` (default `0.0`) times the mean stimulation amplitude.
[`seizure_burden`](../src/neuro/closed_loop_eval.py) is the closed-loop reading of
[`seizure_state`](../src/neuro/metrics.py) — the same $s(t) \in [0,1]$ the `state_scoring` notebook
scores every observable against. It replaced a thresholded suppressed-seed count, which was integer
over `len(seeds)` and so handed the sampler a few wide plateaus with no gradient between them;
averaging over the run rather than reading the terminal window also rewards suppressing *early*.
`max_seizing_regions` still defines the reported `suppressed_seeds` diagnostic but no longer drives
the score.

---

## 11. Notes & gotchas

- **Derived training horizon vs explicit evaluation horizon.** Training rollout width is derived as
  $N = \max_i (\text{span\_steps}_i)$ across active loss terms, ensuring the computational graph is
  only as wide as needed for the losses. Evaluation rollout horizon is set independently by
  `eval_horizon_s` and may exceed the trained rollout width.
- **Seconds vs epochs units.** All spans, window lengths, hop intervals, and evaluation horizons are
  given in seconds (`span_s`, `eval_horizon_s`, `window_s`, `hop_s`). All schedule checkpoints and
  gates are given in integer epoch numbers (`curr_start`, `curr_end`, `start_epoch`).
- **Seconds round to samples with banker's rounding.** `round` is Python's, so a duration landing
  exactly on `.5` samples rounds to **even**: at $f_s = 50$, `hop_s: 0.05` gives $\operatorname{round}(2.5) = 2$
  samples, i.e. an effective 0.04 s. Write the value the sample grid can represent rather than one
  the config will silently reinterpret.
- **`eeg_ms` scores raw units, `psd` scores standardized ones.** `EegMsLoss` calls `ctx.to_raw`
  before taking the power, so its log-ratio includes the standardizer's offset; `PSDLoss` works on
  the standardized tensor and Welch detrends each segment anyway. The two log-ratios are not on the
  same footing, which is one reason their weights are not comparable.
- **Window and span bounds.** Metric losses enforce $\operatorname{round}(\text{window\_s} \cdot f_s) \ge 1$,
  $\operatorname{round}(\text{hop\_s} \cdot f_s) \ge 1$, and $\text{window\_s} \le \text{span\_s}$ at config
  load. `psd` requires $\operatorname{round}(\text{span\_s} \cdot f_s) \ge 2$.
- **Units.** EEG is in arbitrary units (see the project *uncalibrated units* note), so absolute MSE
  values are not physically meaningful; training happens in standardised space, and the reported
  NMSE is normalised by the true signal's energy so that it is scale-invariant.
- **Window overlap.** Sliding windows overlap heavily; the chronological train/val split (no shuffle
  *before* splitting) is what prevents near-duplicate leakage. Batches *within* the training set are
  still shuffled each epoch.
- **`depth = 0` is a genuinely linear model** — no activation is applied, so `activation` and
  `hidden_size` are inert for `linear_full.yaml`. It is also the only case that gets the
  least-squares warm start, and therefore the only case whose loss curves are shorter than `epochs`
  (§7.4).
- **Reproducibility.** A run is reproducible from `training.seed (+ seed_offset)`: it seeds both
  `torch.manual_seed` and the epoch-shuffle RNG. No `torch.use_deterministic_algorithms` is set — it
  buys nothing on the CPU path.
- **`n_channels` is the EEG channel count $C$ everywhere.** Model space and raw EEG differ only by
  the standardizer, which is shape-preserving, so there is a single channel dimension end to end.
- **The control window shifts before the MLP call — in training.** At rollout step $j$ the newest
  control $u_{k+j}$ is already inside $U^{(j+1)}$ when $\hat{y}_{k+j+1}$ is predicted (§5.2), because
  the feature row's u-window ends one step *behind* its y-window. The linear warm start has to
  reproduce that offset, and so does anything else that hand-builds a one-step input.
- **`MLPArtifact.rollout` shifts after, because `prime` aligns the windows.** A `prime` state ends
  both windows at the same step, so the first prediction is made before any future control enters
  and `u_future`'s last entry is never consumed — matching `ESNArtifact` exactly, so
  `accumulate_rollout_errors` can stay polymorphic over both families. Shifting first would score
  the sweep's objective on a model given one step of control lookahead it never had in training
  (§5.3).
