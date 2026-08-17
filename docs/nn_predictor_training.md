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

`prepare_datasets` returns a [`Datasets`](../src/neuro/predictor/data.py) record holding the **raw,
unscaled** train and validation windows *and* the held-out trajectories whole
(`val_trajs`), because free-run rollout scoring (§8) and the comparison plot both need the
un-windowed validation signals, not just their windows.

### 3.1 Optional latent PCA projection

If `model.latent_dim = k` is set (and $k <$ number of EEG channels), the y-transform gains a fixed
orthonormal PCA step *after* the channel standardizer (see §4.2), so **model space** is
standardize-then-project. [`PCAProjection.fit`](../src/neuro/transforms.py) fits the basis
$E \in \mathbb{R}^{k \times C}$ with mean $\mu \in \mathbb{R}^{C}$ on the **standardized training
EEG** (train split only). With $\tilde{y}$ the standardized channels:

$$
z_t = (\tilde{y}_t - \mu)\,E^{\top} \in \mathbb{R}^{k}, \qquad
\tilde{y}_t = z_t E + \mu \quad(\text{decode}).
$$

The network's *inputs* and *rollout* then live in the $k$-dimensional latent space (so the MPC can
run in the reduced state), but the **losses are still computed in EEG channel space**: each
model-space prediction $\hat{z}$ is decoded through the fixed, differentiable inverse projection
$\hat{z}E + \mu$ before the loss (§6). Because $E$ is orthonormal this leaves the MSE gradient
unchanged up to a constant (§6.4), while making the PSD term meaningful — it operates on real
EEG channels rather than the decorrelated latent components. Consequently the PSD loss is **not
incompatible** with a projection (`w_psd` may be > 0 with `latent_dim` set). At evaluation
predictions are decoded fully back to raw EEG so the reported NMSE is comparable across
`latent_dim`. Most shipped configs use `latent_dim = null` (projection disabled); the exceptions are
`meeting_seven/nonlinear_full_pca.yaml` ($k = 25$) and `meeting_seven/nonlinear_selected.yaml`
($k = 20$).

The decode basis and mean ride along inside the torch module as **registered buffers**
(`decode_basis`, `decode_mean`), not as extra arguments — so neither the loss nor the training loop
threads them through its signature, and they move with `model.to(device)`.

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

### 4.2 Transforms (standardize, then optionally project)

The EEG and control transforms are [`Pipeline`](../src/neuro/transforms.py)s of per-timestep-vector
maps, **fit on the training split only** (so no validation statistics leak in) and reused for
validation and at inference. Both PCA and the scalers share one `fit` / `transform` /
`inverse_transform` interface — a [`Standardizer`](../src/neuro/transforms.py) and a
[`PCAProjection`](../src/neuro/transforms.py) — so the projection is just another pipeline step
rather than a special case:

- **y-pipeline** $= [\,\text{Standardizer}\,]$, or $[\,\text{Standardizer},\ \text{PCAProjection}\,]$
  when `latent_dim` is set (§3.1). Fit on the raw past-EEG channel vectors of $X_\text{train}$; the
  PCA step is fit on the *standardized* channels.
- **u-pipeline** $= [\,\text{Standardizer}\,]$, fit on the raw past-control vectors.

The `Standardizer` subsumes the previous sklearn scalers:

- `scaler = "standard"` → subtract mean, divide by std.
- `scaler = "robust"` → subtract median, divide by IQR (robust to the large EEG bursts / bistable
  jumps). Used by all shipped configs.
- `global_scaling = true` → statistics pooled to **one** scalar shared across channels; `false` →
  **per-channel**. All shipped configs use global scaling.

Standardisation (per feature $j$): $\tilde{x} = (x - c_j)/s_j$, with $(c_j, s_j) = (\text{mean},
\text{std})$ for standard, $(\text{median}, \text{IQR})$ for robust.
[`transform_features`](../src/neuro/predictor/data.py) pushes the past-EEG block through the y-pipeline
(→ **model space**: standardized channels, or latent components under a projection) and the
past/future control blocks through the u-pipeline. The **targets $Y$ are only standardized** (the
channel `Standardizer`, not the PCA step), because the losses live in standardized-channel space and
the model's latent output is decoded into that space before scoring (§6). Evaluation is done in raw
EEG units after the full inverse pipeline (Section 8).

---

## 5. Model architecture

### 5.1 The one-step MLP $f_\theta$

An `nn.ModuleList` of `nn.Linear` layers inside
[`AutoregressiveMLP`](../src/neuro/predictor/module.py):

$$
f_\theta : \mathbb{R}^{\,n_y k + n_u m} \;\to\; \mathbb{R}^{k}.
$$

Widths are in **model space**: $k$ is the latent dimension under a projection and $C$ without one
(§3.1), so `n_channels` — not `n_eeg_channels` — sizes every layer.

- **Input width** $= n_y k + n_u m$ — the one-step model sees only $n_y$ past EEG steps and $n_u$
  past controls. (The future controls in the feature vector are fed in one at a time by the rollout,
  not all at once.)
- **Output width** $= k$ — a single next-EEG vector in model space.
- **Layer sizes** $= [\,n_y k + n_u m,\ \underbrace{\texttt{hidden\_size}, \dots}_{\texttt{depth}},\ k\,]$.
  - `depth = 0` ⇒ a single affine layer $f_\theta(v) = Wv + b$ with **no** hidden layer and no
    activation — i.e. a **linear** predictor (this is what `meeting_seven/linear_full.yaml` trains).
    The artifact reports this as `is_linear`, which is simply `len(layers) == 1`.
  - `depth = 1` ⇒ one hidden layer: $f_\theta(v) = W_2\,\sigma(W_1 v + b_1) + b_2$.
- **`activation`** $\sigma$ ∈ {`relu`, `tanh`, `softplus`}, applied after **every layer except the
  last**; the shipped configs use `softplus`, $\sigma(z) = \log(1 + e^{z})$. The literal type is
  enforced by the config schema, so a typo fails at config load rather than hours into a sweep at
  MPC construction time.

The layers are built with `dtype=torch.float64` and initialised by torch's default `nn.Linear`
initialiser under `torch.manual_seed(training.seed + seed_offset)`.

Everything the module sees is already in **model space** — the EEG block transformed by the
y-pipeline, the control blocks by the u-pipeline, exactly as `transform_features` produces them. The
module itself knows nothing about raw units.

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
bridge to `1e-10` over `latent_dim ∈ {None, k}`, `depth ∈ {0, 2}` and all three activations;
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

Implemented in [`predictor_loss`](../src/neuro/predictor/losses.py). The model rolls out in
**model space** (the $k$-dimensional latent components under a projection, else the $C$ channels),
producing $\hat{Z} \in \mathbb{R}^{B \times N \times k}$. Before any loss term this is **decoded into
standardized-channel space** with the fixed, differentiable inverse PCA,

$$
\hat{Y} = \hat{Z}E + \mu \in \mathbb{R}^{B \times N \times C}
$$

(the identity map when there is no projection, i.e. $k = C$). The targets $Y \in \mathbb{R}^{B
\times N \times C}$ are the standardized EEG channels (§4.2), so **both terms below live in EEG
channel space**.

### 6.1 Horizon-length curriculum MSE (the primary loss)

The MSE is scored over the first $L$ rollout steps selected by a **step mask** $s \in \{0,1\}^N$ (a
prefix of $L$ ones, $s_i = \mathbb{1}[i < L]$). With per-step errors

$$
e_i = \frac{1}{BC}\sum_{b,c}\big(\hat{Y}_{b,i,c} - Y_{b,i,c}\big)^2,
$$

the curriculum MSE is their masked mean:

$$
\boxed{\;\mathcal{L}_\text{MSE} = \frac{\sum_{i} s_i\, e_i}{\sum_i s_i}\;}
$$

- $L = 1$ → pure one-step (teacher forcing): only the first step (ground-truth context, no feedback).
- $L = N$ → pure $N$-step rollout: the objective we ultimately care about, but harder (exposes the
  model to its own compounding errors).

$L$ is grown from 1 to $N$ over training (Section 7.3), so the model is trained on progressively
longer free-running rollouts — a curriculum that keeps multi-step structure throughout and never
anneals a teacher-forcing weight down to a pure $N$-step loss.

Note that the *whole* horizon is always rolled out; the mask only decides which steps are scored.
Truncating the rollout to $L$ steps would have been cheaper, but the mask keeps the graph shape
fixed and the value comparable across epochs.

### 6.2 Auxiliary spectral loss (optional, horizon > 1)

One optional term pushes the *statistics* of the rollout toward the data, not just point accuracy.
It is active only when `w_psd > 0`, and is identically 0 when `horizon = 1`.

**PSD loss (log-spectral distance, Welch) — active only at full rollout.** The PSD is a
batch-of-snippets estimate over the **full** horizon window: per channel $c$, all $B$ windows are
concatenated into one length-$BN$ series and Welch's method is applied with `nperseg = N`,
`noverlap = 0` (each horizon-length window becomes one segment; averaging $B$ periodograms gives a
stable spectrum with $\lfloor N/2\rfloor + 1$ frequency bins). A meaningful spectrum needs the whole
window, so the term is **gated on only once $L = N$** (via the last mask entry $s_{N-1}$):

$$
\mathcal{L}_\text{PSD} = s_{N-1}\cdot\frac{1}{C\,F}\sum_{c,f}\Big(\log\big(\hat{P}_{c,f} + \varepsilon\big)
- \log\big(P_{c,f} + \varepsilon\big)\Big)^2, \qquad \varepsilon = 10^{-8}.
$$

[`welch_psd`](../src/neuro/predictor/losses.py) is a differentiable, bit-faithful replica of
`scipy.signal.welch(x, nperseg=N, noverlap=0, axis=-1)` under that call's defaults —
`detrend="constant"` (per-segment mean removed *before* windowing), a **periodic** Hann window,
`scaling="density"` ($1/(f_s \sum w^2)$), one-sided folding (every bin doubled except DC, and except
Nyquist when $N$ is even), and mean averaging over segments. It is pinned against SciPy to `1e-10`
for even and odd `nperseg` in
[`test_predictor_losses.py`](../tests/test_predictor_losses.py); the scaling conventions are the
entire reason that test exists.

There is no functional-connectivity term. An FC loss existed in the JAX pipeline and was deleted
together with its `w_fc` knob: every shipped config ran it at `0.0`, so it was never load-bearing.

### 6.3 Total loss

$$
\boxed{\;\mathcal{L} = \mathcal{L}_\text{MSE} \;+\; w_\text{psd}\,\mathcal{L}_\text{PSD}\;}
$$

$w_\text{psd}$ is a plain weight on the raw log-spectral distance. The JAX version additionally
divided each auxiliary term by its own detached value, making the term's *value* scale-free and its
gradient a relative one; that normalisation went away with the FC term, so `w_psd` now has to absorb
the absolute scale of the log-spectral distance itself. The unweighted `mse` and `psd` values are
returned alongside the total for logging, and are what the progress bar shows.

### 6.4 Why decoding is safe for the MSE (orthonormality)

The PCA basis has orthonormal rows ($EE^\top = I_k$). Split a channel target into its in-subspace
part and residual, $Y - \mu = ZE + r$ with $rE^\top = 0$. For a latent prediction $\hat{Z}$ decoded
to $\hat{Y} = \hat{Z}E + \mu$,

$$
\|\hat{Y} - Y\|^2 = \|(\hat{Z}-Z)E\|^2 + \|r\|^2 = \|\hat{Z} - Z\|^2 + \|r\|^2,
$$

and $\|r\|^2$ is parameter-independent, so $\nabla_\theta$ of the channel-space (summed) MSE equals
$\nabla_\theta$ of the latent MSE. Computing the MSE in EEG space therefore does **not** change what
the model learns for point accuracy — it only makes the reported number honest and, crucially,
unlocks the PSD term, which *is* genuinely different in channel space. The two mean-reductions
divide the same summed error by $B\!N\!C$ vs $B\!N\!k$, so the gradients are parallel with ratio
$k/C$ — pinned as a `torch.autograd` regression test in
[`test_predictor_losses.py`](../tests/test_predictor_losses.py).

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

with decoupled decay $\lambda = \texttt{weight\_decay}$. Unlike the JAX pipeline the learning rate is
**not** constant: a `torch.optim.lr_scheduler.CosineAnnealingLR` steps once **per batch** and anneals
$\eta$ from $\texttt{learning\_rate}$ down to $0$ over

$$
T_\text{max} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil \cdot (\texttt{epochs} - e_\text{first}),
\qquad
\eta_t = \tfrac{\texttt{learning\_rate}}{2}\Big(1 + \cos\tfrac{\pi t}{T_\text{max}}\Big),
$$

where $e_\text{first}$ is the first epoch actually run (§7.5). $T_\text{max}$ is budgeted for the
*full* `epochs`, so an early-stopped run simply never reaches the tail of the cosine.

One training step is `loss.backward()` + `optimizer.step()` + `scheduler.step()`, in eager mode.
Nothing is JIT-compiled or `torch.compile`d.

### 7.2 Batching

`_shuffled_batches` in [`train.py`](../src/neuro/predictor/train.py) reshuffles the training sample
indices **every epoch** from a single `np.random.default_rng(training.seed + seed_offset)` created
once per run, then yields contiguous index slices of `batch_size`:

$$
\text{#batches} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil,
$$

the final batch being smaller when $M_\text{train}$ is not a multiple of the batch size. There is no
`torch.utils.data.DataLoader`: the whole dataset is one resident float64 tensor on the target device,
so a batch is a single fancy-index gather. Validation is evaluated on the **whole** validation set in
a single `torch.no_grad()` call (not batched).

Because the shuffle RNG and `torch.manual_seed` are both driven by `training.seed + seed_offset`, a
run is reproducible from its config. This was **not** true of the JAX pipeline, whose per-epoch
shuffle used a bare unseeded `np.random.default_rng()`; `seed` there only fixed weight
initialisation. `tests/test_predictor_train.py::test_same_seed_reproduces_and_offset_decorrelates`
pins the new behaviour.

### 7.3 Curriculum schedule

Let $e_0 = \texttt{curriculum\_start\_epoch}$, $e_1 = \texttt{curriculum\_end\_epoch}$, and
$N^\star = \min(\texttt{curriculum\_max\_steps} \text{ or } N,\; N)$. The trusted rollout length $L$
holds at 1 (teacher forcing) until $e_0$, ramps linearly $1 \to N^\star$ between $e_0$ and $e_1$, and
holds at $N^\star$ afterwards:

$$
L(e) = \operatorname{clip}\!\Big(\operatorname{round}\big(1 + (N^\star - 1)\cdot\operatorname{clip}(\tfrac{e - e_0}{\max(e_1 - e_0,\,1)},\,0,\,1)\big),\ 1,\ N^\star\Big).
$$

The per-epoch step mask is the prefix of $L(e)$ ones over the full horizon width
([`curriculum_mask`](../src/neuro/predictor/losses.py)). So epochs before $e_0$ train on pure teacher
forcing, epochs in $[e_0, e_1]$ grow the rollout $1 \to N^\star$, and the remainder trains on the
$N^\star$-step objective (with the PSD term active only if $N^\star = N$, since the gate is the
*last* mask entry). `curriculum_max_steps` therefore lets a long-horizon model be trained on a
shorter trusted prefix; leaving it `null` (every shipped config) means $N^\star = N$.

### 7.4 Linear warm start

For `depth = 0` the single affine layer is initialised with the **exact** one-step least-squares
solution rather than randomly: `_warm_start_linear` solves
$\min_W \|[\,X_\text{1step}\;\mathbf{1}\,]W - Z_1\|^2$ with `np.linalg.lstsq` and copies the result
into `layer.weight` / `layer.bias`. $X_\text{1step}$ pairs the past-EEG block with the control window
shifted forward one step, matching what the rollout feeds the MLP at $j = 0$; $Z_1$ is the first
target step, projected into latent space if PCA is on.

A model that already solves the $L = 1$ problem has nothing to learn from the teacher-forcing phase,
so a warm-started run **skips straight to epoch $e_0$** — its loss curves are
$\texttt{epochs} - \texttt{curriculum\_start\_epoch}$ entries long, not `epochs`. That surprises
readers of `training_stats.json`, where a linear preset's arrays are visibly shorter than a
nonlinear one's.

### 7.5 Training loop, validation, early stopping

[`_fit`](../src/neuro/predictor/train.py) runs the loop over epochs $e_\text{first}, \dots,
\texttt{epochs}-1$ (with $e_\text{first} = e_0$ for a warm-started linear model, else 0):

1. Compute the step mask $L(e)$; iterate shuffled batches, take an optimizer step on each,
   accumulate the batch losses; the epoch train loss is the mean over batches.
2. **Validation loss** = `predictor_loss` on the full validation set with the full-horizon mask
   ($L = N$) — i.e. always the pure $N$-step rollout error (plus the PSD term when `w_psd > 0`), the
   quantity we ultimately minimise, comparable across epochs and rollout lengths.
3. **NaN guard**: if either the train or val loss is NaN, raise `ValueError("Loss is NaN…")`
   (the Optuna sweep catches this and prunes the trial).
4. **Best-model tracking**: `copy.deepcopy(model.state_dict())` whenever the validation loss
   improves. The deep copy is not incidental — a torch module is mutable, so keeping a reference
   would alias the live model and early stopping would silently return the *last* weights instead of
   the best. Pinned by
   `tests/test_predictor_train.py::test_returned_artifact_is_the_best_epoch_not_the_last`.
5. **Early stopping**: stop once `patience` consecutive epochs pass with no validation improvement.

On exit the best state dict is loaded back into the model, so `train` always returns the
lowest-validation-loss weights plus the per-epoch train/val loss histories.

---

## 8. Evaluation & artifacts

### 8.1 What `train()` returns

[`train(cfg, data_files, *, seed_offset=0)`](../src/neuro/predictor/train.py) performs **no I/O** —
it neither writes files nor imports `matplotlib`. It returns a
[`TrainingResult`](../src/neuro/predictor/train.py):

| field | is |
| ----- | -- |
| `artifact` | the best-epoch [`MLPArtifact`](../src/neuro/predictor/artifact.py): NumPy weights + fitted pipelines + native `dt` |
| `train_losses`, `val_losses` | per-epoch loss, one entry per epoch actually run |
| `rollout` | a [`RolloutNMSE`](../src/neuro/artifacts.py) — free-run NMSE `pooled` and `per_step` |
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

Lowercase $y$ is always a single time step; uppercase $Y$ is always the stacked $N$-step block
$[\,y_{k+1},\dots,y_{k+N}\,]$ of §3, so $Y_{b,i,c} = y_{t_b + i,\,c}$ for the window starting at
$t_b$. A hat is a prediction. $F_\theta$ is autoregressive, so $\hat{Y}$ is already a free-running
rollout that feeds each predicted step back into its own history (§1, §5.2); there is no
teacher-forced evaluation anywhere in this section. Everything is scored in **raw EEG units**: when a
latent projection is active the model's $\hat{z}$ is decoded through $\hat{z}E + \mu$ and then the
standardizer's inverse (§3.1, §4.2), so the numbers are comparable across `latent_dim`.

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

The pooled NMSE is the **Optuna objective** (minimised) in the sweep — the same quantity `sweep_esn`
minimises, so the two model families rank on one metric. Rolling out *past* the trained horizon $N$
is a separate question, measured by
[`probe_rollout_horizon.py`](../scripts/probe_rollout_horizon.py), which reports the same
$\text{NMSE}_i$ alongside a power ratio $\sum \hat{Y}^2 / \sum Y^2$ — needed because a model that
decays to zero output scores $\text{NMSE}_i \to 1$, indistinguishable on that number alone from one
that is merely wrong.

### 8.3 `du_sensitivity` — is stimulation doing anything?

A predictor can score a good rollout NMSE by modelling the EEG's autonomous dynamics and ignoring the
stimulation entirely. Such a model is useless to the MPC, and the failure is invisible in every
metric above. `_du_sensitivity` makes it visible in seconds instead of after a closed-loop sweep: it
takes `torch.autograd.functional.jacobian` of the rollout with respect to the **future-control block
only** (history held fixed), in forward mode — the cheap direction here, since that block is far
narrower than the $N k$ rollout it drives — and reports the mean Frobenius norm over a fixed
subsample of 8 validation windows. A full Jacobian per window would dominate the training run.

**Read it within one config, never across configs.** The Jacobian is taken in **model space**: the
output is latent under PCA and standardized either way, and the input is standardized control. So the
number scales with `latent_dim` and with whatever the `u` standardizer fitted on this dataset. It is
a "is the model responsive to stimulation at all" signal, not a comparable quantity. The ESN path has
no equivalent.

### 8.4 The artifact: one `.npz`, no framework

[`MLPArtifact`](../src/neuro/predictor/artifact.py) is a frozen dataclass of NumPy arrays. It
imports neither torch nor JAX, and it is what everything downstream of training consumes.
`save` writes a single file (path convention: configs give a suffix-less stem such as
`artifacts/x/model`, and `save`/`load` apply `.with_suffix(".npz")`):

| npz key | contents |
| ------- | -------- |
| `meta` | 0-d unicode array holding `json.dumps(...)` of `model_type`, `activation`, `n_y`, `n_u`, `horizon`, `n_channels`, `n_controls`, `n_eeg_channels`, `dt`, `downsample`, `n_layers`, and the two pipelines' step tags |
| `layer.<i>.weight`, `layer.<i>.bias` | the MLP stack in forward order, $(out, in)$ and $(out,)$ |
| `y.0.center`, `y.0.scale` | the EEG standardizer |
| `y.1.basis`, `y.1.mean` | the PCA step, present only when `latent_dim` is set |
| `u.0.center`, `u.0.scale` | the control standardizer |

Storing `meta` as a 0-d `"<U"` array (not an object array) is deliberate: `np.load` reads it back
without `allow_pickle`, so loading an artifact never executes pickled code.
[`load_any_artifact`](../src/neuro/artifacts.py) dispatches on the presence of `model.npz` — the ESN
writes `model.weights.npz`, so there is no collision — and then on `meta["model_type"]`.

This replaces the JAX pipeline's three-file `model.eqx` / `model.json` / `model.scalers.npz` set.
**There is no backward compatibility**: `.eqx` artifacts cannot be loaded and must be retrained.

### 8.5 What lands in the artifact directory

`TrainingResult.save` writes two files; [`run_nn_predictor.py`](../scripts/run_nn_predictor.py) adds
the plots and a copy of the resolved config YAML for provenance.

| file | written by | contents |
| ---- | ---------- | -------- |
| `model.npz` | `save` | the artifact of §8.4 |
| `training_stats.json` | `save` | exactly `train_loss[]`, `val_loss[]`, `nmse_rollout`, `nmse_rollout_per_step[]`, `du_sensitivity` |
| `loss_curve.png` | run script | train vs. val loss per epoch |
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
framework-free artifact of §8.4. That is what let the whole JAX → torch rewrite happen without the
control path changing at all, and it is what keeps a `torch` import (and its multi-second cost, and
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
| `horizon` $N$ | `5`     | direct prediction horizon                                   |
| `hidden_size` | `128`   | MLP width (ignored when `depth = 0`)                        |
| `depth`       | `2`     | hidden layers; `0` ⇒ linear model                           |
| `activation`  | `relu`  | `relu` / `tanh` / `softplus` (a `Literal`, so typos fail at load) |
| `latent_dim`  | `null`  | PCA latent components $k$; `null` ⇒ no projection           |

### `training`

| key                          | default | meaning                                                       |
| ---------------------------- | ------- | ------------------------------------------------------------- |
| `epochs`                     | `100`   | max epochs                                                    |
| `batch_size`                 | `128`   | SGD batch size                                                |
| `learning_rate` $\eta$       | `1e-3`  | AdamW peak LR, cosine-annealed to 0                           |
| `weight_decay` $\lambda$     | `1e-4`  | AdamW decoupled decay                                         |
| `train_split`                | `0.8`   | fraction used for training (tail held out for val)            |
| `curriculum_start_epoch`     | `0`     | epoch the rollout length starts growing ($L = 1$ before it)   |
| `curriculum_end_epoch`       | `80`    | epoch the rollout length reaches $N^\star$ (held after)       |
| `curriculum_max_steps`       | `null`  | cap $N^\star$ on the trusted rollout length; `null` ⇒ the horizon |
| `seed`                       | `69`    | seed for weight init **and** the epoch shuffle (`+ seed_offset` in sweeps) |
| `w_psd`                      | `0.0`   | weight of the auxiliary PSD loss                              |
| `patience`                   | `50`    | early-stopping patience (epochs)                              |
| `scaler`                     | `standard` | `standard` / `robust`                                      |
| `global_scaling`             | `false` | one shared scalar vs. per-channel scaling                     |
| `device`                     | `cpu`   | `cpu` / `cuda` — see the caveat below                         |

> **`device: cuda` is untested.** Only the CPU path has been exercised. The code moves the model and
> both resident dataset tensors onto `torch.device(training.device)` and is float64 throughout, which
> is a poor fit for consumer GPUs; treat CUDA as unvalidated rather than supported.

`curriculum_end_epoch >= curriculum_start_epoch` is enforced by a model validator on
`NNPredictorConfig`.

### Shipped presets

| preset                                                              | `depth` | `n_u` | `lr`      | `wd`     | `batch` | `curr.` | `w_psd` |
| ------------------------------------------------------------------- | :-----: | :---: | --------- | -------- | :-----: | :-----: | :-----: |
| [`meeting_seven/linear_full.yaml`](../configs/nn_predictor/meeting_seven/linear_full.yaml)      | 0 (linear) | 7  | $10^{-4}$ | $10^{-3}$ | 512   | 0.9     | 0       |
| [`meeting_seven/nonlinear_full.yaml`](../configs/nn_predictor/meeting_seven/nonlinear_full.yaml)| 1       | 10    | $10^{-5}$ | $5\!\cdot\!10^{-4}$ | 128 | 0.8 | 0     |

`curr.` is `curriculum_end_epoch` as a fraction of `epochs`. Common to both: `n_y=15`, `horizon=20`,
`hidden_size=128`, `activation=softplus`, `latent_dim=null`, `epochs=250`, `train_split=0.8`,
`seed=69`, `patience=100`, `scaler=robust`, `global_scaling=true`, and the 100 Hz / 20 s data
described in Section 2.2. The other `meeting_seven/` presets vary the montage
(`*_selected.yaml`, 25 channels) and the projection (`nonlinear_full_pca.yaml`, $k = 25$);
`nonlinear_full_8s*.yaml` target the 50 Hz / 8 s ROAST dataset at `horizon=50`, `depth=2`.
No shipped config sets `w_psd > 0`, `curriculum_max_steps`, `cutoff_hz` or `device`.

---

## 10. Hyperparameter sweeps

[`scripts/sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py) wraps the same `train()` in an
**Optuna** study (`direction="minimize"`, persisted to a SQLite study DB in the sweep artifact
directory). A `sweep` config section declares per-field search spaces (`categorical` / `int` /
`float` / `loguniform`); each trial samples overrides, re-validates the merged `model` / `training`
sections, uses `seed_offset = trial.number` to decorrelate otherwise identical trials, and saves a
full artifact under `trial_<n>/` along with its resolved `trial_config.yaml`. A `ValueError`
mentioning NaN is converted into `optuna.TrialPruned`.

The objective is `result.rollout.pooled` — the free-run rollout NMSE of §8.2 — recorded on the trial
as the `nmse_rollout` user attribute. If the sweep config carries a `closed_loop` section, the trial
instead returns the closed-loop suppression score from
[`evaluate_closed_loop_suppression`](../src/neuro/closed_loop_eval.py), which loads the trial's
`model.npz` and runs the MPC; the rollout NMSE is still recorded alongside it.

---

## 11. Notes & gotchas

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
- **The PSD loss needs `horizon > 1`** and is computed in EEG channel space, so it works **with**
  `latent_dim` too (the latent rollout is decoded first; §3.1, §6). For `horizon = 1` it is skipped
  outright.
- **Reproducibility.** A run is reproducible from `training.seed (+ seed_offset)`: it seeds both
  `torch.manual_seed` and the epoch-shuffle RNG. No `torch.use_deterministic_algorithms` is set — it
  buys nothing on the CPU path. Note this is a *change*: the JAX pipeline's shuffle was unseeded, so
  its runs were not bit-reproducible even at fixed `seed`.
- **`n_channels` is model space, `n_eeg_channels` is raw EEG.** Under a projection the first is the
  latent $k$ and the second is $C$; without one they are equal. Nearly every shape bug in this
  pipeline is these two swapped.
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
