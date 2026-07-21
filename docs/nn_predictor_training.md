# NN Predictor — Training Setup

A detailed, end-to-end description of how the neural-network EEG predictor is
trained. The predictor is a JAX/[Equinox](https://github.com/patrick-kidger/equinox)
MLP, optimised with [Optax](https://github.com/google-deepmind/optax), that learns a
**one-step-ahead** EEG map and is unrolled autoregressively to produce a
multi-step-ahead forecast conditioned on past EEG and past/future stimulation.

Source of truth:

- Training pipeline: [`src/neuro/nn_training.py`](../src/neuro/nn_training.py)
- Model + artifact + inference: [`src/neuro/prediction.py`](../src/neuro/prediction.py)
- Config schema: [`src/neuro/config.py`](../src/neuro/config.py)
- Entry points: [`scripts/run_nn_predictor.py`](../scripts/run_nn_predictor.py),
  [`scripts/sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py)
- Example configs: [`configs/nn_predictor/`](../configs/nn_predictor/)

> All computation runs in **float64** (`jax.config.update("jax_enable_x64", True)`).

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

---

## 2. Training data

### 2.1 Where it comes from

Each trajectory is an `.npz` file produced by the tES simulation experiments and stored in a
directory such as `data/experiment_excited/train/`. Files are discovered and sorted by
[`resolve_data_files`](../src/neuro/config.py); **every** `.npz` in the directory becomes a training
trajectory. Each file provides two arrays:

| key (new / legacy)                | meaning                       | shape        |
| --------------------------------- | ----------------------------- | ------------ |
| `y_mea` / `universal_y_mea`       | measured EEG output           | $(T, C)$     |
| `u` / `universal_u`               | stimulation input             | $(T, m)$     |

The recordings are **persistently exciting** tES sequences (broadband stimulation), so the control
$u$ visits enough of the input space for the network to learn the input→output response rather than
just the autonomous dynamics. The stimulation obeys **Kirchhoff's current law** — each row of $u$
sums to zero across the $m$ electrodes (no net injected current), matching the constraint the MPC
controllers enforce. Data generated before this fix is physically invalid; if `data/experiment_excited/`
is empty, regenerate it with `uv run scripts/run_simulation.py configs/simulation/experiment_excited.yaml`
before training (older wrong-stimulation datasets are archived under `data/pre_kirchhoff_wrong_stim/`).

### 2.2 Loading and downsampling

[`load_trajectory`](../src/neuro/nn_training.py) reads at most `n_steps · downsample` raw samples and
decimates by taking every `downsample`-th sample:

$$
y^{\text{ds}}_k = y^{\text{raw}}_{k\cdot d}, \qquad
u^{\text{ds}}_k = u^{\text{raw}}_{k\cdot d}, \qquad k = 0,\dots,n_\text{steps}-1,
$$

with $d = \texttt{downsample}$. The effective sample step is

$$
\Delta t_\text{real} = \Delta t \cdot d.
$$

Example (all shipped configs): $\Delta t = 10^{-4}\,\text{s}$, $d = 100 \Rightarrow \Delta t_\text{real}
= 10^{-2}\,\text{s}$ (100 Hz), and $n_\text{steps} = 500$ ⇒ **5 s per trajectory**. This
$\Delta t_\text{real}$ is stored in the artifact and becomes the model's native step at inference.

---

## 3. Dataset construction (supervised windows)

[`build_dataset_for_trajectory`](../src/neuro/nn_training.py) turns each trajectory into
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
along the sample axis in [`prepare_datasets`](../src/neuro/nn_training.py):

$$
X \in \mathbb{R}^{M \times (n_y C + n_u m + N m)}, \qquad
Y \in \mathbb{R}^{M \times N C}, \qquad
M = \sum_{\text{traj}} \big(T_\text{traj} - N - k_\text{start}\big).
$$

The control count is recovered from the feature width:

$$
m = \frac{\text{width}(X) - n_y C}{\,n_u + N\,}.
$$

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
unchanged up to a constant (§6.4), while making the PSD/FC terms meaningful — they operate on real
EEG channels rather than the decorrelated latent components. Consequently PSD/FC are **no longer
incompatible** with a projection (`w_psd`, `w_fc` may be > 0 with `latent_dim` set; see
[`latent_stats.yaml`](../configs/nn_predictor/latent_stats.yaml)). At evaluation predictions are
decoded fully back to raw EEG so the reported MSE is comparable across `latent_dim`. Most shipped
configs use `latent_dim = null` (projection disabled).

---

## 4. Preprocessing: split and scaling

### 4.1 Train / validation split

A **chronological** (index-order, no shuffle) split is applied to the concatenated dataset:

$$
\text{split} = \lfloor \texttt{train\_split} \cdot M \rfloor, \qquad
(X_\text{train}, Y_\text{train}) = (X_{0:\text{split}},\,Y_{0:\text{split}}), \qquad
(X_\text{val}, Y_\text{val}) = (X_{\text{split}:},\,Y_{\text{split}:}).
$$

Holding out the tail (rather than random sampling) limits leakage between the heavily-overlapping
sliding windows: a random split would put nearly-identical adjacent windows in both sets.

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
[`transform_features`](../src/neuro/nn_training.py) pushes the past-EEG block through the y-pipeline
(→ **model space**: standardized channels, or latent components under a projection) and the
past/future control blocks through the u-pipeline. The **targets $Y$ are only standardized** (the
channel `Standardizer`, not the PCA step), because the losses live in standardized-channel space and
the model's latent output is decoded into that space before scoring (§6). Evaluation is done in raw
EEG units after the full inverse pipeline (Section 8).

---

## 5. Model architecture

### 5.1 The one-step MLP $f_\theta$

Built with `eqx.nn.MLP` in [`create_model`](../src/neuro/nn_training.py):

$$
f_\theta : \mathbb{R}^{\,n_y C + n_u m} \;\to\; \mathbb{R}^{C}.
$$

- **`in_size`** $= n_y C + n_u m$ — the one-step model sees only $n_y$ past EEG steps and $n_u$ past
  controls. (The future controls in the feature vector are fed in one at a time by the rollout, not
  all at once.)
- **`out_size`** $= C$ — a single next-EEG vector.
- **`width_size`** $= \texttt{hidden\_size}$, **`depth`** $= \texttt{depth}$ hidden layers.
  - `depth = 0` ⇒ a single affine layer $f_\theta(v) = Wv + b$ with **no** hidden layer and no
    activation — i.e. a **linear** predictor (this is what `linear_best.yaml` trains).
  - `depth = 1` ⇒ one hidden layer: $f_\theta(v) = W_2\,\sigma(W_1 v + b_1) + b_2$.
- **`activation`** $\sigma$ ∈ {`relu`, `tanh`, `softplus`} between hidden layers (see
  [`get_activation`](../src/neuro/prediction.py)); the shipped configs use `softplus`,
  $\sigma(z) = \log(1 + e^{z})$. The final layer is linear (identity output activation).

Weights are initialised from a JAX PRNG key seeded by `training.seed (+ seed_offset)`.

### 5.2 Autoregressive rollout $F_\theta$

[`AutoregressivePredictor.__call__`](../src/neuro/prediction.py) unrolls $f_\theta$ over the horizon
with `jax.lax.scan`. Given one feature vector $x$, it splits it into the history windows
$Y^{(0)} = y_{k-n_y+1:k} \in \mathbb{R}^{n_y \times C}$, $U^{(0)} = u_{k-n_u:k-1} \in \mathbb{R}^{n_u
\times m}$, and the future-control sequence $u_{k:k+N-1}$. For each rollout step $j = 0,\dots,N-1$
with incoming control $u_{k+j}$:

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
distinction drives the curriculum loss below.

`predict_batch` = `eqx.filter_jit(jax.vmap(F_θ))` applies the rollout across a batch.

---

## 6. Loss function

Implemented in [`compute_loss`](../src/neuro/nn_training.py). The model rolls out in **model space**
(the $k$-dimensional latent components under a projection, else the $C$ channels), producing
$\hat{Z} \in \mathbb{R}^{B \times Nk}$. Before any loss term this is **decoded into standardized-channel
space** with the fixed, differentiable inverse PCA,

$$
\hat{Y} = \hat{Z}E + \mu \in \mathbb{R}^{B \times NC}
$$

(the identity map when there is no projection, i.e. $k = C$). The targets $Y \in \mathbb{R}^{B \times
NC}$ are the standardized EEG channels (§4.2), so **all terms below live in EEG channel space**. Write
$\hat{Y}^{(1)} = \hat{Y}[:, :C]$ for the first predicted step.

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

### 6.2 Auxiliary spectral & connectivity losses (optional, horizon > 1)

Two optional terms push the *statistics* of the rollout toward the data, not just point accuracy.
Only active when their weight is > 0; both are 0 when `horizon = 1`. Reshape $\hat{Y}, Y$ to
trajectories $(B, N, C)$.

**FC loss (functional connectivity) — grows with the curriculum.** Like the MSE, the FC is scored over
the first-$L$ trusted steps: the step mask $s$ weights the pooled $(BN, C)$ timepoints and a single
weighted channel×channel Pearson-correlation matrix is formed (diagonal zeroed). A channel
correlation is well-estimated from the pooled samples and needs no long window, so it can grow with
$L$. With an all-ones mask this reduces to the plain pooled correlation:

$$
\mathcal{L}_\text{FC} = \frac{1}{C^2}\sum_{i,j}\big(\widehat{\text{FC}}_{ij} - \text{FC}_{ij}\big)^2,
\qquad \text{FC} = \operatorname{corr}_s(Y),\ \ \operatorname{diag}=0.
$$

**PSD loss (log-spectral distance, Welch) — active only at full rollout.** The PSD stays a
batch-of-snippets estimate over the **full** horizon window: per channel $c$, all $B$ windows are
concatenated into one length-$BN$ series and Welch's method is applied with `nperseg = N`,
`noverlap = 0` (each horizon-length window becomes one segment; averaging $B$ periodograms gives a
stable spectrum with $\lfloor N/2\rfloor + 1$ frequency bins). A meaningful spectrum needs the whole
window, so the term is **gated on only once $L = N$** (via the last mask entry $s_{N-1}$):

$$
\mathcal{L}_\text{PSD} = s_{N-1}\cdot\frac{1}{C\,F}\sum_{c,f}\Big(\log\big(\hat{P}_{c,f} + \varepsilon\big)
- \log\big(P_{c,f} + \varepsilon\big)\Big)^2, \qquad \varepsilon = 10^{-8}.
$$

### 6.3 Total loss with scale-free normalisation

$$
\boxed{\;\mathcal{L} = \mathcal{L}_\text{MSE}
\;+\; w_\text{psd}\,\frac{\mathcal{L}_\text{PSD}}{\operatorname{sg}(\mathcal{L}_\text{PSD} + \varepsilon)}
\;+\; w_\text{fc}\,\frac{\mathcal{L}_\text{FC}}{\operatorname{sg}(\mathcal{L}_\text{FC} + \varepsilon)}\;}
$$

where $\operatorname{sg}(\cdot) = $ `jax.lax.stop_gradient`. Dividing each auxiliary term by its own
**detached** value makes its *value* ≈ $w$ (scale-free) while its *gradient* becomes a relative one,
$\nabla \big(\mathcal{L}_\text{aux}/\operatorname{sg}(\mathcal{L}_\text{aux})\big) =
\nabla \mathcal{L}_\text{aux} / \mathcal{L}_\text{aux}$. This keeps the raw MSE the dominant driver of
point accuracy regardless of the absolute magnitudes of the spectral/FC errors; $w_\text{psd}$,
$w_\text{fc}$ then act as pure relative weights. The raw (unweighted) `mse`, `psd`, `fc` values are
returned as `aux` for logging.

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
unlocks the PSD/FC terms, which *are* genuinely different in channel space (in latent space FC is
degenerate: PCA decorrelates the components, so their correlation matrix is $\approx I$). The two
mean-reductions divide the same summed error by $B\!N\!C$ vs $B\!N\!k$, so the gradients are parallel
with ratio $k/C$ — pinned by the regression test in
[`test_nn_losses.py`](../tests/test_nn_losses.py).

---

## 7. Optimisation

### 7.1 Optimizer

`optax.adamw(learning_rate, weight_decay)` — Adam with **decoupled** weight decay. Per parameter
$\theta$, with $g_t = \nabla_\theta \mathcal{L}$, defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$,
$\hat\varepsilon = 10^{-8}$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, &
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\\
\hat m_t &= \tfrac{m_t}{1-\beta_1^t}, &
\hat v_t &= \tfrac{v_t}{1-\beta_2^t},\\
\theta_{t+1} &= \theta_t - \eta\Big(\tfrac{\hat m_t}{\sqrt{\hat v_t}+\hat\varepsilon} + \lambda\,\theta_t\Big).
\end{aligned}
$$

with learning rate $\eta = \texttt{learning\_rate}$ and decoupled decay $\lambda = \texttt{weight\_decay}$.
The learning rate is **constant** (no schedule). The optimizer state is initialised over the array
leaves only (`eqx.filter(model, eqx.is_array)`). A step is one JIT-compiled
`eqx.filter_value_and_grad(compute_loss, has_aux=True)` + `optimizer.update` + `eqx.apply_updates`
([`step`](../src/neuro/nn_training.py)).

### 7.2 Batching

[`get_dataloaders`](../src/neuro/nn_training.py) reshuffles the training sample indices **every epoch**
with a fresh `np.random.default_rng()` (non-seeded → run-to-run stochastic ordering), then yields
contiguous slices of `batch_size`:

$$
\text{#batches} = \Big\lceil \tfrac{M_\text{train}}{\texttt{batch\_size}} \Big\rceil,
$$

the final batch being smaller when $M_\text{train}$ is not a multiple of the batch size. Each batch is
materialised as `jnp` arrays on the fly. Validation is evaluated on the **whole** validation set in a
single call (not batched).

### 7.3 Curriculum schedule

Let $e_0 = \texttt{curriculum\_start\_epoch}$ and $e_1 = \texttt{curriculum\_end\_epoch}$. The trusted
rollout length $L$ holds at 1 (teacher forcing) until $e_0$, ramps linearly $1 \to N$ between $e_0$
and $e_1$, and holds at the horizon $N$ afterwards:

$$
L(e) = \operatorname{clip}\!\Big(\operatorname{round}\big(1 + (N - 1)\cdot\operatorname{clip}(\tfrac{e - e_0}{\max(e_1 - e_0,\,1)},\,0,\,1)\big),\ 1,\ N\Big).
$$

The per-epoch step mask is the prefix of $L(e)$ ones
([`curriculum_state`](../src/neuro/nn_training.py)). So epochs before $e_0$ train on pure teacher
forcing, epochs in $[e_0, e_1]$ grow the rollout $1 \to N$, and the remainder trains on the full
$N$-step objective (with the PSD term active).

### 7.4 Training loop, validation, early stopping

[`train_model`](../src/neuro/nn_training.py) runs the loop:

1. For each epoch $e$: compute the step mask $L(e)$; iterate shuffled batches, take an optimizer `step`
   on each, accumulate the batch losses; the epoch train loss is the mean over batches.
2. **Validation loss** = `compute_loss` on the full validation set with the full-horizon mask ($L = N$)
   — i.e. always the pure $N$-step rollout error (plus the active PSD/FC terms), the quantity we
   ultimately minimise, comparable across epochs and rollout lengths.
3. **NaN guard**: if either the train or val loss is NaN, raise `ValueError("Loss is NaN…")`
   (the Optuna sweep catches this and prunes the trial).
4. **Best-model tracking**: keep the model with the lowest validation loss so far.
5. **Early stopping**: stop once `patience` consecutive epochs pass with no validation improvement.

The function returns the **best** (lowest-val-loss) model plus the per-epoch train/val loss histories.

---

## 8. Evaluation & artifacts

[`evaluate_model`](../src/neuro/nn_training.py) runs the best model on the scaled validation inputs,
**inverse-scales** the predictions back to raw EEG units, and reports

$$
\text{MSE} = \frac{1}{M_\text{val} \cdot NC}\sum \big(Y_\text{val} - \hat{Y}_\text{val}^{\text{unscaled}}\big)^2,
$$

against the raw (unscaled) targets. When a latent projection is active, both prediction and target are
first decoded to raw EEG channels ($\hat{y} = z E + \mu$) so the MSE is in the raw EEG space and
comparable across `latent_dim`. This MSE is the scalar returned by
[`train_and_save_predictor`](../src/neuro/nn_training.py) and used as the **Optuna objective**
(minimised) in the sweep.

[`train_and_save_predictor`](../src/neuro/nn_training.py) writes into the artifact directory:

| file                   | contents                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `model.eqx`            | serialised Equinox leaves of the `AutoregressivePredictor`                                |
| `model.json`           | architecture metadata (`in/out size`, `hidden_size`, `depth`, `activation`, `n_y/n_u/horizon`, `n_channels/n_controls`, `dt`, `downsample`, projection flags) |
| `model.scalers.npz`    | `u_mean/u_scale/y_mean/y_scale` (+ `latent_basis/latent_mean` if projected)               |
| `loss_curve.png`       | train vs. val scaled-MSE per epoch                                                        |
| `training_stats.json`  | `{train_loss[], val_loss[], mse}`                                                         |
| `comparison.png`       | $N$-step-ahead prediction vs. truth on up to 200 validation windows, 4 channels          |

These three model files (`.eqx` / `.json` / `.scalers.npz`) are exactly what
[`MLPArtifact.load`](../src/neuro/prediction.py) reads back for inference (and for the CasADi MPC
port). The run script additionally copies the resolved config YAML into the artifact directory for
provenance.

---

## 9. Configuration reference

All fields are validated strictly by pydantic ([`config.py`](../src/neuro/config.py)); unknown keys or
out-of-range values raise `ValidationError` rather than silently defaulting.

### `simulation`

| key           | meaning                                   |
| ------------- | ----------------------------------------- |
| `dt`          | raw simulation step (s)                   |
| `downsample`  | decimation factor $d$                     |
| `n_steps`     | downsampled steps loaded per trajectory   |
| `data_path`   | directory of `.npz` trajectories          |

### `model`

| key           | meaning                                                     |
| ------------- | ----------------------------------------------------------- |
| `n_y`         | past EEG steps in history                                   |
| `n_u`         | past control steps in history                               |
| `horizon` $N$ | direct prediction horizon                                   |
| `hidden_size` | MLP width (ignored when `depth = 0`)                        |
| `depth`       | hidden layers; `0` ⇒ linear model                           |
| `activation`  | `relu` / `tanh` / `softplus`                                |
| `latent_dim`  | PCA latent components $k$; `null` ⇒ no projection           |

### `training`

| key                          | meaning                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `epochs`                     | max epochs                                                    |
| `batch_size`                 | SGD batch size                                                |
| `learning_rate` $\eta$       | AdamW constant LR                                             |
| `weight_decay` $\lambda$     | AdamW decoupled decay                                         |
| `train_split`                | fraction used for training (tail held out for val)           |
| `curriculum_start_epoch`     | epoch the rollout length starts growing ($L = 1$ before it)   |
| `curriculum_end_epoch`       | epoch the rollout length reaches the horizon $N$ (held after) |
| `seed`                       | PRNG seed for weight init ( `+ seed_offset` in sweeps)        |
| `w_psd`, `w_fc`              | relative weights of the auxiliary losses                     |
| `patience`                   | early-stopping patience (epochs)                             |
| `scaler`                     | `standard` / `robust`                                         |
| `global_scaling`             | one shared scalar vs. per-channel scaling                    |

### Shipped presets

| preset                                                              | `depth` | `n_u` | `lr`      | `wd`     | `batch` | `curr.` | `w_psd` | `w_fc` |
| ------------------------------------------------------------------- | :-----: | :---: | --------- | -------- | :-----: | :-----: | :-----: | :----: |
| [`linear_best.yaml`](../configs/nn_predictor/linear_best.yaml)      | 0 (linear) | 7  | $10^{-3}$ | $10^{-3}$ | 256   | 0.7     | 0       | 0      |
| [`nonlinear_best.yaml`](../configs/nn_predictor/nonlinear_best.yaml)| 1       | 10    | $10^{-5}$ | $5\!\cdot\!10^{-4}$ | 128 | 0.6 | 0     | 0      |
| [`stats_experiment.yaml`](../configs/nn_predictor/stats_experiment.yaml)| 1   | 10    | $10^{-5}$ | $5\!\cdot\!10^{-4}$ | 128 | 0.6 | 0.1   | 0.1    |

Common to all three: `n_y=15`, `horizon=20`, `hidden_size=128`, `activation=softplus`,
`latent_dim=null`, `epochs=250`, `train_split=0.8`, `seed=69`, `patience=100`, `scaler=robust`,
`global_scaling=true`, and the 100 Hz / 5 s data described in Section 2.2. `stats_experiment.yaml`
differs from `nonlinear_best.yaml` only by turning on the PSD and FC auxiliary losses.

---

## 10. Hyperparameter sweeps

[`scripts/sweep_nn_predictor.py`](../scripts/sweep_nn_predictor.py) wraps the same
`train_and_save_predictor` in an **Optuna** study (`direction="minimize"`, objective = raw-units
validation MSE, persisted to a SQLite study DB). A `sweep` config section declares per-field search
spaces (`categorical` / `int` / `float` / `loguniform`); each trial samples overrides, re-validates
the merged `model`/`training` sections, uses `seed_offset = trial.number` to decorrelate otherwise
identical trials, and saves a full artifact under `trial_<n>/`. A `ValueError` mentioning NaN is
converted into `optuna.TrialPruned`.

---

## 11. Notes & gotchas

- **Units.** EEG is in arbitrary units (see the project *uncalibrated units* note), so absolute MSE
  values are not physically meaningful; training happens in standardised space, and downstream
  comparison metrics ([`prediction.py`](../src/neuro/prediction.py): NRMSE, Pearson, FC-pattern
  correlation) are deliberately scale-invariant.
- **Window overlap.** Sliding windows overlap heavily; the chronological train/val split (no shuffle
  *before* splitting) is what prevents near-duplicate leakage. Batches *within* the training set are
  still shuffled each epoch.
- **`depth = 0` is a genuinely linear model** — no activation is applied, so `activation` and
  `hidden_size` are inert for `linear_best.yaml`.
- **Auxiliary losses need `horizon > 1`** and are computed in EEG channel space, so they now work
  **with** `latent_dim` too (the latent rollout is decoded first; §3.1, §6). Only for `horizon = 1`
  are PSD/FC skipped.
- **Non-determinism.** The per-epoch dataloader shuffle uses an unseeded RNG, so runs are not
  bit-reproducible even at fixed `seed` (which only fixes weight initialisation).
