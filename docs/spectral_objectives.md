# Spectral Objectives: the `stft` Loss and the MPC Spectral Cost

The loop has two spectral objectives, and they are not the same object. One trains the predictor
offline against a recorded trajectory; the other is minimised online against a fixed healthy
envelope. They share a geometry vocabulary and a log floor, and differ in almost everything else.
This note takes both apart dimension by dimension. It expands §6.2 of
[`nn_predictor_training.md`](nn_predictor_training.md); for *why* the training term is time-resolved
at all and what remains unmeasured, see [`spectrogram_loss_guide.md`](spectrogram_loss_guide.md).

Source of truth:

- Training loss: [`StftLoss`, `spectrogram`, `pool_bins`, `frame_kernel`, `smooth_frames`](../src/neuro/predictor/losses.py)
- Loss config and its validation: [`StftSpec`](../src/neuro/config.py)
- MPC stage cost: [`_spectral_hinge_cost`](../src/neuro/control/nlp.py)
- Envelope type and analysis-side periodograms: [`src/neuro/spectral.py`](../src/neuro/spectral.py)
- Envelope construction: [`scripts/build_healthy_psd.py`](../scripts/build_healthy_psd.py)
- Geometry consistency checks: [`_check_psd_reference`](../src/neuro/validation.py)

---

## 1. Vocabulary and the rate

Five terms, as [`CONTEXT.md`](../CONTEXT.md) defines them:

- **Rollout** — one trajectory run forward under a fixed control sequence, uncorrected by
  measurement. The offline training rollout and the controller's predicted horizon are both rollouts.
- **Span** — the leading portion of a rollout one loss term scores. Only the training loss has one;
  the MPC cost scores the whole horizon.
- **Segment** — the fixed-length slice of a trajectory fed to one Fourier transform. Both objectives
  use segments; the word *window* is avoided because it already means three other things.
- **Frame** — the spectrum one segment produces, indexed $m$ by its position on the hop grid.
- **Frame Kernel** — the non-negative smoother applied to *power* along $m$, before the log. It buys
  estimator degrees of freedom, not frequency resolution. Both objectives reduce over a rollout held
  in full rather than over measured signal arriving sample by sample, so a *centred* kernel, which
  reads frames on both sides of its output, is legitimate in both.

The rate fixes everything else. The plant runs at $\Delta t = 10^{-4}$ s and is decimated by 200, so

$$f_s = 50\ \text{Hz}, \qquad \text{Nyquist} = 25\ \text{Hz}, \qquad
\Delta f = \frac{f_s}{W}, \qquad n_\text{bins} = \frac{W}{2} + 1,$$

with $W$ the segment length in samples. At $W = 50$ that is $\Delta f = 1$ Hz and 26 bins, of which
the band the metrics layer commits to — $(3, 12)$ Hz — occupies 10. This is why the training loss
takes its geometry in **samples**: at 50 Hz, 0.1 s is five samples, and a seconds-valued config
cannot address most of the useful grid.

---

## 2. The training loss (`stft`)

### 2.1 Parameters

The nine `StftSpec` fields, with the symbol each carries in the formulas below:

| field | symbol | type | default | meaning |
| :--- | :--- | :--- | :--- | :--- |
| `weight` | $\lambda$ | `float ≥ 0` | required | multiplier in the total loss |
| `start_epoch` | — | `int ≥ 0` | `0` | first epoch the term contributes a gradient |
| `n_span` | $S$ | `int > 0` | required | rollout steps scored, in samples |
| `n_segment` | $W$ | `int > 0` | required | segment length; $W = S$ is the Welch endpoint |
| `n_hop` | $H$ | `int > 0` | required | segment spacing; $H < W$ overlaps, $H = W$ tiles |
| `band_hz` | $[f_\text{lo}, f_\text{hi}]$ | tuple or `None` | `None` | frequency range kept; `None` keeps everything above DC |
| `n_bin_pool` | $K_f$ | `int ≥ 1` | `1` | bins averaged pre-log; `1` is no pooling |
| `kernel` | $w_m$ | `boxcar`, `triangular`, `hann` | `boxcar` | frame-kernel shape |
| `kernel_width` | $N_w$ | `int ≥ 1` | `1` | frames pooled; `1` is no smoothing |

Everything else follows from those nine and the rate:

| symbol | code | value |
| :--- | :--- | :--- |
| $\Delta f$ | — | $f_s / W$ |
| $M$ | `StftGeometry.n_segment_frames(S)` | $\lfloor (S - W)/H \rfloor + 1$ — frames the segment grid cuts |
| $F$ | `bin_hi - bin_lo` | in-band bins, DC already excluded |
| $F'$ | — | $\lfloor F / K_f \rfloor$ — bins left after pooling |
| $M_\text{out}$ | `StftGeometry.n_frames(S, fs)` | $M - N_w + 1$ — frames left after the valid-support convolution |
| $K_\text{eff}$ | `K_eff` | frames the kernel effectively pools, §2.4 |

`bin_range(fs)` turns `band_hz` into a half-open rfft index range. DC is dropped unconditionally —
there is no per-segment detrend to protect it, and a floored log of near-zero power is noise. The
band edges are inclusive in frequency: at $f_s = 50$, $W = 50$, `band_hz: [3.0, 12.0]` gives bins
3…12, i.e. `(3, 13)`.

**Rejected at config load**, not at runtime: $W > S$; a kernel wider than $M$; any kernel leaving
$M_\text{out} < 2$ (a kernel that collapses the frame axis is Welch with *fewer* effective dof,
since $\operatorname{Var}[\sum_m w_m P_m] = \sigma^2 \sum_m w_m^2$ is minimised by uniform
weights); a non-increasing `band_hz`; a band leaving no bins below Nyquist; $K_f$ larger than the
in-band bin count.

Every field is on a slider in
[`stft_loss_walkthrough.py`](../notebooks/stft_loss_walkthrough.py), which plots one channel of one
rollout through each step of §2.2.

### 2.2 Every dimension, in order

`StftLoss.__call__` receives `pred` and `true`, both `(B, n_span, C)` in **standardised** units, and
returns a scalar. Each is pushed through `log_spectrogram` independently:

| step | operation | shape after |
| :--- | :--- | :--- |
| 0 | input rollout, standardised | `(B, S, C)` |
| 1 | `ctx.to_raw`, truncate to `span_steps`, transpose | `(B, C, S)` |
| 2 | `unfold(n_segment, n_hop)` — cut into segments | `(B, C, M, W)` |
| 3 | periodic Hann taper, `rfft`, magnitude², density scaling, one-sided fold | `(B, C, M, W//2 + 1)` |
| 4 | drop DC, apply `band_hz` | `(B, C, M, F)` |
| 5 | `pool_bins(n_bin_pool)` — mean over bin groups | `(B, C, M, F')` |
| 6 | `smooth_frames(frame_kernel(...))` — valid-support convolution along $m$ | `(B, C, M_out, F')` |
| 7 | `log(· + LOG_FLOOR)` | `(B, C, M_out, F')` |
| 8 | `pred − true`, square | `(B, C, M_out, F')` |
| 9 | `torch.mean` over every axis | `()` |

Written out, with $b$ over the batch, $c$ over channels, $m$ over frames and $f$ over pooled bins:

$$\mathcal{L}_\text{STFT} = \frac{1}{B\,C\,M_\text{out}\,F'}\sum_{b,c,m,f} \Big(\log(\hat{P}\_{b,c,m,f} + \varepsilon) - \log(P_{b,c,m,f} + \varepsilon)\Big)^2, \qquad \varepsilon = \texttt{LOG\_FLOOR} = 10^{-8}.$$

### 2.3 Where the reductions sit, and why the order is fixed

Three reductions happen at three different places, and the placement is the whole design:

| reduction | axis | position | what it does |
| :--- | :--- | :--- | :--- |
| $K_f$ (`n_bin_pool`) | $f$ | **pre-log** | buys estimator dof, costs $\Delta f$ |
| $w_m$ (frame kernel) | $m$ | **pre-log** | buys estimator dof, costs time localisation |
| final `mean` | $b, c, m, f$ | **post-square** | reduces gradient noise, changes nothing about the estimand |

The rule they implement: *pool power across cells whose underlying spectrum is smooth relative to
the pooling width; pool the squared residual freely; never pool the signed log residual.* A signed
residual averaged over any axis lets an over-prediction cancel an under-prediction — which is
exactly what the superseded `PSDLoss` did on the batch axis, reaching a low loss while being wrong
on every rollout.

This is enforced structurally, not documented. There is no configurable step between the log and the
square, so no config can express the illegal pooling. The order of steps 5–9 is not a parameter.

Two axes are never pooled pre-log: the batch, because distinct trajectories are not draws of one
spectrum, and the channels, because 62 channels have genuinely different spectra.

### 2.4 What pre-log pooling costs

A periodogram cell is $\chi^2_2$ around the underlying spectrum however long the segment is —
lengthening $W$ buys resolution and *no* variance reduction. Taking the log of a single cell
therefore estimates $\mathbb{E}[\log P]$, not $\log \mathbb{E}[P]$, which is 0.577 nats (2.5 dB) low.
Averaging $K$ cells first cuts that to $\psi(K) - \ln K$. So pre-log pooling is the only thing that
moves the estimand toward mean power, and every unit of it is paid for in resolution — see §4 and
§5.1 of the guide for the table and the four-way trade.

The frame kernel's effective count is

$$K_\text{eff} = \frac{\left(\sum_m w_m\right)^2}{\sum_m w_m^2},$$

reported per step as `K_eff`. Endpoints are kept strictly positive, so a width-$N_w$ taper really
pools $N_w$ frames. Asymptotically $K_\text{eff}/N_w$ is 1.00 for `boxcar`, 0.75 for `triangular`,
0.67 for `hann`; at small widths the exact value differs (width 3 Hann is $[0.5, 1, 0.5]$, so
$K_\text{eff} = 8/3$, not 2). Boxcar maximises dof, Hann maximises localisation.

Note that overlapping frames ($H < W$) share samples, so $M$ overstates the independent frame count
and $K_\text{eff}$ overstates the dof actually bought.

### 2.5 Diagnostics

Two numbers per step, `M_out` and `K_eff` — the two quantities the bias/variance table is indexed
by. Everything else ($M$, $F$, $\Delta f$) is a deterministic function of the config and readable
from the YAML.

### 2.6 Worked example: the live config

[`nonlinear_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_mse02_psd.yaml) sets
`n_span: 50, n_segment: 50, n_hop: 50` at $f_s = 50$ Hz:

$$M = 1, \quad M_\text{out} = 1, \quad \Delta f = 1\ \text{Hz}, \quad F = 25\ \text{bins (DC dropped)}, \quad K_\text{eff} = 1.$$

One frame per rollout: Welch's geometry, scored per sample. It differs from the superseded
`PSDLoss` by the batch-pooling fix and nothing else. §8 of the guide measured a time-resolved
replacement (`n_segment: 25, n_hop: 12`) that separates a predicted spectrum from a perfect one
far better, trained it, and found it suppresses seizures worse. The Welch endpoint stays until
that gap is understood.

---

## 3. The MPC spectral cost

### 3.1 What it is

A one-sided hinge on how far the *predicted* spectrum rises above a fixed healthy envelope, added to
the stage cost when `w_psd > 0`:

$$J_\text{PSD} = \frac{1}{M\,C\,(n_\text{bins} - 1)} \sum_{m,c,f>0} \Big[\max\big(0,\; \log(P_{m,c,f} + \varepsilon) - \log P^\text{ref}_{c,f}\big)\Big]^2 .$$

One-sided is the point. The cost is exactly zero once the spectrum is under the envelope everywhere,
which leaves `w_u` alone to ask for the least stimulation that keeps it there. Any nonzero `w_y`
destroys that "done" point, because quadratic output power keeps pushing after the spectral
objective is satisfied.

### 3.2 Every dimension, in order

Everything here is symbolic CasADi built once at solver-construction time, not a tensor.

| step | operation | shape |
| :--- | :--- | :--- |
| 0 | predicted outputs over the horizon, `horzcat` of the rollout nodes | `(C, horizon)` |
| 1 | slice segment $m$: `[:, m*R : m*R + L]` | `(C, L)` |
| 2 | multiply by a periodic Hann taper — no detrend | `(C, L)` |
| 3 | two dense DFT matmuls (cos, sin), squared and summed, density-scaled and folded | `(C, n_bins)` |
| 4 | drop DC: `power[:, 1:]` scored against `log P_ref[:, 1:]` | `(C, n_bins - 1)` |
| 5 | `fmax(0, log(P + LOG_FLOOR) − log P_ref)`, squared | `(C, n_bins - 1)` |
| 6 | summed into the running total; repeat for each of the $M$ segments | scalar |
| 7 | divide by $M \cdot C \cdot (n_\text{bins} - 1)$ | scalar |

with $L$ and $R$ the envelope's own segment length and hop, and
$M = \lfloor (\text{horizon} - L)/R \rfloor + 1$. The build raises if `horizon < L`, if the envelope's
channel count differs from the model's, or (in `PsdEnvelope.load`) if the stored bin count does not
match $L/2 + 1$ — the envelope must carry every bin, so that dropping DC stays a decision the cost
makes explicitly at its use site rather than one an under-sized npz makes silently.

CasADi has no FFT, so step 3 is an explicit $(C, L) \times (L, n_\text{bins})$ product, twice, per
segment. Solver graph size therefore scales with $M \times n_\text{bins}$, which is the practical
limit on controller-side geometry.

### 3.3 Parameters

| field | type | default | meaning |
| :--- | :--- | :--- | :--- |
| `w_psd` | `float ≥ 0` | `0.0` | weight on the hinge; `0` disables it entirely |
| `psd_ref` | path or `None` | `None` | the envelope npz; required when `w_psd > 0` |
| `psd_window_s` | `float` or `None` | `None` | declared segment length, in seconds |
| `psd_hop_s` | `float` or `None` | `None` | declared segment spacing, in seconds |

The two `*_s` fields do **not** configure the cost. The envelope npz is the single source of truth
for $L$ and $R$; the YAML fields exist so the config states the geometry it expects, and
[`_check_psd_reference`](../src/neuro/validation.py) raises when the two disagree — as it does when
`controller.dt` does not match the envelope's $1/f_s$.

### 3.4 The reference envelope

[`build_healthy_psd.py`](../scripts/build_healthy_psd.py) runs healthy simulations, computes every
hopped periodogram over every trajectory with
[`compute_periodograms`](../src/neuro/spectral.py) (no detrend, periodic Hann, one-sided,
density-scaled — the training loss's convention), pools them across segments *and* trajectories, and takes a per-$(c, f)$ quantile —
0.90 by default. The stored npz carries `Pref`, `freqs`, `fs`, `L`, `R`, the quantile, the pooled
window count and a plant fingerprint. Defaults are `window_s = 1.0`, `hop_s = 0.5`, i.e. $L = 50$,
$R = 25$ at 50 Hz. The decimation is read from the simulation config rather than passed in, so the
envelope is always measured at the rate the loop runs the cost at. Every bin including DC is
stored: the cost declines to score DC, but the envelope stays a complete description of healthy
power for analysis code that wants it.

Because the reference is a fixed quantile of measured healthy power, it is deterministic — and the
predicted rollout is deterministic given $u$. There is no $\chi^2$ noise on either side of the hinge,
so none of §2.4's variance-reduction machinery belongs here.

### 3.5 Normalisation

The hinge is a **mean** over $(m, c, f)$, never a sum over segments, so a hot segment cannot be
cancelled by a cold one and `w_psd` stays independent of the segment count. The stagewise part of the
cost is likewise divided by the horizon. Both are means, so weights stay comparable when the horizon
or the envelope geometry changes.

---

## 4. Side by side

| | training `stft` loss | MPC spectral cost |
| :--- | :--- | :--- |
| reference | the true rollout, per sample | a fixed healthy quantile envelope |
| direction | two-sided (squared log ratio) | one-sided hinge (excess only) |
| zero at | perfect spectral match | anywhere under the envelope |
| units | raw EEG | raw EEG |
| log floor | `LOG_FLOOR` on both sides | `LOG_FLOOR` on the prediction; the envelope is strictly positive |
| per-segment detrend | **no** | **no** |
| DC bin | dropped | dropped (the envelope still stores it) |
| band | optional `band_hz` | all bins to Nyquist |
| pre-log pooling | bins and frames, configurable | none |
| reduction | mean over $(b, c, m, f)$ after the square | mean over $(m, c, f)$ after the square |
| geometry source | `StftSpec`, in samples | the envelope npz, in samples, declared in seconds |
| transform | `torch.fft.rfft` | dense DFT matmul (CasADi has no FFT) |
| differentiable w.r.t. | model parameters | the control sequence $u$ |
| evaluated | once per training batch | once per solver iteration, per control step |

The two agree exactly on the quantity being scored: raw units, no detrend, DC dropped, periodic
Hann, one-sided fold, density scaling, the same `LOG_FLOOR`, and a mean over every remaining axis
after the square. Neither pools a signed log residual. What the table still separates is the part
that is *meant* to differ — reference, direction, substrate — plus two things that remain open:
the geometry ($L$, $R$ against `n_segment`, `n_hop`) and the training side's optional pre-log
pooling and `band_hz`.

---

## 5. Known asymmetries and open items

1. **Detrend and DC — settled on the training side's terms.** The MPC used to detrend each segment
   and then score the resulting ~0 DC bin, while the training loss did neither and dropped DC. The
   training-side convention won: a per-segment detrend is a high-pass whose corner moves with the
   segment length, which would contaminate any segment-length sweep. `compute_periodograms` now
   passes `detrend=False`, `_spectral_hinge_cost` tapers the raw segment and slices bin 0 off both
   the power and the log reference, and `data/healthy_psd.npz` was rebuilt at the same geometry
   ($L = 50$, $R = 25$, $q = 0.90$, 1190 pooled windows). Only envelope bins 0 and 1 moved — a Hann
   taper spreads a removed constant over those two bins and no further — so the 3–12 Hz band is
   numerically identical and no tuned `w_psd` is invalidated. The envelope still stores all
   $L/2 + 1$ bins and `PsdEnvelope.load` still rejects one that does not.
2. **Training geometry and controller geometry need not match, and currently do not have to.** Whether
   they *should* is open question 5 of the guide. The envelope's own geometry ($L = 50$, $R = 25$) is
   equally open and can be re-derived.
3. **The weights are not comparable across objectives.** Both are means, so each is insensitive to
   its own geometry — but `w_psd: 1000` against `w_u: 10` still does not mean what it looks like,
   because the quantities they scale have different natural magnitudes.
4. **The predictor loss is symmetric; the controller's hinge is not.** A training loss that treats
   over- and under-prediction alike does not prioritise the bins that actually drive control. That is
   open question 7, and `band_hz` is only the crude version of the knob.
5. **Nothing has been trained on a time-resolved geometry.** The live config sits at the Welch
   endpoint. The measurement that should choose a geometry — correlation width along $m$, then loss
   floor versus signal, ranked by discriminability — is §8 of the guide and has not been run.
