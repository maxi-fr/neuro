# Spectrogram losses for predictor training

Why the predictor's Welch PSD term should become a time-resolved STFT term, what that costs in
estimator quality, and what happened when we ran the deciding measurement.

The short version. The measurement worked and picked `n_segment = 25` and `n_hop = 12`. But the predictor
trained on that geometry suppresses seizures worse than the Welch endpoint it replaced. Seizure burden
jumped from 0.052 to 0.217, only 1 of 3 seeds was suppressed (down from 3 of 3), and delivered charge
was 3.4× higher. The live config is back on Welch. §7 records what the measurement established, while §8
records the measurement and the closed-loop result that overruled it.

Source of truth:

- The implemented STFT loss: [`StftLoss`, `spectrogram`](../src/neuro/predictor/losses.py)
- The measurement that chose the geometry: [`probe_stft_geometry.py`](../scripts/probe_stft_geometry.py)
- Windowed power loss already in place: [`EegMsLoss`](../src/neuro/predictor/losses.py)
- MPC spectral stage cost: [`_spectral_hinge_cost`](../src/neuro/control/nlp.py)
- Both objectives dimension by dimension: [`spectral_objectives.md`](spectral_objectives.md)
- Analysis-side periodograms and the healthy envelope: [`src/neuro/spectral.py`](../src/neuro/spectral.py)
- Envelope construction: [`scripts/build_healthy_psd.py`](../scripts/build_healthy_psd.py)
- Live configs: [`nonlinear_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_mse02_psd.yaml),
  [`mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml)

---

## 0. Status

The instrument is built, the experiment ran, and its answer did not survive contact with the
closed loop.

| item | status |
| :--- | :--- |
| §2.1 batch pooling inside the log | Fixed. `StftLoss` scores each rollout separately |
| §2.2 no time resolution inside the span | Available and measured, but not used. The live config is back on one frame |
| §5.1 four knobs | `n_segment`, `n_hop`, frame kernel and `n_bin_pool` are configurable; tapers are not |
| §5.2 frame kernel | Implemented, and measured to hurt discriminability. §8.3 ranks every kernel width below width 1 |
| §5.3 preference weights | Not implemented, and §8 found nothing to prefer (§7 question 7) |
| §5.4 multitaper | Not implemented; the one untested knob |
| §6 geometry in samples | Enforced. `StftSpec` takes `n_span`, `n_segment`, `n_hop`; seconds are not accepted |
| open question 6 (`_EPS` vs `LOG_FLOOR`) | Settled. One constant, `LOG_FLOOR`, in raw units on both sides |
| detrend / DC across the two objectives | Unified. No detrend, DC dropped on both sides; envelope rebuilt, 3-12 Hz unchanged |
| §7 open questions 1-5, 7 | Answered as offline questions by §8.3; question 3 is reopened by §8.5 |
| §8 the deciding measurement | Run, and overruled. See §8.5 |
| the live geometry | Welch endpoint, `n_segment = n_hop = n_span = 50`, unchanged from before this note |

`PSDLoss` and `welch_psd` no longer exist. Welch is now the `n_segment = n_span` endpoint of
`StftLoss`. It is the endpoint §8.3 ranks last in the spectrogram family and the one §8.5 keeps.

---

## 1. What actually changes

Welch is not a different representation from a spectrogram. It is a spectrogram followed by a
mean over the frame axis. The last line of `welch_psd` is `.mean(dim=-2)` over exactly the frames
an STFT returns:

```text
periodograms P(m, c, f)  --mean over m-->  Welch PSD  P(c, f)
                         \--no reduction-->  spectrogram
```

This is not a proposal to estimate anything more accurately. The same numbers come out of the
same FFTs. What changes is that Welch hard-couples three decisions into one call: segmentation,
overlap, and which axis gets averaged. The axis it averages is the time axis, which is the one
we want to keep. An STFT decouples them.

That is the whole argument. It does not need a non-stationarity premise, and we should not
justify it with one. The claim that time-resolving the spectral term improves the predictor was a
hypothesis to test in §7, not an established fact.

## 2. Two defects in the current loss

### 2.1 Batch pooling puts the mean inside the log

`PSDLoss.__call__` reshapes `(B, T, C)` into `(C, B*T)` and calls `welch_psd` with
`nperseg = span_steps`. The batch dimension therefore becomes Welch's segment axis, so the loss
computes

$$\log \mathbb{E}_b[\hat{P}] - \log \mathbb{E}_b[P]$$

An over-predicted trajectory can offset an under-predicted one before the log is ever taken. The
model can reach a low loss while being wrong on every individual rollout.

This is a property of the pooling, not of Welch. In `StftLoss`, we keep the batch axis
separate all the way to the square, and pool it only afterwards.

### 2.2 There is no time resolution inside the span

With `span_s: 1.0` at `fs = 50 Hz`, `span_steps = 50` and `nperseg = 50`, so each trajectory
contributes exactly one periodogram. The `K` in Welch's segment average is the batch size, not a
time index. Within a rollout the spectral term sees a single phase-blind magnitude spectrum over the
full second. That is still the geometry the live config runs. §8.3 measured it against 17
alternatives and it came last in the spectrogram family on offline separation. Then §8.5 trained its best
challenger and found it controls worse.

### 2.3 What is not a defect

`curriculum_mse` runs at weight 1.0 alongside it and is fully time-resolved. Timing error already
has a gradient path. The open question is whether time-resolving the spectral term adds anything
on top of that, not whether the composite objective is blind to timing. It is not.

## 3. What the loop already has

Two pieces of this already exist and should not be re-proposed:

- `_spectral_hinge_cost` is already a sliding-window periodogram cost. It reduces with a mean over
  `(window, channel, bin)`, never over windows alone, so a hot sub-window cannot be cancelled by a
  cold one, and `w_psd` stays independent of the window count.
- `EegMsLoss` is already hopped, per-sample, per-window, log-space, and in raw units. It is a
  spectrogram loss with a single frequency bin.

The open question was narrow: does resolving frequency inside the window beat
broadband `eeg_ms`, and does per-window scoring beat pooled Welch? Any comparison that omits `eeg_ms`
measures against a strawman. §8 ran both comparisons. Both answers are yes, the first by a wide
margin.

## 4. The cost of removing averaging

Dropping the mean over frames changes the estimand from $\log \mathbb{E}[P]$ to
$\mathbb{E}[\log P]$. For a periodogram cell, which is $\chi^2_2$ around the underlying spectrum,
those differ by Euler-Mascheroni. Averaging $K$ independent cells before the log gives bias
$\psi(K) - \ln K$ and variance $\psi'(K)$:

| avg factor `K` | bias (nats) | bias (dB) | std (nats) |
| :--- | ---: | ---: | ---: |
| 1 | -0.577 | -2.51 | 1.28 |
| 2 | -0.270 | -1.17 | 0.80 |
| 4 | -0.130 | -0.57 | 0.53 |
| 8 | -0.064 | -0.28 | 0.36 |

Consequences:

- **A naive per-cell loss teaches the geometric mean spectrum.** This lands about 56% of true mean power,
  or 2.5 dB low.
- **The bias is state-dependent.** High-SNR bins (a strong oscillation) are near-deterministic and
  barely biased. Background bins are fully $\chi^2_2$ and get pushed down. The net effect is an inflated
  peak-to-background contrast. That matters because the controller thresholds absolute power against
  a fixed envelope.
- **There is an irreducible floor.** The predictor is deterministic and the target is one noise
  realisation, so even a perfect model cannot match the realised periodogram fluctuations. The loss
  will not approach zero, and its floor depends on the geometry, which is why we must compare geometries
  by discriminability rather than loss value (§7).

This is a training-loss concern only. Inside the MPC the rollout is deterministic given
`u` and the reference is a fixed 0.90-quantile envelope, so there is no $\chi^2$ noise in
`_spectral_hinge_cost` to average away. Variance-reduction machinery does not belong on the
controller side.

## 5. Which axes are safe to pool

One rule covers every axis: pool power across cells whose underlying spectrum is smooth relative to
the pooling width, pool the squared residual freely after the log, and never pool the signed log
residual. The last point restates §2.1 generally, because a signed residual pooled over anything lets a
positive error cancel a negative one on any axis.

| axis | pre-log (power) | note |
| :--- | :--- | :--- |
| batch `b` | no | Distinct trajectories are not draws of one spectrum (§2.1) |
| channels `c` | no | 62 channels with genuinely different spectra |
| frames `m` | yes, bounded | Up to the envelope's correlation width (§5.2) |
| frequency bins `f` | yes, bounded | Up to the spectrum's smoothness in `f`; costs `Δf` |
| tapers (DPSS) | yes | Near-independent estimates of the same cell |

Pooling the loss over the batch after the log and square is always safe and always helps, because it
reduces gradient noise without touching the estimand. §2.1 and §4 are therefore not in tension. Moving
the mean outside the nonlinearity serves both.

### 5.1 Pooling is unavoidable

A periodogram is an inconsistent estimator. Each cell carries 2 degrees of freedom no matter how long
the window is. Lengthening `W` buys frequency resolution and buys no variance reduction. No geometry
escapes the §4 table. The only question is which axis pays for the degrees of freedom. Welch pays with
the entire time axis, which is the choice this note rejects. Four knobs produce four distinct trades:

| knob | `Δf` | effective time resolution | variance |
| :--- | :--- | :--- | :--- |
| lengthen `W` | improves | worsens | unchanged |
| pool `n` frames | unchanged | worsens by ~`n·H` | `/n_eff` |
| pool `n` bins | worsens by `n` | unchanged | `/n` |
| `K` tapers | worsens by `2·NW` | unchanged | `/K` |

Pooling frames is therefore not the same operation as lengthening the window, and the two should
not be conflated.

### 5.2 The frame axis is a filter, not a switch

Welch and a raw spectrogram are the endpoints of one continuum: smooth along `m` with a kernel,
then decimate. Welch is a full-span boxcar decimated to a single frame; consecutive Welch blocks are
a boxcar of width `b` at stride `b`; a raw spectrogram is the identity kernel. Nothing forces an
endpoint.

Extracting frames densely (small `H`) and pooling them with a kernel is strictly more flexible than
blocking, at the same FFT cost. Blocking couples extraction to pooling, plants hard boundaries where
a burst can be split across two blocks, and discards overlap that is free.

What sets the kernel width is an empirical quantity: the correlation width of the log-power
trajectory along `m`. Pooling narrower than that is free, because it buys degrees of freedom at no
cost in signal: the envelope genuinely does not move within the kernel. Pooling wider destroys what
the loss exists to measure. §8 measured this. Past the correlation created by window overlap alone,
only about two frames of correlation are real. That is why every kernel width we tested lost.

Two constraints on the kernel:

- **Power or squared residual only.** Smoothing the signed log residual along `m` reintroduces
  cancellation on the time axis instead of the batch axis.
- **Centred kernels are fine wherever the trajectory is held in full.** A training loss sees the whole
  rollout offline, and the MPC sees its whole predicted horizon before it evaluates the cost, so a
  symmetric kernel is legitimate in both. Causality binds only on reductions over measured signal
  consumed as it arrives, including the metrics layer, the seizure-threshold window, and the `eeg_ms`
  observable, where samples after `t` do not yet exist. `EegMsLoss` uses that trailing geometry to
  mirror its observable, not because the loss itself needs it.

The kernel must also keep `M_out > 1`. A weighted mean that still collapses the frame axis to a
single output is not a midpoint between Welch and a spectrogram. It is Welch with fewer effective
degrees of freedom, since `Var[Σ w_m P_m] = σ² Σ w_m²` is minimised by uniform weights. Weighting only earns
anything in the sliding form.

Kernel shape is a separate knob from the four in §5.1. At fixed `W` and fixed `Δf`, it trades
localisation against effective degrees of freedom, `K_eff = (Σw)² / Σw²`.

| kernel | `K_eff / n` | character |
| :--- | ---: | :--- |
| boxcar | 1.00 | maximum dof, weakest localisation |
| triangular | 0.75 | mild |
| Hann | 0.67 | width `n` carries the dof of a boxcar of width `2n/3` |
| Gaussian (`σ = n/6`) | 0.59 | tightest localisation, most expensive |
| EMA (`α`) | `(2-α)/α` frames | one parameter, causal, controller-computable |

Two cautions before reaching for a taper. There is no leakage to fight. The kernel is
non-negative and the smoothed quantity is positive power, so the usual reason to taper an analysis
window does not transfer; a taper here earns its place only through localisation. And once frames
overlap (`H < W`) they are correlated, at which point the variance-optimal weights are edge-heavy,
not centre-heavy. Localisation and variance then pull in opposite directions, and which wins
depends on the correlation width measured in §8.

### 5.3 Estimator weights versus preference weights

Two reasons to weight frames, with different homes:

- **Estimator quality.** Applied pre-log inside the §5.2 kernel. This changes what is estimated and pays
  in bias and degrees of freedom.
- **Preference.** Applied post-square as `Σ_m v_m D(m, c, f)²` for statements like "late-horizon frames matter more" or "frames near onset matter more". This is free by the §5 rule, incurring no degrees of freedom cost or bias while leaving the estimand untouched.

Any weighting motivated by importance rather than by variance belongs in the second form. Putting it
in the first silently changes the estimand to buy something the second gives away.

### 5.4 Multitaper

Multitaper is the principled version of frequency pooling. `K` orthogonal Slepian tapers on the same
window give near-independent estimates with no time smearing, from fixed `scipy.signal.windows.dpss`
weights, so it is `K` precomputed weighted FFTs and trivially differentiable. This is not free either.
Time-bandwidth `NW` costs a resolution bandwidth of `2·NW·Δf` for `K ≤ 2·NW − 1` tapers, and the band of
interest is only 9 Hz wide.

## 6. Constraints that bound the geometry

These are measured facts about the loop, not choices:

- `fs = 50 Hz` (`dt = 1e-4` plant, `downsample: 200`), so Nyquist is 25 Hz. Every geometry must
  be stated in samples, not seconds. For example, 0.1 s is five samples here.
- The band the metrics layer commits to is `SEIZURE_BAND_HZ = (3.0, 12.0)`.
- `Δf = 1/W`, `n_bins = W/2 + 1`, `M = ⌊(span − W)/H⌋ + 1`.
- Span budget. An earlier rollout probe found the `mse02_psd` predictor holds power out to ~75 steps
  (1.5 s). That caps the loss span until re-measured.
- The MPC's DFT is an explicit dense matmul per window in CasADi (no FFT available), so controller
  cost scales with `n_windows × n_bins`. Training geometry and controller geometry need not be
  identical, but any mismatch has to be a deliberate decision (§7).

A 75-sample span carries at most ~75 real degrees of freedom, whatever tiling is imposed on it.
Frames `M`, in-band bins, and estimator degrees of freedom `K` all draw from that one pool:

| `W` (samples) | `Δf` (Hz) | in-band bins | `H` | `M` |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1.00 | 10 | 25 | 2 |
| 50 | 1.00 | 10 | 12 | 3 |
| 37 | 1.35 | 6 | 19 | 3 |
| 25 | 2.00 | 5 | 12 | 5 |
| 15 | 3.33 | 3 | 15 | 5 |

`M` overstates the independent frame count whenever `H < W`, since overlapping frames share samples.
The table is the shape of the problem, not a recommendation.

## 7. What the measurement answered

Every answer here is about offline separation, namely how well a geometry tells a wrong spectrum from a
right one, which is what §8 set out to measure. Numbers are the `AUC` column of §8.3 unless stated
otherwise. §8.5 is the reason that is not the same question as "which geometry trains a better
controller", and question 3 is the one it overturns.

1. **Does frequency resolution inside the window beat broadband `eeg_ms` at all?** Yes, and by more
   than anything else in the table. `eeg_ms` scores 0.60; every spectrogram geometry scores 0.79 or
   better. A perfect model would score `eeg_ms` at 3.30 and the `mse02_psd` predictor scores it at
   3.48, so at a 1 s span broadband power sits on its own floor and carries almost no gradient.
   That is the one comparison this note insisted on (§3), and it is not close.
2. **Does per-window scoring beat pooled Welch once §2.1 is fixed?** Yes. Holding `W = 25` and
   `H = 6` fixed and varying only the frame kernel, 5 unpooled frames score 0.881, a 2-frame boxcar
   0.872, a 3-frame boxcar 0.860. Pooling the frame axis costs monotonically, and Welch is that
   kernel taken to the end of its range.
3. **What `(W, H, span)`?** Reopened by §8.5. Offline, `W = 25`, `H = 12`, span 50: `W = 25`
   beats `W = 15` (0.869), `W = 37` (0.841) and `W = 50` (0.822), so 2 Hz resolution looks like the
   sweet spot of the §6 trade. Overlap earns little (0.885 against 0.879 for the non-overlapping
   `H = 25`) and span 75 ties span 50. Trained and put in the loop, that geometry lost to Welch by
   4× on seizure burden, so this question has an offline answer and no closed-loop answer.
4. **Which pooling axes and widths?** Bin pooling yes, frame pooling no. Pooling 2 bins ranks first
   at 0.888, inside the noise of the unpooled 0.884, while cutting the floor from 4.66 to 3.16,
   exactly the trade §5.1 predicts. The live config still leaves it at 1. Setting it higher would push
   `Δf` from 2 Hz to 4 Hz across a 9 Hz band, which blurs the theta peak the controller thresholds, and
   the ranking gain is not real. Frame kernels lose (question 2). Multitaper is the one knob nothing
   has measured.
5. **Must training geometry match the envelope's?** No, and matching is expensive. The envelope is
   built at `W = 50`, `H = 25`, and every `W = 50` geometry lands in the bottom third: 0.822 and
   0.815 against 0.885 for the best challenger. Alignment with the controller costs about 0.07 of
   separation, and buys nothing the offline measurement can see. §8.5 is a reason to hold that
   answer loosely, because it relies on the same style of offline inference that the closed loop just contradicted.
6. **Standardised units versus raw units with LOG_FLOOR.** Settled. Every training loss now scores raw
   units and floors with `LOG_FLOOR`, the constant the MPC hinge uses.
7. **Two-sided, or weighted toward the hinge-active regime?** Two-sided. Split by whether the true
   power exceeds the envelope, 51.6% of cells are hinge-active, and they carry a smaller squared
   residual than the inactive ones: 30.6 against 35.8 for the predictor, 4.48 against 5.02 for the
   floor. The error is not concentrated where the hinge bites. Band-limiting to 3-12 Hz is the
   strongest version of that preference and it costs 0.07 (0.884 to 0.814).

## 8. The deciding measurement

[`probe_stft_geometry.py`](../scripts/probe_stft_geometry.py) scores 18 candidate geometries with
no training runs, on existing trajectories only. Reproduce with:

```bash
uv run python scripts/probe_stft_geometry.py \
  --artifact artifacts/nonlinear_mse02_psd/model \
  --data data/experiment_excited_roast/test \
  --ensemble data/predictability_ensemble
```

### 8.1 Where the pairs come from

- **Floor.** Evaluated as `L(true_A, true_B)` over the branch ensemble's children. Two children of one
  parent share the snapshot and held current, differing only in noise seed. This represents what a
  perfect deterministic model would score. Only the four seizure branches are used, as the healthy
  branch runs a different `A` vector than the predictor's training plant.
- **Signal.** Evaluated as `L(pred, true)` with the existing `mse02_psd` artifact rolled out on the same
  children. Its rollout NMSE (0.6 to 1.2 over 75 steps) brackets what it scores on the in-distribution
  test set (0.55 to 0.75), confirming the ensemble does not score distribution shift.
- **Stranger.** Evaluated as `L(true_C, true_A)` with `C` from a different parent. This establishes the
  no-information baseline.

One caveat on the floor. The stored EEG carries no pre-branch history, so the predictor is primed on
each child's own first 15 samples and the earliest `t0` is 0.3 s after the branch, by which point
`A` and `B` have already diverged a little. The floor is therefore an upper bound. The bound is
tight. It reads 3.68 at `t0 = 0.3 s` and rises to 5.10 by 1.5 s, so extrapolating back to the branch
gives about 3.3, which matches `2 ψ'(1) = 3.29`, the theoretical variance of the difference of two
independent log-`χ²₂` cells that §4 predicts. Every candidate carries the same bias, so the ranking
is unaffected.

### 8.2 Frame-axis correlation width

Autocorrelation of `log P(m, c, f)` along `m` at hop 1, over 40 held-out trajectories, against a
phase-randomised surrogate with the same PSD and no time-varying envelope. The surrogate serves as
a control: at hop 1, consecutive frames share samples, so a correlation width appears even when
nothing in the envelope moves.

| `W` | measured (frames) | surrogate (frames) | measured (s) | surrogate (s) |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 6.5 | 3.9 | 0.129 | 0.077 |
| 25 | 8.1 | 5.9 | 0.161 | 0.118 |
| 37 | 10.4 | 8.2 | 0.209 | 0.165 |
| 50 | 12.6 | 10.5 | 0.252 | 0.210 |

Most of the width is the overlap artefact. The genuine envelope structure is the 2 to 3 frame gap,
about 40 to 50 ms at `W = 25`, which is under one hop of the chosen geometry. That is the whole
budget for the §5.2 kernel, and it explains why every kernel width measured below width 1.

### 8.3 Floor, signal and separation

2560 rollouts per cell, in nats². `d` is the criterion this note proposed,
`(E[signal] − E[floor]) / std`; `AUC` is the probability that a scored rollout ranks worse than a
floor pair. `AUC_str` is the same for the stranger.

| candidate | floor | signal | stranger | `d` | `AUC` | `AUC_str` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `W25 H12 s75 pool2` | 3.16 | 58.11 | 4.79 | 1.57 | **0.888** | 0.626 |
| `W25 H12 s50` | 4.45 | 43.68 | 6.10 | 1.63 | **0.885** | 0.633 |
| `W25 H12 s75` | 4.66 | 60.79 | 6.18 | 1.58 | 0.884 | 0.624 |
| `W25 H6 s75` | 4.67 | 61.35 | 6.20 | 1.58 | 0.881 | 0.625 |
| `W25 H25 s50` | 4.45 | 44.16 | 6.10 | 1.62 | 0.879 | 0.632 |
| `W25 H6 s75` boxcar 2 | 3.14 | 60.81 | 4.59 | 1.57 | 0.872 | 0.614 |
| `W15 H8 s75` | 4.75 | 52.74 | 6.65 | 1.53 | 0.869 | 0.641 |
| `W25 H6 s75` Hann 3 | 2.76 | 59.94 | 4.19 | 1.57 | 0.861 | 0.610 |
| `W25 H6 s75` boxcar 3 | 2.66 | 59.41 | 4.08 | 1.57 | 0.860 | 0.609 |
| `W37 H19 s75` | 4.64 | 61.97 | 5.99 | 1.58 | 0.841 | 0.615 |
| `W50 H12 s75` | 4.84 | 61.19 | 6.08 | 1.60 | 0.822 | 0.603 |
| `W50 H25 s75` | 4.84 | 59.67 | 6.08 | 1.59 | 0.815 | 0.606 |
| `W25 H12 s75` 3-12 Hz | 5.28 | 54.91 | 7.63 | 1.41 | 0.814 | 0.625 |
| Welch 50 | 4.64 | 32.47 | 5.99 | 1.60 | 0.793 | 0.610 |
| Welch 75 | 4.66 | 46.82 | 5.84 | 1.58 | 0.793 | 0.593 |
| Welch 50, 3-12 Hz | 4.96 | 26.21 | 7.17 | 1.43 | 0.785 | 0.612 |
| `eeg_ms` span 50 | 3.30 | 3.48 | 6.52 | 0.05 | 0.607 | 0.653 |
| `eeg_ms` span 75 | 3.68 | 3.89 | 6.70 | 0.05 | 0.595 | 0.647 |

The metric `d` does not rank these geometries, which was the smaller of the two things this note got wrong (§8.5 has the larger one). Every spectrogram geometry is admissible under `d`, after which the metric saturates. Because signal runs an order of magnitude above the floor, `d` collapses to `E[signal] / std[signal]`, reflecting the model's error distribution rather than estimator quality. All 16 land between 1.41 and 1.63. While `d` rejects `eeg_ms` at 0.05, distinguishing between surviving candidates requires the rank statistic, which resists heavy tails and spreads them across 0.79 to 0.89.

The floor column is the clean confirmation of §4 and §5.1. Pooling 2 bins takes it from 4.66 to
3.16, a 3-frame boxcar to 2.66, and an unpooled single cell sits just above the `χ²₂` value of 3.29.
The estimator theory holds exactly. It just does not predict which geometry trains better, because
lowering the floor by smoothing also removes the structure the loss is there to score.

### 8.4 What the offline measurement picked

`n_span = 50`, `n_segment = 25`, `n_hop = 12`, no frame kernel, no bin pooling, full band. Every
spectrogram geometry is admissible under the `d` criterion; this one has the best rank separation,
0.885 against 0.793 for the Welch endpoint it would replace.

### 8.5 What the closed loop said

Trained with exactly that geometry and nothing else changed (300 epochs,
[`nonlinear_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_mse02_psd.yaml)), then scored against
the Welch-endpoint predictor on
[`mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml) over 3 plant
seeds, 20 s each, via `evaluate_closed_loop_suppression`:

| | Welch 50 | `W25 H12 s50` |
| :--- | ---: | ---: |
| rollout NMSE | 0.498 | 0.501 |
| log-energy error | 0.565 | 0.842 |
| `‖∂rollout/∂u‖` | 10.76 | 2.65 |
| offline rank separation | 0.793 | **0.885** |
| seizure burden | **0.052** | 0.217 |
| seeds suppressed | **3 / 3** | 1 / 3 |
| mean delivered charge | **23.3** | 78.4 |

The geometry that separates spectra best trains the worse controller, by 4× on burden while
spending 3.4× the charge. Waveform NMSE is identical, so this is not a worse model in the usual
sense.

Two things in the table point at the same mechanism. The per-bin loss focuses on spectral shape while
sacrificing the broadband energy trajectory. Log-energy error rises from 0.565 to 0.842, which directly
degrades what the MPC integrates. Meanwhile `‖∂rollout/∂u‖` falls 4×, so the model believes stimulation
moves the EEG far less than it does. A controller optimising against a model with too little control
authority pushes harder for less effect, producing the 3.4× charge at 4× the burden. Whether the
sensitivity collapse causes the failure or merely accompanies it remains untested.

What this costs the note's method. §8 was built on the premise that a geometry which cannot separate a
predicted spectrum from a perfect one cannot be teaching anything, and that premise still holds, as seen
when it eliminates `eeg_ms`. What does not hold is the converse. Separation ranks how legible a wrong
spectrum is to the loss, and says nothing about which errors the model gives up to lower it. Between two
admissible geometries, offline separation provides no evidence about closed-loop control. Only the closed
loop does, and that requires a 3 h training run plus a 3 h sweep per candidate.

The live config is therefore unchanged at the Welch endpoint with `n_segment = n_hop = 50`. The trained
challenger is kept at `artifacts/nonlinear_mse02_psd_w25h12` so the comparison can be re-run without
retraining.

### 8.6 What would settle it

In rough order of cost:

1. **More seeds.** Three is enough to see 3/3 against 1/3 and a 4× burden gap, but not enough to size it precisely.
2. **Isolate the energy course.** Add `eeg_ms` alongside the time-resolved `stft` term and retrain.
   If restoring the broadband course recovers suppression, §8.5 is a weighting failure rather than a
   verdict on time resolution, and the two terms are complements rather than substitutes.
3. **Score `‖∂rollout/∂u‖` as a first-class objective.** It moved 4× on a loss change that nobody
   expected to touch it, and it is the one quantity here with a direct mechanical path to controller
   behaviour. It is already computed every run and currently only printed.

## 9. Explicitly not doing yet

**Multi-resolution (MRSTFT).** At `fs = 50 Hz`, the multi-scale ladder breaks down because a 0.1 s
window contains only 5 samples and 3 bins. It also adds one weight per scale on top of an existing
weighting problem, because the spectral cost is a mean while control effort is a sum. That discrepancy
is why `w_psd: 1000` against `w_u: 10` does not behave as expected. Revisit only if a single scale wins
and the residual failure is demonstrably scale-related. §8.5 makes that condition further off than
expected, as no single scale has yet beaten Welch in the closed loop.
