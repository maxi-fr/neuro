# Spectrogram Losses for Predictor Training

Why the predictor's Welch PSD term should become a time-resolved STFT term, what that costs in
estimator quality, and what has to be measured before a window geometry can be chosen. The geometry
itself — window `W`, hop `H`, loss span, averaging scheme — is **open**; this note fixes the
constraints that bound it and the experiment that should decide it, not the values.

Source of truth:

- Current Welch loss and the differentiable Welch replica: [`PSDLoss`, `welch_psd`](../src/neuro/predictor/losses.py)
- Windowed power loss already in place: [`EegMsLoss`](../src/neuro/predictor/losses.py)
- MPC spectral stage cost: [`_spectral_hinge_cost`](../src/neuro/control.py)
- Analysis-side periodograms and the healthy envelope: [`src/neuro/spectral.py`](../src/neuro/spectral.py)
- Envelope construction: [`scripts/build_healthy_psd.py`](../scripts/build_healthy_psd.py)
- Live configs: [`nonlinear_mse02_psd.yaml`](../configs/nn_predictor/nonlinear_mse02_psd.yaml),
  [`mse02_psd_mpc_spectral.yaml`](../configs/simulation/mse02_psd_mpc_spectral.yaml)

---

## 1. What actually changes

Welch is not a different representation from a spectrogram. It *is* a spectrogram followed by a
mean over the frame axis — the last line of `welch_psd` is `.mean(dim=-2)` over exactly the frames
an STFT would return:

```text
periodograms P(m, c, f)  --mean over m-->  Welch PSD  P(c, f)
                         \--no reduction-->  spectrogram
```

So this is not a proposal to estimate anything more accurately. The same numbers come out of the
same FFTs. What changes is that Welch hard-couples three decisions into one call — segmentation,
overlap, and which axis gets averaged — and the axis it averages is the time axis, which is the one
we want to keep. An STFT decouples them.

**That is the whole argument.** It does not need a non-stationarity premise, and it should not be
justified by one: the claim that time-resolving the spectral term improves the predictor is a
hypothesis this note proposes to test (§7), not a given.

## 2. Two defects in the current loss

### 2.1 Batch pooling puts the mean inside the log

`PSDLoss.__call__` reshapes `(B, T, C)` into `(C, B*T)` and calls `welch_psd` with
`nperseg = span_steps`. The batch dimension therefore *becomes* Welch's segment axis, so the loss
computes

$$\log \mathbb{E}_b[\hat{P}] - \log \mathbb{E}_b[P]$$

An over-predicted trajectory can offset an under-predicted one before the log is ever taken. The
model can reach a low loss while being wrong on every individual rollout.

This is a property of the pooling, not of Welch. It is fixable in place.

### 2.2 There is no time resolution inside the span

With `span_s: 1.0` at `fs = 50 Hz`, `span_steps = 50` and `nperseg = 50`, so each trajectory
contributes exactly **one** periodogram. The `K` in Welch's segment average is the batch size, not a
time index. Within a rollout the spectral term sees a single phase-blind magnitude spectrum over the
full second.

### 2.3 What is *not* a defect

`curriculum_mse` runs at weight 1.0 alongside it and is fully time-resolved. Timing error already
has a gradient path. The open question is whether time-resolving the *spectral* term adds anything
on top of that — not whether the composite objective is blind to timing. It isn't.

## 3. What the loop already has

Two pieces of this already exist and should not be re-proposed:

- `_spectral_hinge_cost` is already a sliding-window periodogram cost. It reduces with a mean over
  `(window, channel, bin)` — never over windows alone — so a hot sub-window cannot be cancelled by a
  cold one, and `w_psd` stays independent of the window count.
- `EegMsLoss` is already hopped, per-sample, per-window, log-space, and in raw units. It is a
  spectrogram loss with a single frequency bin.

So the genuinely open question is narrow: **does resolving frequency inside the window beat
broadband `eeg_ms`, and does per-window beat pooled Welch?** Any comparison that omits `eeg_ms` is
measuring against a strawman.

## 4. The cost of removing averaging

Dropping the mean over frames changes the estimand from $\log \mathbb{E}[P]$ to
$\mathbb{E}[\log P]$. For a periodogram cell, which is $\chi^2_2$ around the underlying spectrum,
those differ by Euler–Mascheroni. Averaging $K$ independent cells before the log gives bias
$\psi(K) - \ln K$ and variance $\psi'(K)$:

| avg factor `K` | bias (nats) | bias (dB) | std (nats) |
| :--- | ---: | ---: | ---: |
| 1 | −0.577 | −2.51 | 1.28 |
| 2 | −0.270 | −1.17 | 0.80 |
| 4 | −0.130 | −0.57 | 0.53 |
| 8 | −0.064 | −0.28 | 0.36 |

Consequences:

- **A naive per-cell loss teaches the geometric mean spectrum** — about 56% of true mean power, 2.5
  dB low.
- **The bias is state-dependent.** High-SNR bins (a strong oscillation) are near-deterministic and
  barely biased; background bins are fully $\chi^2_2$ and get pushed down. Net effect is an inflated
  peak-to-background contrast. That matters because the controller thresholds absolute power against
  a fixed envelope.
- **There is an irreducible floor.** The predictor is deterministic and the target is one noise
  realisation, so even a perfect model cannot match the realised periodogram fluctuations. The loss
  will not approach zero, and its floor depends on the geometry — which is why geometries must be
  compared by *discriminability*, not by loss value (§7).

**Scope:** this is a training-loss concern only. Inside the MPC the rollout is deterministic given
`u` and the reference is a fixed 0.90-quantile envelope, so there is no $\chi^2$ noise in
`_spectral_hinge_cost` to average away. Variance-reduction machinery does not belong on the
controller side.

## 5. Which axes are safe to pool

One rule covers every axis: **pool power across cells whose underlying spectrum is smooth relative to
the pooling width; pool the squared residual freely after the log; never pool the signed log
residual.** The last clause is §2.1 stated generally — a signed residual pooled over anything lets a
positive error cancel a negative one, whichever axis it happens on.

| axis | pre-log (power) | note |
| :--- | :--- | :--- |
| batch `b` | no | distinct trajectories are not draws of one spectrum — this is §2.1 |
| channels `c` | no | 62 channels with genuinely different spectra |
| frames `m` | yes, bounded | up to the envelope's correlation width — §5.2 |
| frequency bins `f` | yes, bounded | up to the spectrum's smoothness in `f`; costs `Δf` |
| tapers (DPSS) | yes | near-independent estimates of the *same* cell |

Pooling the loss over the batch **after** the log and square is always safe and always helps — it
reduces gradient noise without touching the estimand. §2.1 and §4 are therefore not in tension: move
the mean outside the nonlinearity and both are served.

### 5.1 Pooling is unavoidable

A periodogram is an inconsistent estimator: each cell carries 2 dof however long the window is.
Lengthening `W` buys frequency resolution and buys **no** variance reduction. No geometry escapes the
§4 table; the only question is which axis pays for the dof, and Welch's answer — the time axis, all
of it — is the one answer this note rejects. Four knobs, four distinct trades:

| knob | `Δf` | effective time resolution | variance |
| :--- | :--- | :--- | :--- |
| lengthen `W` | improves | worsens | unchanged |
| pool `n` frames | unchanged | worsens by ~`n·H` | `/n_eff` |
| pool `n` bins | worsens by `n` | unchanged | `/n` |
| `K` tapers | worsens by `2·NW` | unchanged | `/K` |

Pooling frames is therefore *not* the same operation as lengthening the window, and the two should
not be conflated.

### 5.2 The frame axis is a filter, not a switch

Welch and a raw spectrogram are the endpoints of one continuum — smooth along `m` with a kernel,
then decimate. Welch is a full-span boxcar decimated to a single frame; consecutive Welch blocks are
a boxcar of width `b` at stride `b`; a raw spectrogram is the identity kernel. Nothing forces an
endpoint.

Extracting frames densely (small `H`) and pooling them with a kernel is strictly more flexible than
blocking, at the same FFT cost. Blocking couples extraction to pooling, plants hard boundaries where
a burst can be split across two blocks, and discards overlap that is free.

What sets the kernel width is an empirical quantity: **the correlation width of the log-power
trajectory along `m`.** Pooling narrower than that is free — it buys dof at no cost in signal,
because the envelope genuinely does not move within the kernel. Pooling wider destroys what the loss
exists to measure. That width is measurable from existing trajectories (§8) and is currently unknown.

Two constraints on the kernel:

- **Power or squared residual only.** Smoothing the signed log residual along `m` reintroduces
  cancellation, on the time axis instead of the batch axis.
- **Centred is fine here, but not everywhere.** A training loss sees the whole rollout offline, so a
  symmetric kernel is legitimate. Anything the controller consumes must be trailing — which is why
  `EegMsLoss` uses causal trailing windows.

The kernel must also keep `M_out > 1`. A weighted mean that still collapses the frame axis to a
single output is not a midpoint between Welch and a spectrogram — it is Welch with fewer effective
dof, since `Var[Σ w_m P_m] = σ² Σ w_m²` is minimised by uniform weights. Weighting only earns
anything in the sliding form.

**Kernel shape** is a knob distinct from the four in §5.1: at fixed `W` and fixed `Δf` it trades
localisation against effective dof, `K_eff = (Σw)² / Σw²`.

| kernel | `K_eff / n` | character |
| :--- | ---: | :--- |
| boxcar | 1.00 | maximum dof, weakest localisation |
| triangular | 0.75 | mild |
| Hann | 0.67 | width `n` carries the dof of a boxcar of width `2n/3` |
| Gaussian (`σ = n/6`) | 0.59 | tightest localisation, most expensive |
| EMA (`α`) | `(2−α)/α` frames | one parameter, causal, controller-computable |

Two cautions before reaching for a taper. There is **no leakage to fight** — the kernel is
non-negative and the smoothed quantity is positive power, so the usual reason to taper an analysis
window does not transfer; a taper here earns its place only through localisation. And once frames
overlap (`H < W`) they are correlated, at which point the variance-optimal weights are **edge-heavy,
not centre-heavy**. Localisation and variance then pull in opposite directions, and which wins
depends on the correlation width measured in §8.

### 5.3 Estimator weights versus preference weights

Two reasons to weight frames, with different homes:

- **Estimator quality** — pre-log, inside the kernel of §5.2. Changes *what is estimated*, and pays
  in bias and dof.
- **Preference** ("late-horizon frames matter more", "frames near onset matter more") — post-square,
  as `Σ_m v_m D(m, c, f)²`. Free by the §5 rule: no dof cost, no bias, estimand untouched.

Any weighting motivated by importance rather than by variance belongs in the second form. Putting it
in the first silently changes the estimand to buy something the second gives away.

### 5.4 Multitaper

The principled version of frequency pooling: `K` orthogonal Slepian tapers on the same window give
near-independent estimates with no time smearing, from fixed `scipy.signal.windows.dpss` weights, so
it is `K` precomputed weighted FFTs and trivially differentiable. Not free either — time-bandwidth
`NW` costs a resolution bandwidth of `2·NW·Δf` for `K ≤ 2·NW − 1` tapers, and the band of interest is
only 9 Hz wide.

## 6. Constraints that bound the geometry

These are measured facts about the loop, not choices:

- `fs = 50 Hz` (`dt = 1e-4` plant, `downsample: 200`), so **Nyquist is 25 Hz**. Every geometry must
  be stated in samples, not seconds — 0.1 s is five samples here.
- The band the metrics layer commits to is `SEIZURE_BAND_HZ = (3.0, 12.0)`.
- `Δf = 1/W`, `n_bins = W/2 + 1`, `M = ⌊(span − W)/H⌋ + 1`.
- Span budget: an earlier rollout probe found the `mse02_psd` predictor holds power out to ~75 steps
  (1.5 s). That caps the loss span until re-measured.
- The MPC's DFT is an explicit dense matmul per window in CasADi (no FFT available), so controller
  cost scales with `n_windows × n_bins`. Training geometry and controller geometry need not be
  identical, but any mismatch has to be a deliberate decision (§7).

**The three-way trade.** A 75-sample span carries at most ~75 real degrees of freedom, whatever
tiling is imposed on it. Frames `M`, in-band bins, and estimator dof `K` all draw from that one
pool:

| `W` (samples) | `Δf` (Hz) | in-band bins | `H` | `M` |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1.00 | 10 | 25 | 2 |
| 50 | 1.00 | 10 | 12 | 3 |
| 37 | 1.35 | 6 | 19 | 3 |
| 25 | 2.00 | 5 | 12 | 5 |
| 15 | 3.33 | 3 | 15 | 5 |

`M` overstates the independent frame count whenever `H < W`, since overlapping frames share samples.
The table is the shape of the problem, not a recommendation.

## 7. Open questions

1. Does frequency resolution inside the window beat broadband `eeg_ms` at all?
2. Does per-window scoring beat pooled Welch once §2.1 is fixed in place?
3. What `(W, H, span)`? The three-way trade above has no obvious optimum.
4. Which pooling axes and widths — frame-axis kernel, bin pooling, multitaper — and at what
   effective `K`? These are independent knobs (§5.1) and can be combined.
5. Must training geometry match the envelope's, or is alignment with the controller worth less than
   a better-conditioned training geometry? The envelope's own geometry is equally open and can be
   re-derived.
6. `_EPS` is applied in standardised units in the training loss while `LOG_FLOOR` is the same
   constant in raw units in the MPC. Same number, different effective threshold. Pick one
   convention.
7. Should the predictor stay two-sided, or be weighted toward the regime where the controller's
   hinge is active? A symmetric loss over all bins does not prioritise the bins that drive control.

## 8. How to decide it

Measure the noise floor rather than arguing it. For each candidate geometry and averaging scheme,
over existing trajectories and with **no training runs**:

- **Envelope bandwidth (first, and independent of any loss)** — measure the autocorrelation of
  `log P(m, c, f)` along `m` on existing trajectories. Its correlation width upper-bounds the
  frame-axis kernel (§5.2) and constrains `H`, so it is a prerequisite for the rest.
- **Floor** — evaluate the candidate loss between two independent realisations of the same config,
  `L(true_A, true_B)`. This is what a perfect model would score.
- **Signal** — evaluate `L(pred, true)` using the existing `mse02_psd` artifact.
- **Criterion** — rank candidates by discriminability,
  `d = (E[L(pred, true)] − E[L(A, B)]) / std`, not by loss value. A geometry whose signal and floor
  overlap is not measuring model quality, however well motivated its representation is.

A candidate is admissible only if `d` is clearly positive. Among admissible candidates, prefer one
that also ranks known-good against known-bad rollouts correctly. Only then train, and only then
score the closed loop.

## 9. Explicitly not doing yet

**Multi-resolution (MRSTFT).** At `fs = 50 Hz` the usual multi-scale ladder degenerates: a 0.1 s
window is 5 samples and 3 bins. It also adds one weight per scale on top of an existing weighting
problem — the spectral cost is a *mean* while control effort is a *sum*, which is already why
`w_psd: 1000` against `w_u: 10` does not mean what it looks like. Revisit only if a single scale
wins and the residual failure is demonstrably scale-related.
