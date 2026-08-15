# Predictability & controllability of candidate control metrics — experiment design

**Status:** agreed and being implemented. Design settled 2026-08-13 in a design interview; the two
decisions left open there (region labelling, channel weighting) were ratified the same day and are
folded into the text below. §Bandwidth is a later extension (2026-08-14), agreed and implemented;
it re-scores the same ensemble and needs no regeneration of it.
**Date:** 2026-08-13
**Question that started it:** "Which observable should the MPC actually minimise?" Every controller
in this repo has optimised 62-channel scalp EEG power, and
[`tes_field_geometry.md`](tes_field_geometry.md) §2.3 measured that this "is not a proxy for
regional propagation control at `lTCI`". Before choosing a replacement, the candidates need scoring.

## TL;DR

Score a set of candidate metrics on four axes — **predictability**, **controllability**,
**healthy/seizure separability**, and **feasibility** — and plot the first two against each other.
Both experiments read one generated ensemble; neither trains a predictor nor runs an MPC.

Predictability is measured **intrinsically**: branch many noise realisations from one shared plant
state and watch how fast the metric's future stops being determined. That is a ceiling no predictor
can beat, and being model-free it does not depend on the state of the predictor pipeline.

Controllability is measured by a **paired open-loop probe**, not closed-loop MPC — deliberately, so
that a metric's score cannot be contaminated by identification or solver failure.

## Why the two axes are independent

The obvious simplification is to assume a metric that can be predicted can be controlled, and test
only predictability. The evidence in [`tes_field_geometry.md`](tes_field_geometry.md) §2 says
otherwise, on both sides:

- **§2.3 — a good predictor did not buy control.** The nonlinear predictor "fits the multi-step
  rollout cleanly (validation loss 0.1796)" and under MPC drove 62-channel scalp EEG power to 59.2,
  57 % below the unstimulated 139.3 baseline, on half the control effort. Regional seizure count
  stayed at **31/34, 0/7 seeds suppressed**. The objective was predicted and minimised; the seizure
  was untouched.
- **§2.2 — control authority is a separate quantity, and it can be tiny.** Under `roast_3d` the
  control input explains only **1.4 × 10⁻⁴** of single-step EEG variance and **5.0 × 10⁻³** of
  20-step variance. Whether a metric can be *forecast* says nothing about whether the actuator
  moves it.
- [`kcl_control_authority_investigation.md`](kcl_control_authority_investigation.md) shows the same
  from the actuation side: authority can be structurally near-null irrespective of model quality.
  (Read its "Superseded" banner first — the root cause it names was a symptom.)

So predictability and controllability get separate axes. A metric that is highly predictable and
uncontrollable is an expected quadrant of the result, not a failure of the experiment.

## Design

### Plant and branch points

Healthy and seizure plants differ **only** in the `A` vector — `3.25` uniform against `3.6` at the
EZ (`lHC`, `lPHC`, `lAMYG`) and `3.4` at the PZ (`lTCI`, `lTCV`). Everything else is shared:
`K: 0.60`, `sigma: 280`, `dt: 1e-4`, `initial_state: rest`, `roast_3d` stimulation
(`data/roast_leadfield_3d.npz`), `u_max: 2.0`. Pinned to the dynamics blocks of
[`jansen_rit_baseline.yaml`](../configs/simulation/jansen_rit_baseline.yaml) and
[`nonlinear_full_mpc.yaml`](../configs/simulation/nonlinear_full_mpc.yaml), so the healthy/seizure
contrast carries no confound.

**16 parent runs**: 8 healthy seeds to 4 s, 8 seizure seeds to 14 s. Each seizure parent is
snapshotted at all four seizure branch points, so one run serves four branches.

| branch | plant | `t_branch` | what it represents |
| --- | --- | --- | --- |
| healthy | `A = 3.25` | 4 s | separability reference, no seizure |
| pre-onset | EZ/PZ | 1 s | before the EZ ignites — the actionable window |
| EZ ignited | EZ/PZ | 3 s | focus seizing, spread not begun |
| mid-spread | EZ/PZ | 7 s | PZ recruited, hemisphere recruiting |
| saturated | EZ/PZ | 14 s | the regime where nothing helps |

Branch times follow the propagation schedule in [`seizure.py`](../src/neuro/seizure.py) (EZ ~1.5 s,
PZ ~5 s, left-hemisphere half ~10 s).

Phase is an explicit axis because the value of acting is known to depend on it:
[`tes_field_geometry.md`](tes_field_geometry.md) §3.4 — "intermittent bursts after seizure spread
fail; effective control requires early intervention to keep `lTCI` closed". Branching everything
from one operating point would measure that point's idiosyncrasy and generalise it wrongly.

### Branching mechanics

**"Same initial condition" is not just `x`.** Per [`jansen_rit.py`](../src/neuro/jansen_rit.py) the
plant state is `x` (6, n_nodes) **plus** a `history` ring buffer of length `max_delay + 1` **plus**
the step counter `k` **plus** the RNG. The public `initial_state=` constructor argument sets only
`x` and fills `history` with a constant sigmoid of it — correct from rest, wrong for branching
mid-seizure, and wrong silently.

Branch with `copy.deepcopy(dyn)` followed by `dyn.rng = np.random.default_rng(child_seed)`. That
preserves `x`, `history` and `k` exactly and needs no API change.

### Stimulation arms

Two arms per child, **sharing child seeds**:

- `u = 0`
- sustained hold `u = d1 = [+2.0, 0.0, -2.0]` mA over `(TP9, CP5, Ex8)`

`d1` is the `GOOD_COMMAND` constant in
[`probe_payoff_crossover.py`](../scripts/probe_payoff_crossover.py). At `u_max: 2.0` it sits exactly
on the box corner, and it is zero-sum, so KCL-valid.

Sharing seeds across the two arms is common random numbers. The plant's noise is additive on `x5'`
and state-independent ([`jansen_rit.py:166`](../src/neuro/jansen_rit.py#L166)), so the two arms stay
tightly coupled and the paired difference is far less noisy than the unpaired one — free
statistical power.

**The arm list is a config parameter.** With only `{0, +d1}`, a null controllability result cannot
distinguish "this metric cannot be steered" from "not along the one direction probed" — the exact
ambiguity the `gamma` investigation turned on. Under KCL the `roast_3d` leadfield's 3 electrodes
give an admissible input space of **exactly 2 dimensions**, so adding `-d1` and a matched-norm
orthogonal `d2` would characterise first-order controllability completely. That is deferred, not
excluded; adding it must be a config line, not a redesign.

### Ensemble size

Per branch: **8 parents × 16 children × 2 arms**, 3 s rollout. **1280 rollouts total.**

Multiple parents are load-bearing twice over. They supply the R² denominator (below), and without
them the separability arm has n = 1 per class, which makes Cohen's d, AUC and every other
separability statistic undefined.

## Metrics

Computed on **raw 10 kHz scalp EEG** (62 channels). Scalp is primary because a metric the
controller cannot measure is disqualified whatever else it scores; region LFP is used only to
generate ground-truth labels.

Raw rather than the controller-visible signal (`AntiAliasEstimator` + ZOH, see
[`filtering.py`](../src/neuro/filtering.py)): that path is parameterised by `downsample`, and
scoring through it would make the metric ranking conditional on a controller-configuration choice
that has nothing to do with which observable is worth targeting. Keeping them decoupled means the
ranking stays valid whatever the estimator is later set to. The predicted-tier metrics are all
inherently low-pass, so the two signals differ little for exactly the metrics likely to survive.

That last claim is an assumption, and §Bandwidth measures it rather than resting on it: raw 10 kHz
is one point on a bandwidth axis, and this section argues only that the point is *neutral*, not
that it is a good one.

**Trailing (causal) windows** — the controller only ever has the past. **50 ms hop**, giving ≤60
points over the 3 s rollout.

### Predicted tier

Scored on all axes.

| metric | window | note |
| --- | --- | --- |
| block PTP | 100 ms | incumbent; block-max is its own low-pass |
| 3–12 Hz band power | 500 ms | the seizure band; ≈2.5 cycles at 5 Hz |
| line length | 100 ms | clinical standard, trivially differentiable |
| synchronization R | 500 ms | scalar FC reduction, already in `utils.processing` |
| eegMS | 100 ms | the incumbent objective — included as the baseline to beat |
| spectral centroid | 500 ms | scalar PSD reduction, the only frequency-shape metric |

The centroid is taken over **1–45 Hz**, not to Nyquist. The plant's noise is white on `x5'`
([`jansen_rit.py:166`](../src/neuro/jansen_rit.py#L166)), so at a 10 kHz raw rate a full-band
centroid would be dominated by a noise tail carrying no physiology and would sit near-constant
regardless of state. Line length is reported as mean `|diff|` per sample rather than the classical
window sum, so it does not scale with the window length and stays comparable across the grid.

Every one of these pools over a channel set (mean across channels), so the pooling convention is
identical for all six and the channel set below is a clean second axis.

### Channel sets

Each predicted-tier metric is scored **twice**, over two channel sets:

| set | channels |
| --- | --- |
| `all62` | all 62 scalp channels |
| `lTCI` | the four channels loading hardest on `lTCI` — `TP9`, `TP7`, `T7`, `P7` |

The subset is derived from the EEG forward operator, not chosen by hand: it is the top-`k` of
`|L[:, lTCI]|` from [`build_eeg_gain`](../src/neuro/eeg.py), with `k = 4` the default. The loading
profile is sharply `TP9`-dominated — `TP9` is `2.77`, the next channel (`TP7`) `0.80`, a 3.4× gap —
so the set is in practice `TP9` plus its left-temporal ring.

This axis exists because the repo's strongest negative result is about pooling, not about any
individual metric. [`tes_field_geometry.md`](tes_field_geometry.md) §2.3 and §3.2: "minimising global
62-channel scalp EEG power is not a proxy for regional propagation control at `lTCI`" — "target
state-space gatekeeping, not global scalp power". And the one controller that works,
`AmplitudeThresholdController` at 6/7 seeds suppressed, triggers on **`TP9` alone** at a 10 mV
threshold (§2.1). Run over `all62` only, the experiment could plausibly conclude that no metric is
both predictable and controllable, for a reason already documented — and would not be able to tell
that conclusion apart from "we pooled it away".

It costs no extra simulation: same metric code, different channel slice. Twelve scored series per
branch instead of six.

### Descriptive tier

Plots and separability only, never predicted at 62×62: FC matrices, full PSD, topoplots, spread
rasters. Their scalar reductions are what bridge into the predicted tier.

The split follows from an identifiability limit, not from taste. A 62×62 correlation estimated over
a window of `W` samples has rank ≤ `W − 1`, so at a control-relevant window it is rank-deficient and
cannot express ∂FC/∂u. An honest 62-channel FC needs a window on the order of seconds, which is not
a control timescale. `_masked_fc` in [`nn_training.py`](../src/neuro/nn_training.py) is the existing
instance of this problem.

### Implementation

`src/neuro/metrics.py` provides a uniform registry with signature
`(signals, fs, window, hop) -> (times, values)` and **delegates** to the existing
[`processing.py`](../src/utils/processing.py) (`compute_psd`, `band_energy`, `synchronization`)
rather than reimplementing them.

## Scores

Per metric × branch. All scores are reported as **curves over lookahead `h`**, not single points.
`h_eval` sets the summary point, defaults to **1.0 s**, and is a script parameter and a notebook
slider — it is a chosen working lookahead, not a value this design derives.

### Predictability

```text
R2(h) = 1 - Var_within_ensemble(h) / Var_across_parents(h)
```

Conditional variance reduction, scoped per phase: the numerator is what remains unknown after
fixing `x0`, the denominator is what was unknown before. The across-parents denominator is *why*
multiple parents exist — a global pooled variance would be dominated by between-phase swings, so a
metric that merely moves a lot between healthy and saturated would score well without being locally
forecastable at all.

Reported alongside: **raw `sigma_ens(h)` in the metric's native units**, referenced to a **p5–p95**
range rather than min–max (min–max depends on the single most extreme sample and grows with sample
size, so rankings under it shift as seeds are added).

### Controllability

```text
d_ctrl(h) = delta_bar(h) / sigma_ens(h)      signed, positive = toward healthy
```

where `delta_bar(h)` is the mean paired difference `M_stim - M_zero`.

The **unpaired** `sigma_ens` denominator is deliberate. The paired `sd(delta_i)` answers "is there a
real effect?" and is far more sensitive; `sigma_ens` answers "can a controller exploit it?". A real
controller **cannot know the noise realisation**, so the effect it must steer against is buried in
`sigma_ens`. Scoring on the paired spread would license calling a metric controllable when the
effect is invisible to any causal controller. The two can differ by an order of magnitude.

The sign matters and is not a formality. [`tes_field_geometry.md`](tes_field_geometry.md) §2.3 is
the cautionary case: stimulation moved the scalp-power objective 57 % in the intended direction
while the seizure spread regardless. A metric that stimulation moves confidently in a direction
that does not correspond to recovery is controllable in the linear-systems sense and useless.

Secondary columns: paired `sd(delta_i)` for significance — it distinguishes a true null from an
underpowered one — and `delta_bar / Delta` for clinical magnitude.

**Measured caveat on the pairing.** The design assumed `sd(delta_i)` would be far below
`sigma_ens`, since the arms share their noise realisation. On a smoke run (3 parents x 4 children,
so provisional) the ratio `sd(delta_i) / sigma_ens` is ~**0.3 at h = 0.15-0.5 s**, reaches ~**1.0
by h = 1 s**, and at the `ez_ignited` branch *exceeds* 1 — up to ~3 for `block_ptp` and ~7 for
`band_power`. Shared noise couples the arms only until their trajectories separate; the plant is
chaotic, and stimulation changes which cycle the trajectory settles into, so past ~0.5 s the paired
difference is no more sensitive than the unpaired one and during ignition it is worse.

Common random numbers is still the right construction — it costs nothing and it is what makes the
difference a *difference of the same realisation* rather than of two populations. But the extra
statistical power it buys is a short-lookahead effect, and at the default `h_eval` of 1.0 s there
is none to speak of. `metric_scoring.py` plots this ratio against `h` so the claim is checked per
run rather than assumed. The headline `d_ctrl` is unaffected: it uses the unpaired `sigma_ens` by
design, for the reason above.

`d_ctrl` and `R2` are both dimensionless SNRs, so the headline scatter has commensurate axes.

### Separability, observability, feasibility

- **Separability** — Cohen's d between the healthy and saturated branches, n = 8 per class.
- **Observability** — per-metric scalp↔region correlation across the time grid. Nearly free (same
  metric code, different input array) and it catches metrics killed by volume conduction.
- **Feasibility** — minimum window length, as a hard filter. A metric needing a 1 s window is
  unusable at a 20 ms control rate however well it scores. Three independent filters, not one
  blended number.

Region-space labels use the existing [`SpreadProfile`](../src/neuro/seizure.py) criterion unchanged:
5 mV PTP, 1 s window, 1 s persistence.

## Bandwidth and the raw-signal baseline

**Date:** 2026-08-14. An extension, not a redesign: it re-scores the same generated ensemble and
adds no simulation.

Two gaps in the above. First, the scores rank the metrics against *each other* and never against
the signal the predictor pipeline actually forecasts — so the premise that a metric is easier to
forecast than the waveform is assumed rather than measured. Second, "raw 10 kHz" is a *point on a
bandwidth axis*, not a neutral origin; the §Metrics argument for it keeps the ranking independent
of the estimator's `downsample`, which is right, but it does not establish that the point is a good
one.

### The bandwidth axis

A causal 4th-order Butterworth low-pass applied to the scalp EEG before scoring, **with the sample
rate held at 10 kHz** — no decimation. Cutoff and rate stay separable that way, so a change in a
score is attributable to bandwidth alone and not to `line_length` and `block_ptp` being rate-
dependent by construction.

Cutoff grid: **`{raw, 500, 100, 45, 20, 10} Hz`**. 45 Hz is the clinical anchor and the centroid's
own bound, ~100 Hz is roughly where Jansen-Rit content ends given `a = 100 s⁻¹`, and 20/10 Hz
straddle the 3–12 Hz seizure band so its collapse is observed rather than assumed.

Swept: **`block_ptp`, `line_length`, `eeg_ms`, `synchronization`**, plus the two baselines below.
Held at the current scoring: `band_power` and `spectral_centroid`. The filter does not denoise
those two, it *redefines* them — a low-pass below 12 Hz mutilates the 3–12 Hz integral, and the
centroid of a filtered signal moves down mechanically, so neither is the same quantity at each
sweep point. `synchronization` is in despite not being in the original "broadband" list: it is
cheap, and narrowing the band raises inter-channel correlation more or less mechanically, so it is
plausibly the metric the filter moves most.

**The filter must be causal**, `sosfilt` via the existing
[`design_antialias_sos`](../src/neuro/filtering.py) path. The zero-phase licence granted elsewhere
in this document ("the causal constraint applies to the control path, not to an artefact read back
offline") does **not** extend here: a zero-phase filter mixes future samples into the trailing
window, which inflates predictability by construction. In a predictability experiment that is not a
stylistic choice, it is a wrong answer.

**R² alone is gameable by this axis.** Filtering strips the fast, child-specific component while
leaving the slow, parent-determined one, so `sigma_ens` falls faster than the total variance and R²
rises — monotonically, to a degenerate optimum at DC where the observable is perfectly predictable
and perfectly useless. Two things make the sweep a trade-off rather than a maximisation, and both
already exist:

- **All four axes are recomputed per cutoff**, not just R². The expectation is that R² and `d_ctrl`
  both rise — the stim arm is a DC hold, so its effect is low-frequency and survives filtering
  better than the noise does — and that what dies is separability and observability. If instead
  `d_ctrl` rises while Cohen's d collapses, that is the documented "controllable but meaningless"
  quadrant reappearing as a function of bandwidth.
- **Effective latency** `window_s + group_delay` joins the feasibility axis. Group delay is
  **measured at DC from the designed sections**, not quoted from a formula. Reported against R²,
  the sweep becomes a curve a controller designer can read.

**Measured group delay**, order 4 at `fs = 10 kHz`: `500 Hz` 0.8 ms, `100 Hz` 4.2 ms, `45 Hz`
9.2 ms, `20 Hz` 20.8 ms, `10 Hz` 41.6 ms. So the latency price of the whole axis is small against
a 100 ms window, and the trade-off curve is correspondingly flat in `x` — worth knowing before
reading it, because it means a bandwidth win is close to free and the axis will *look* one-sided.

It is measured by differencing the phase across the sections rather than through
`scipy.signal.sos2tf` + `scipy.signal.group_delay`. At the narrow end — 10 Hz against 10 kHz —
every pole sits within `1e-3` of `z = 1`, the assembled denominator is ~`1e-9` at DC, and
`group_delay` warns about the singularity it has landed on. Each second-order section stays well
conditioned.

**Filter transient.** `sosfilt` from zero state at the start of each 3 s rollout gives a settling
response near-identical across children of a parent, inflating short-`h` R² on top of the window
confound already recorded below. Handled twice over: `sosfilt_zi` steady-state initialisation, and
a settle time added to the shaded invalid region — computed from the cutoff, since it grows as the
cutoff falls.

### The raw-signal baselines

Both are scored on the **`zero` arm only** — the question is how predictable the unstimulated
signal is — and both are **per-channel**, so the existing `all62` / `lTCI` channel sets fall out of
the same computation rather than needing a second pass.

| rung | definition | what it bounds |
| --- | --- | --- |
| waveform | per-channel EEG sample at lookahead `h` | a predictor forecasting `y(t)` sample-wise |
| envelope | causal 3–12 Hz envelope detector | forecasting band amplitude, phase discarded |
| `eeg_ms` | already in the registry, 100 ms window | forecasting broadband power |

R² is the same estimator throughout: `1 - mean_parents Var_children / Var_all`, evaluated
cross-sectionally at each `h`, then averaged **across channels as R², not across channels as
signal** — a channel-mean of a signed waveform cancels.

The gap between the first two rungs is the **phase-divergence cost, quantified**. That is the
number that says whether moving the MPC objective off the waveform is worth anything, and it is
currently the missing link between this experiment and the predictor pipeline.

**The envelope is a causal detector, not Hilbert.** The analytic signal at time `t` depends on the
whole record, so a Hilbert envelope leaks the future into every sample — the same failure as
zero-phase filtering, in a different costume. Instead: causal band-pass to 3–12 Hz, rectify, causal
low-pass at **3 Hz**, scaled by `pi / 2` so the output is an amplitude rather than a mean-rectified
value. The smoothing cutoff is the band's *lower* edge, which puts the rectified carrier — from
`2 * 3 = 6 Hz` upward — at least an octave into the stop band whatever the input frequency.

That makes it **window-free**, which is what distinguishes it from `band_power` (500 ms window) and
earns it the middle rung. Its effective latency is its two filters' group delay, **measured at
0.218 s** (79 ms band-pass at band centre, 139 ms smoother): cheaper than `band_power`'s window,
but not by much. It is a bound, not a deployment candidate.

Its band is **fixed at 3–12 Hz and not swept**, for the same reason `band_power` is not: the
bandwidth axis would redefine it rather than filter it. So it is scored once, on the unfiltered
signal, and the archive stores it only there.

**The waveform baseline reads the full-rate store, never `cached_scalp`.** That cache is built with
zero-phase `scipy.decimate` ([`ensembles.py:476`](../src/neuro/ensembles.py#L476)), which mixes
future samples into every output sample — the same trap again. `score_ensemble_dir` already goes to
the full-rate store; the baselines must too.

**The baselines need a finer `h` grid.** The plant is chaotic in phase, so waveform R² is expected
to collapse within a couple hundred ms — well below the 50 ms hop's resolution and far below
`h_eval = 1.0 s`. On the shared grid the reference would read as a flat zero and look like a bug.
Grid: **5 ms steps to 0.5 s**, then the shared 50 ms grid onward.

### Implementation deltas

Regeneration of `scores.npz` is accepted where it buys simpler code; the cache is derived data.

| file | change |
| --- | --- |
| `filtering.py` | `design_lowpass_sos`, `design_bandpass_sos`, `causal_filter`, `group_delay_s` |
| `metrics.py` | variance-ratio core generalised; causal `envelope`; `baseline_grid`/`sample_at`/`baseline_r2`; `score_store` made row-major |
| `ensembles.py` | `cutoff` joins the `ScoreArchive` key; baselines cached as per-channel R² |
| `notebooks/metric_scoring.py` | rungs figure, and R² against effective latency per cutoff |

`design_antialias_sos` is parameterised by `downsample`, and inverting a free cutoff through it
gives non-integer factors (45 Hz → `downsample = 111.1`). Splitting out a cutoff-parameterised
designer keeps the repo to **one** filter definition rather than letting the sweep drift into a
second, silently different one.

Cutoff is a **key dimension, not a `Metric` field**. Making the filter part of a metric's identity
would turn `METRICS` into a cross product and re-filter once per metric; as a key dimension there
is one filter pass per rollout with every metric scored off it.

Baselines cache **per-channel R² curves, not per-rollout series**. Series would be ~500 MB against
the archive's current few MB, and buy only a baseline `d_ctrl` that nothing needs.

`score_store` became **row-major and multi-metric**, which is what the cost claim below rests on.
It previously traversed the store once per metric and channel set — 12 traversals of a 1.9 GB file
per branch and arm — which was affordable exactly once and is not affordable per cutoff.

### Cost

Cheaper than the metric sweep it extends, because the two Welch metrics are excluded — they are
the expensive half. Per cutoff: one `sosfilt` pass and three cheap reductions per rollout, so the
bottleneck is reading the store, not computing.

Traversals per branch and arm: **one per cutoff** for the metrics, since `score_store` scores every
metric and channel set off each rollout as it is read; plus **one more per cutoff on the reference
arm** for the baselines. The baselines are not folded into the metric pass on purpose — they are
per-channel where the metrics are scalar, and one function doing both would be doing two unrelated
things to save a traversal that costs a `sosfilt` and a memory-mapped read.

## Known confounds

**Windowed metrics inflate short-`h` scores.** For `h` shorter than the metric window, the trailing
window still overlaps pre-branch history that every child shares, so `sigma_ens` is artificially
near zero. Scores are valid only for `h >= window`; the invalid region is **shaded in plots, not
hidden**.

**The plant is bimodal across seeds.** [`tes_field_geometry.md`](tes_field_geometry.md) §1.3:
outcomes split into ~4–5 seizing regions (suppressed) or ~27–35 (unsuppressed). Consequences:

- `Var_across_parents` at the later branches is partly *which basin the parent fell into*, not
  within-phase variation. That is still the correct denominator — it genuinely is the uncertainty
  before knowing `x0` — but it is bimodal, and n = 8 estimates it coarsely.
- The `saturated` and `mid-spread` branches will contain parents that never seized, so those
  ensembles are mixtures.

Handling: **record each parent's basin** (final seizing-region count from its `SpreadProfile`) in
the run manifest, and report R² and Cohen's d both pooled and split by basin. Cheap, and it keeps
the mixture visible rather than averaged away.

**Interpretation is post-hoc by decision.** Pass/fail thresholds were deliberately not
pre-registered. Reporting full curves over `h` rather than a single summary point is the
mitigation: any threshold is then applied in the open, against the whole curve.

## Deliverables

| path | purpose |
| --- | --- |
| `src/neuro/metrics.py` | windowed metric registry, uniform signature; and the scores below |
| `src/neuro/ensembles.py` | parent runs, snapshot/branch, arm execution, scoring sweep and caches |
| `scripts/run_predictability_experiment.py` | generation CLI |
| `notebooks/ensemble_explorer.py` | trajectories, FC, PSD, topoplots, healthy vs seizure |
| `notebooks/metric_scoring.py` | R², `d_ctrl`, separability table, the 2D scatter |

The score functions (`predictability_r2`, `sigma_ens`, `separability`, `controllability`,
`scalp_region_correlation`) live in `metrics.py` rather than in the notebook. They are the
quantitative claim of the experiment and every one of them is a formula that fails quietly when it
is wrong — a sign convention, a variance denominator, a `ddof`. In a notebook cell none of that is
testable; `tests/test_metric_scores.py` pins all of it. The notebook only selects and plots.

### Running it

```bash
uv run python scripts/run_predictability_experiment.py --out data/predictability_ensemble
uv run marimo edit notebooks/metric_scoring.py     # scores and caches on first run
uv run marimo edit notebooks/ensemble_explorer.py
```

Add `--dry-run` to print the cost and storage plan without generating, and `--n-parents 1
--n-children 2` for a smoke run.

Notebooks split **by purpose** (visual against quantitative), not by experiment — splitting by
experiment would break the headline figure across two files, since it needs both axes.

**Headline figure:** R² against signed `d_ctrl` at `h_eval`, one point per metric × channel set, one
panel per phase, marker size = Cohen's d, marker shape = channel set.

### Cost and storage

Simulated time is fixed by the design: 16 parent runs come to **144 s**, and 1280 child rollouts of
3 s come to **3840 s**, so **3984 s** in total.

**Measured**, not extrapolated, on the real `ensembles.py` path (1 parent per plant kind, all five
branches, 78 simulated s): **36 s wall for 78 simulated s, i.e. 0.46x realtime**. Generation is
therefore about **31 min single-threaded**, not the ~2.7 h a `probe_payoff_crossover.py`-based
estimate suggests. The gap is the orchestrator: that probe drives a full `Simulation` with sensor,
estimator, controller and logger per step, where this path steps `simulate_network` directly and
reduces to EEG once at the end. Left serial — it is a one-off overnight-scale job at worst, and
`--n-parents` / `--n-children` scale it down for a smoke run.

**Scoring is the expensive half.** Measured at 241 s for 120 rollouts, so **~43 min** for the full
1280, dominated by the Welch transform `band_power` and `spectral_centroid` need (~0.8 s per
rollout each at 62 channels; the other four metrics together cost ~0.04 s). It is therefore done
once by `score_ensemble_dir` and cached to `scores.npz` beside the stores — a few MB, since the
scored series are ~60 points per rollout. Every downstream choice, `h_eval` included, is then
instant.

Storage: scalp EEG at full 10 kHz, `1280 × 30000 × 62 × 8 B` ≈ **19 GB**. One `.npy` per
branch × arm (10 files, ~1.9 GB each), opened with `mmap_mode='r'` — the repo already does this in
[`closed_loop_eval.py:101`](../src/neuro/closed_loop_eval.py#L101). `.npy` rather than `.npz`
because `np.load(..., mmap_mode=...)` is silently ignored for `.npz`; and each file is written
through `np.lib.format.open_memmap`, so generation never holds 1.9 GB resident.

Region LFP is stored too — the observability axis is a scalp↔region correlation, so it needs region
metrics — but **decimated to 1 kHz at float32**: `1280 × 3000 × 76 × 4 B` ≈ **1.2 GB**, against
23 GB if it were kept at 10 kHz like the scalp. Region space is only ever a reference signal, never
the controller's input, so it does not need the raw rate; and correlation across the time grid is
scale-invariant, so metrics whose value depends on `fs` (line length) still compare cleanly.

19 GB will not sit in a marimo kernel, so the notebooks build a **1 kHz scalp cache on first run**
(~950 MB at float32; every metric here tops out at 12 Hz of interest, so 1 kHz is ~40×
oversampled). Offline decimation uses zero-phase `scipy.decimate` — the causal constraint that
forces `sosfilt` in the control path does not apply to post-hoc analysis.

## Pointers

- Objective mismatch, control-variance share, the working threshold baseline, seed bimodality:
  [`tes_field_geometry.md`](tes_field_geometry.md) §1.3, §2, §3.
- Control-authority structure: [`kcl_control_authority_investigation.md`](kcl_control_authority_investigation.md)
  (note its "Superseded" banner — the root cause it names was a symptom).
- Seizure criterion and propagation schedule: [`seizure.py`](../src/neuro/seizure.py).
- Existing metric implementations to reuse: [`processing.py`](../src/utils/processing.py).
- Existing descriptive notebook, check before duplicating plots:
  [`healthy_vs_seizure_eeg.py`](../notebooks/healthy_vs_seizure_eeg.py).
- Target chapter: `\chapter{Experiments and Evaluation}` in
  [`TUDaThesis.tex`](../thesis/TUDaThesis.tex), currently empty.
