# Why the MPC does not suppress, and what to change

**Date:** 2026-08-10
**Context:** closes the last open item on the `roast_3d` closed loop.
Evidence and full result tables: [`tes_field_geometry.md`](tes_field_geometry.md) §2.
Artifacts, commands and traps: [`roast_operations.md`](roast_operations.md).

## Findings

**The nonlinear MPC is tractable.** At `shooting_depth: 20` (single shooting) a 20 s seed costs
~20 min and a 7-seed ensemble ~1 h at 3 workers, ~400–500 MB each. The earlier
">8.4 h, computationally impractical" write-up was measuring `shooting_depth: 1`, i.e. full
multiple shooting, and is retracted. No fallback (`expand`, lower `max_iter`, thread pinning, PCA)
was needed.

**It still does not suppress**, but it fails differently from the linear one (7-seed medians):

| run | suppressed | seizing L | eegMS | mean \|u\| |
| --- | --- | --- | --- | --- |
| no stimulation | 0/7 | 30 | 139.3 | 0 |
| threshold controller | 6/7 | 4 | 6.0 | 1.27 |
| linear MPC, `w_u` = 10 | 0/7 | 32 | 228.5 | 1.32 |
| nonlinear MPC, `w_u` = 10 | 0/7 | 31 | **59.2** | 0.67 |
| nonlinear MPC, `w_u` = 0 | 1/7 | 29 | 184.4 | 1.32 |

The linear MPC drove scalp power *above* the unstimulated baseline (228 vs 139) — §2.2 reads that
as an identification failure, correctly. The nonlinear MPC drives it **57 % below** baseline on
half the command effort, and 31/34 regions still seize. It is minimising its objective and
succeeding. The lone 1/7 at `w_u = 0` is one draw into the bimodal suppressed basin, not a result.

**The objective is anti-correlated with the outcome at its own horizon.** Five seeds, settled 12 s
into the seizure, comparing `u = 0` against the known-good sustained `[+2, 0, -2]` mA:

| what | median good/zero | improves in |
| --- | --- | --- |
| model's 20-step cost (what the MPC minimises) | 1.019 | 1/5 |
| true EEG power over 0.2 s (= the horizon) | 1.032 | 1/5 |
| true EEG power over 2.0 s | 0.968 | 3/5 |

Per-seed rows matter more than the medians. Seed 69: the good command makes EEG power *worse* over
200 ms (4.68 → 4.83) and **20× better** over 2 s (112.5 → 5.4); seed 1026 is the same shape
(5.83 → 7.23, then 46.7 → 6.4). In exactly the states where stimulating is hugely beneficial, the
200 ms cost reports it as slightly harmful, so an MPC minimising that cost is *correct* to refuse.

**The predictor is not the weak link.** The model's ratio (1.019) tracks the plant's true 200 ms
ratio (1.032): it accurately forecasts a quantity that does not matter.

**The window to act closes.** Seeds 1023–1025, already fully seizing at t = 12 s (EEG MS ~210–260),
do not benefit even over 2 s (227 → 231, 227 → 237, 259 → 251). That is §2.1's "trigger early,
then hold" measured directly — the controller must value an action whose payoff is seconds away.

## Next steps

**1. Measure where the payoff crosses over, before picking a `dt`.** The probe above scored two
points — 0.2 s (wrong sign) and 2.0 s (right sign) — and everything below depends on where it
flips. Rerun the same 5-seed `u = 0` vs `[+2, 0, -2]` mA probe scoring true EEG power at
0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.5 / 2.0 s. "~2 s of lookahead" is an assumption until this is run,
and it is the difference between a one-line config change and rewriting the output representation.

**2. Extend the horizon with the mildest `downsample` that clears the crossover.** Not by raising
`horizon` to 200 — the NLP grows 10× and the predictor is trained to a 20-step rollout. Retrain at a
coarser `downsample` keeping `horizon: 20`: same NLP size, more lookahead. But the coarsening is not
free. Measured on `sim_000.npz` (62 ch, 20 s), bare striding against `scipy.decimate`:

| `downsample` | fs | Nyquist | 20-step lookahead | alias RMS |
| --- | --- | --- | --- | --- |
| 100 (current) | 100 Hz | 50 Hz | 0.20 s | 1.0 % |
| 200 | 50 Hz | 25 Hz | 0.40 s | 2.2 % |
| 250 | 40 Hz | 20 Hz | 0.50 s | 3.0 % |
| 400 | 25 Hz | 12.5 Hz | 0.80 s | 11.1 % |
| 1000 | 10 Hz | 5 Hz | 2.00 s | **53.9 %** |

66 % of EEG variance sits above 5 Hz — 60.9 % of it in the 5–12 Hz seizure band alone — so at
`downsample: 1000` the strided signal is majority fold-back, with its variance inflated 2.35×.
Prefer `downsample: 250`: 500 ms of lookahead at 3 % alias, raw EEG still intact (Nyquist 20 Hz
clears the whole seizure band), no representation change. Go past 400 only if step 1 says the
crossover demands it — and then step 4 is mandatory in the same change.

**3. Causal anti-alias low-pass, in training *and* in the loop.** `load_trajectory`
([`nn_training.py:44`](../src/neuro/nn_training.py#L44)) decimates by bare striding,
`y_mea[:max_idx:downsample]`, with no filter anywhere. That is harmless at `downsample: 100` (1 %)
and indefensible past 250. Low-pass at the new Nyquist before striding, and apply the *identical*
filter online to the EEG feeding the controller's `y`-buffer — otherwise the model sees a different
signal in the loop than it was fit on. It must be **causal** (`sosfilt`, not `sosfiltfilt`):
`scipy.decimate` defaults to `zero_phase=True`, i.e. non-causal `filtfilt`, which cannot be
reproduced online, so using it on training data bakes in a train/serve skew that reads as model
error. Same coefficients both paths; the group delay then belongs to the identified plant instead
of being a mismatch.

**4. Forecast an envelope, not the waveform** — needed only if step 1 pushes `dt` past ~40 ms. At
100 ms sampling raw EEG is unrepresentable in both directions: strided it aliases (54 %), correctly
filtered it has the 5–12 Hz seizure rhythm deleted. The escape is to predict a quantity that is
already slow. AC-power containment below the 5 Hz Nyquist of a 10 Hz grid:

| feature | inside Nyquist | note |
| --- | --- | --- |
| 100 ms block PTP | ~99 % (99 % below 4.8 Hz) | block-max is its own low-pass; no extra filter, cheap in CasADi |
| 3–12 Hz Hilbert envelope | ~93 % (95 % below 5.6 Hz) | needs a quadrature FIR; 12 % error if strided naively |

Prefer block PTP. The envelope is slower than the waveform but not as slow as assumed — seizure
onset is sharp, and 5–7 % of its amplitude dynamics live above 5 Hz. Either feature is also the
quantity that should be small, so `sumsqr(y)` stops being a proxy and becomes the target. Weight it
toward the channels loading on `lTCI` rather than uniformly across 62, since suppression here is a
propagation block at that region (§1.2). The cost, stated plainly: carrier phase is gone, so
phase-locked policies become inexpressible — acceptable only because the known-good policy is a
seconds-long hold. Check the tES path tolerates a 100 ms ZOH on `u` before committing.

**5. Reshape the excitation set: sustained holds, and coverage of the transition.**
`hold_ms: [10, 50, 200]` means the longest constant command in training is 200 ms, while the working
policy holds for seconds at 95 % duty. Regenerate with 1000/2000/4000 ms added. This is a
*prerequisite for step 2*, not the root cause: at a 200 ms horizon the gap does not bind; past
500 ms the model would extrapolate on every solve. Changing the data alone would not have moved the
closed loop.

The same regeneration should fix what the set spends its samples on. Over the 22
`experiment_excited_roast/train` trajectories only 9 show a clear seizure onset (plateau ≥ 3×
baseline EEG PTP), and among those the envelope reaches 90 % of its plateau at a median of **3.0 s**
[p10 2.2, p90 4.0] — leaving ~85 % of each 20 s trajectory in saturated, stimulation-insensitive
dynamics. Stimulation is injected throughout the excitation runs, which confounds the envelope, so
read these as indicative rather than exact; the direction is not in doubt. Under plain MSE that
saturated majority is where the capacity goes. Generate a **new dataset with shorter end times**,
trajectories ending around the transition rather than at 20 s, rather than reweighting the existing
one. Hold the total sample count constant: the set is 22 × 20 s = 440 s today, so an ~8 s end time
needs ~55 trajectories to keep the same 440 s. That also buys more onset draws, worth having when
only 9 of the current 22 produce one.

**Set the closed-loop eval `t_end` to 12 s in the same change, and re-run the baselines.** Seeds
1023–1025 are fully seizing at 12 s and do not benefit even over 2 s, so nothing after that changes
the outcome. The scoring split in [`closed_loop_eval.py`](../src/neuro/closed_loop_eval.py) is what
makes the choice bite: `n_seizing()[-1]` is an end-state verdict and shrugs off a predictor
extrapolating late, but `mean_amplitude` averages `|u| / u_max` over *every* logged step — so a
model identified only on transitions, commanding garbage from 12–20 s, corrupts precisely the
duty-cycle number the success criterion below is written against. 12 s also cuts ~40 % off the
~20 min/seed runtime. The cost: all five rows of the Findings table were measured at `t_end: 20.0`
and stop being comparable, so re-run them before scoring anything new.

**Skip:** solver tuning, `w_du` / `w_u_l1`, bigger networks, PCA. The `w_u` sweep already showed the
controller side is not a lever, and the probe above shows why.

**Skip — LQR terminal cost `x_N' P x_N`.** Tested against `artifacts/roast/linear_full`, whose exact
Jacobian gives a 951-dim state: ρ(A) = 1.119 with 3 unstable modes that are only marginally
stabilisable (PBH σ_min 2e-5 – 2e-4). `solve_discrete_are` returns, but `P` has cond 3.5e34 and a
min eigenvalue of −2.7e-7 — indefinite, so `ca.qpsol(..., "osqp")` would get a non-convex Hessian.
On a random state `x'Px` exceeds the stage cost by ~1e6: it would not extend the 20-step horizon, it
would replace it. The predictor is also affine rather than linear (‖c‖ = 0.27, output offset up to
1.8 mV) and `x'Px` silently drops that term. Above all it asks a model fit to 20-step rollouts to be
right to infinity, on the same linear identification that drove scalp power *above* baseline.
If revisited, falsify cheaply first — scale `w_y` on the final step only (2 lines, works for
`MPCController` too). The probe above predicts it changes nothing, because the payoff sign is wrong
across the whole window, not just at its end.

**Skip for now — the `w_psd` / `w_fc` auxiliary losses.** Both are 0 in every config under
`configs/nn_predictor/roast/`, and enabling them as-is is not a one-line change. Measured on a
512-sample batch against `artifacts/roast/linear_full`, raw components are `mse` 0.234, `psd` 5.779,
`fc` 0.012 — the two auxiliaries are miscalibrated against the MSE in *opposite* directions, by 25×
and 20×. `w_psd = 1` puts the spectral term 25:1 over the MSE and inflates the gradient norm 4.1×;
`w_fc = 1` moves it by 5 %. Commensurate values are roughly `w_psd ≈ 0.04` and `w_fc ≈ 20`.

Worse, `psd_gate = step_mask[-1]` ([`nn_training.py:451`](../src/neuro/nn_training.py#L451)) makes
the spectral term exactly zero until the curriculum mask is full:

| curriculum `L` | `mse` | `psd` | `fc` |
| --- | --- | --- | --- |
| 1 | 0.006 | **0.000** | 0.000 |
| 10 | 0.147 | **0.000** | 0.004 |
| 19 | 0.224 | **0.000** | 0.011 |
| 20 | 0.234 | **5.779** | 0.012 |

At `curriculum_start_epoch: 50` / `curriculum_end_epoch: 225` / `epochs: 250` the mask fills around
epoch 221, so `w_psd > 0` is inert for 220 epochs and then a term 25× the MSE appears in a single
step, just as `patience: 100` is winding the run down. That is a late shock, not an objective.

Both statistics also measure the wrong thing for a control objective. `_masked_fc` pools over batch
*and* horizon, yielding one 62×62 matrix from 5120 rows: two disjoint halves of the same batch
differ by 5.7 % Frobenius, i.e. it is very nearly a dataset constant, averaging stimulated and
unstimulated samples into the same number. It cannot express ∂FC/∂u. The pooling is not gratuitous
— a per-rollout FC would estimate 62×62 from 20 samples (covariance rank ≤ 19), which is
unidentifiable — so FC is unavailable at this window length rather than merely mis-implemented. The
PSD term meanwhile has 11 bins at 5 Hz spacing (`nperseg = horizon` at fs = 100 Hz), of which the
3–12 Hz seizure band occupies two; the log-space error equalises predicted bins spanning seven
decades (8.1e-7 to 5.9), so most of the gradient budget goes to matching near-empty
high-frequency bins. It whitens, it does not target the band.

Revisit only after step 2. FC becomes estimable once the effective window lengthens, or restricted
to the `lTCI`-loading channels step 4 already wants, where 20 samples suffices. PSD needs
band-limiting to the 3–12 Hz bins and a gate that ramps with the mask (`mean(step_mask)`, or gating
once `L >= 10`) before any weight is set. Note the Welch segmentation itself is sound: `nperseg` is
`horizon`, so segments always align with rollout boundaries.

**Success criterion:** the MPC rediscovers "trigger early, then hold" on its own, and beats the
threshold controller on **duty cycle** (95 % is poor) rather than on suppression count — 6/7 is
probably near the ceiling given the bimodality.
