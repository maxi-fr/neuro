# Why the MPC does not suppress, and what to change

**Date:** 2026-08-10, updated 2026-08-11
**Context:** closes the last open item on the `roast_3d` closed loop.
Evidence and full result tables: [`tes_field_geometry.md`](tes_field_geometry.md) §2.
Artifacts, commands and traps: [`roast_operations.md`](roast_operations.md).

## Update 2026-08-11 — read this before the rest

Steps 4 and 5 were bypassed by a cheaper experiment: keep the current representation and extend the
horizon against the *existing* artifact. It failed. Every arm was then re-measured from scratch in
region space, at two run lengths, on one plant. **The premise of this whole document — "the MPC does
not suppress" — is too generous. The MPC spreads the seizure.**

All rows below: the `roast_3d` plant, the *identical* dynamics block across arms, seeds 69 and 1026,
scored on per-region LFP PTP over 76 regions at the 5 mV criterion — the instrument
`max_seizing_regions` names and
[`sweep_closed_loop.yaml`](../configs/nn_predictor/roast/sweep_closed_loop.yaml) calibrates
("suppressed basin is ~4-5 regions, unsuppressed ~27-35").

| arm | seizing regions, `t_end: 12` | seizing regions, `t_end: 20` | eegMS at 20 | duty | mean \|u\| (mA) |
| --- | --- | --- | --- | --- | --- |
| no stimulation | 5, 6 | 31, 34 | 84.7, 79.2 | 0 | 0 |
| threshold controller | 4, 5 | **4, 4** | **5.8, 6.4** | 0.951 | 1.27 |
| nonlinear MPC, `horizon: 20` | **35, 26** | 32, 32 | **117.2, 159.4** | 0.993 | 1.04 |

**The MPC does not merely fail to suppress — it accelerates the spread.** At `t_end: 12` it leaves
26-35 regions seizing where doing nothing leaves 5-6: it recruits roughly **six times more tissue
than leaving the plant alone**, while stimulating at 98.8 % duty. By 20 s the untreated seizure has
spread on its own, so the counts converge (32, 32 against 31, 34) and only the 1.4-2.0x higher scalp
power still separates them. What the MPC costs is the eight seconds in between.

**Score at `t_end: 20`, not 12.** At 12 s the two arms that differ most — no stimulation and the
threshold controller — are one region apart (5 vs 4), because seed 69 has not spread yet; the metric
has nothing to measure. It only discriminates once spread completes, where it separates cleanly by
27 regions. This **overturns the recommendation in step 1 below** to move the closed-loop eval to
`t_end: 12`, which was argued on `eegMS` and duty-cycle grounds before region scoring existed. The
12 s column is still worth keeping for the MPC, which is caught out there at 40 % of the cost.

**`eegMS` must never be compared across run lengths.** The same unstimulated arm reads 6.37 at 12 s
and 84.7 at 20 s. Most of the confusion in the tables below is this one effect.

Three earlier variants are quoted on `eegMS` alone, since they were measured before region scoring
existed and have not been re-run: `horizon: 20` terminal-only cost 133.3, and `horizon: 150` 146.1,
against 32.8 for the shipped `horizon: 20` — all at `t_end: 12`, all far above the unstimulated 6.37.

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
as an identification failure, correctly. The lone 1/7 at `w_u = 0` is one draw into the bimodal
suppressed basin, not a result.

**The `w_u = 10` row is retracted (2026-08-11).** Its "57 % below baseline on half the command
effort — minimising its objective and succeeding" reading does not survive re-measurement: on seeds
69 and 1026 at the same `t_end: 20`, the nonlinear MPC scores eegMS 117.2 and 159.4 against 84.7 and
79.2 unstimulated, i.e. *above* baseline, on mean `|u|` 1.04 rather than 0.67. It fails the same way
the linear one does, less severely. The region counts in this table do reproduce (30 -> 31/34
unstimulated, 4 -> 4 threshold), so the "seizing L" column is sound and the eegMS column is a
7-seed median that should not be read against any 2-seed number above.

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

**Ordering, revised again 2026-08-11: step 1, then step 5, then step 4's `downsample` route.**
Step 4's cheap variant has been
tried and failed (see step 4); the failure was not about lookahead, so the ordering below is
otherwise intact. Step 1 is now backed by direct evidence rather than inference, and is much cheaper
on this branch than costed below: staying at `downsample: 100` with 20 s trajectories needs ~22
trajectories and ~4.5 GB, not 690 and ~57 GB. Note the base config
`jansen_rit_seizure_excited.yaml` selects a **25-channel** montage while the roast predictors are
62-channel — regenerating off it as-is produces an untrainable set.

**Superseded ordering (2026-08-10), kept for the reasoning.** Steps 2 and 3 are done; 1, 4 and 5
remain. Step 3's answer reshuffled the
rest. Step 2 landed first and turned out to be a hard prerequisite rather than a hygiene fix, since
the `downsample` step 3 demands aliases 20–40 % without it — and it uncovered a silent break in the
excitation data pipeline (`controller.u` unlogged) that would have wasted step 1's entire generation
run. Step 5 is promoted from optional to mandatory and should be settled *before* step 1 generates
anything, since it decides what the dataset is even a dataset of. **Step 1 has moved from "start
now" to last**: its hold grid, its trajectory count, and its window geometry are all functions of a
`downsample` step 3 just changed, and generating first means generating twice — at ~57–98 GB a throw.

One practical note before any of it: **the original `data/experiment_excited_roast/` is not on this
machine** — `data/` holds only the leadfield and geometry files. Any instruction here that reads
"compare against the existing training set" needs regenerating it first.

**1. Regenerate the excitation set: sustained holds, and coverage of the transition.**
`hold_ms: [10, 50, 200]` means the longest constant command in training is 200 ms, while the working
policy holds for seconds at 95 % duty. This is a *prerequisite for step 4*, not the root cause: at a
200 ms horizon the gap does not bind; past 500 ms the model would extrapolate on every solve.
Changing the data alone would not have moved the closed loop.

The grid is `[dt_mpc, 200, 2000]` ms — short holds down to whatever the MPC step allows, one long
hold added, and nothing above 2000 ms. Three constraints fix it:

* **Longest = 2000 ms** (revised — see below). The original grid capped at 1000 ms because that
  exceeded the longest lookahead then on step 4's table, 0.80 s at `downsample: 400`, which is all
  the model needs in order not to extrapolate within a single solve. **Step 3 moved the target
  lookahead to 1.5 s**, so a 1000 ms cap is now *shorter than one solve's horizon* and the model
  would extrapolate on every solve — the exact failure the constraint existed to prevent. Raise the
  cap to 2000 ms. The old objection to 2000 ms was that it eats the trajectory, and it does: one
  such block is a quarter of an 8 s run. That is affordable only because of the inverse-length
  weighting below, which is what keeps long blocks rare in *time* while still drawing them; without
  it, a 2000 ms entry would take **88 %** of every trajectory. 4000 ms remains excluded — half a run
  in one block, and past the 1.5 s the probe says is the ceiling on useful lookahead anyway.
* **Shortest = the MPC step**, `dt_mpc = downsample × 1e-4`. Anything below it is a command the
  controller cannot issue and the strided training sample cannot represent — the logged `u` at the
  kept sample stands for a value the plant only held for part of the step. This is step 2's problem
  on the input channel: striding `u` aliases exactly as striding `y` does, and the fix there is a
  grid floor rather than a filter only because `u` is ours to choose. At the current
  `downsample: 100` the floor is 10 ms, unchanged; **at step 3's 500/750 it is 50 or 75 ms** — which
  swallows the existing 10 ms *and* 50 ms entries, leaving a grid of roughly
  `[dt_mpc, 200, 2000]`. **This couples step 1 to step 4 hard enough that step 1 should not be run
  first**: fix `downsample` before generating, or accept regenerating after. Since step 3 has now
  moved `downsample` twice, generate against the final value.
* **Weight the draw `p ∝ 1 / hold_steps`** (two lines at
  [`control.py:211`](../src/neuro/control.py#L211)). `build_input_schedule` currently draws block
  lengths uniformly over the *values* in `hold_ms`, so a value's share of total time is proportional
  to its own length: on `[75, 200, 2000]` the 2000 ms hold would take **88 %** of every trajectory
  and the 75 ms hold **3 %**, which is the same skew the old `[10, 50, 200]` grid had (200 ms took
  77 %) only far worse. Weighting by the inverse length splits time evenly — an equal share per
  entry — out of block draws that are overwhelmingly short. The weighting is what lets the grid be
  widened at all, and it is now load-bearing rather than a tidy-up: without it the 2000 ms entry
  step 3 forces would leave almost no short-hold data in the set.

The same regeneration should fix what the set spends its samples on. Over the 22
`experiment_excited_roast/train` trajectories only 9 show a clear seizure onset (plateau ≥ 3×
baseline EEG PTP), and among those the envelope reaches 90 % of its plateau at a median of **3.0 s**
[p10 2.2, p90 4.0] — leaving ~85 % of each 20 s trajectory in saturated, stimulation-insensitive
dynamics. Stimulation is injected throughout the excitation runs, which confounds the envelope, so
read these as indicative rather than exact; the direction is not in doubt. Under plain MSE that
saturated majority is where the capacity goes. Generate a **new dataset with shorter end times**,
trajectories ending around the transition rather than at 20 s, rather than reweighting the existing
one. That also buys more onset draws, worth having when only 9 of the current 22 produce one.

**Size it in training windows, not in seconds.** `build_dataset_for_trajectory`
([`nn_training.py:138`](../src/neuro/nn_training.py#L138)) yields `T_src − 34` windows per
trajectory at `n_y: 15` / `horizon: 20`, and both terms move under this change: coarsening
`downsample` shrinks `T_src`, and shortening trajectories pays the 34-sample edge loss far more
often. Today's set is 22 × 20 s = 440 s = **43.3 k windows**. Holding 440 s constant would *not*
hold that — and **step 3 makes this much worse than first costed**, because the edge loss scales
with `dt_mpc` and step 3 pushed `dt_mpc` from 25 ms to 50–75 ms:

| plan | trajectories | sim seconds | windows/traj | edge loss | windows | vs today |
| --- | --- | --- | --- | --- | --- | --- |
| today (20 s, `downsample: 100`) | 22 | 440 | 1966 | 1.7 % | 43.3 k | 1.0× |
| 8 s, `downsample: 250` (superseded) | 300 | 2400 | 286 | 10.6 % | 85.8 k | 2.0× |
| 8 s, `downsample: 400` (superseded) | 520 | 4160 | 166 | 17.0 % | 86.3 k | 2.0× |
| 8 s, `downsample: 500`, target | **690** | 5520 | 126 | **21.3 %** | 86.9 k | 2.0× |
| 8 s, `downsample: 750`, target | **1200** | 9600 | 72 | **32.1 %** | 86.4 k | 2.0× |

An 8 s trajectory at `downsample: 750` is only 106 samples long, of which the `n_y: 15` /
`horizon: 20` window geometry discards 34 — a third of every run is edge. That is the dominant cost
of step 3's result on the data side, and it is worth asking whether `n_y` should shrink with `dt`
before generating: 15 lags at 75 ms is 1.1 s of history, where 15 lags at 10 ms was 150 ms. Halving
`n_y` would recover most of the loss. **This is an open question, not a decided plan.**

Generate against the final `downsample`, not before it is fixed: `scripts/generate_experiment.py
--n-trials 690` (or 1200) off a base config with `t_end` and the controller's `duration` both at
8.0, plus ~20 % more on a disjoint seed range for test. Generation is no longer the free part —
measured at 1.47 s wall per simulated second, 5520 s of trajectory is ~2.3 h single-core and ~45 min
at 3 workers; 9600 s is ~3.9 h and ~1.3 h.

Two costs that scale worse. **Disk:** trajectories are stored at the full 10 kHz whatever
`downsample` later does with them, so an 8 s run is ~82 MB — **~57 GB at 690 trajectories and ~98 GB
at 1200** (`--compress` helps; the existing 22 × 20 s set is ~4.5 GB). At 1200 this stops being a
footnote and needs a decision before the run starts. Nearly half of it is avoidable — under
`IdentityEstimator` the
logged `estimator.x_hat` is a byte-for-byte duplicate of `sensor_0.y_mea`, 40 MB of every 82 MB
file. It stops being redundant once step 2 puts a filter in that slot, but for excitation runs, where
training filters at load time, it is pure duplication. **RAM:** `prepare_datasets` materialises
`X_full` and `Y_full` as float64 with `jax_enable_x64`; at 2× windows that is ~1.6 GB resident
before training starts, and epoch wall time scales with it.

`transient_ms` has been dropped from all four excitation configs, so the full trajectory is active.
It was there to protect against a settling artifact that `initial_state: rest` already removes:
measured over 4 s of zero-stimulation from the rest state, per-100 ms channel RMS sits inside its
steady band (2.39 ± 0.63) from 100 ms onward, with no decay to wait out. The one residual is the
opposite of a transient — the first 100 ms reads *low* (1.17) while noise variance accumulates from
a deterministic fixed point. Worth knowing when trajectories are only 8 s: the first ~10 MPC steps
of every run are quieter than the plant's steady behaviour.

**Keep the closed-loop eval at `t_end: 20`.** This paragraph previously argued for 12 s, and the
measurement at the top overturns it: in region space, 12 s cannot tell no stimulation (5, 6) from
the threshold controller (4, 5), because the untreated seizure has not spread yet. The argument for
12 s was that `mean_amplitude` averages `|u| / u_max` over *every* logged step, so a model
identified only on transitions and commanding garbage from 12–20 s corrupts the duty-cycle
tie-break, while `n_seizing()[-1]` is an end-state verdict that shrugs it off. That effect is real
but costs less than losing the primary metric; if the tie-break needs protecting, window
`mean_amplitude` rather than shorten the run. The saving was ~40 % of the ~20 min/seed runtime.

**2. DONE — causal anti-alias low-pass, in training *and* in the loop.** `load_trajectory` decimated
by bare striding with no filter anywhere. It now low-passes at the decimated Nyquist before striding,
and the *identical* filter runs online. Shipped in
[`filtering.py`](../src/neuro/filtering.py): `design_antialias_sos(fs, downsample)` (4th-order
Butterworth, `output="sos"`) is the single design both paths call, so they cannot drift;
`antialias_filter` does the offline half and `AntiAliasEstimator` the online half, both starting
`sosfilt` from zero state. The estimator runs at the plant `dt` (1e-4), filtering at the full 10 kHz
rate, and the controller's ZOH picks off every `downsample`-th filtered sample — which *is* "filter,
then stride". It is `sosfilt`, not `sosfiltfilt`: `scipy.decimate` defaults to `zero_phase=True`,
which cannot be reproduced online, so using it would bake in a train/serve skew that reads as model
error. The new estimator went into the MPC configs alone; the threshold and no-stimulation runs stay
on `IdentityEstimator`, so the baselines remain comparable. **Trap:** this edit initially added the
filter to both MPC configs while leaving them pointed at artifacts identified on bare-strided data,
so every closed-loop run between 2026-08-10 and 2026-08-11 fed the model a signal carrying 8.4 ms of
group delay it was never fit on. The nonlinear config now points at
`artifacts/roast/nonlinear_full_lowpass`; the linear one is back on `IdentityEstimator`, since no
filter-trained linear artifact exists and a mismatch is worse than no filter. Scoring is unaffected
either way — `closed_loop_eval` scores the plant's region LFP, not the estimate.

**The A/B says it is free; it is not, and the "1 %" framing was misleading.** Retrained at
`downsample: 100` with the filter on vs off — same 22 trajectories, same seed, same config, 3
held-out test trajectories, per-step NMSE in raw EEG units:

| horizon step | 1 | 5 | 10 | 15 | 20 | avg |
| --- | --- | --- | --- | --- | --- | --- |
| filter off | 0.0322 | 0.1627 | 0.1993 | 0.2350 | 0.2535 | 0.1897 |
| filter on | 0.0320 | 0.1627 | 0.1992 | 0.2348 | 0.2533 | **0.1896** |

No regression at any horizon — marginally better everywhere, and better than the shipped
`nonlinear_full` artifact on the same held-out data. But the "alias is only 1 % at `downsample: 100`"
figure describes the wrong quantity. Zero-phase filtering moves the decimated signal by 0.75 % RMS,
matching that number; the **causal** filter moves it by **24.9 %**, almost entirely group delay
rather than alias. Cross-correlation puts the delay at **8.4 ms** — 0.84 samples of the 100 Hz
decimated grid — and shifting by one decimated sample collapses the difference to 4.9 %. The delay
is identical on both paths, so it belongs to the identified plant exactly as intended and costs
nothing measurable. Anyone reading "1 %" as "the training signal barely moves" will be wrong by 25×.

Artifacts: `artifacts/roast/nonlinear_full_lowpass` (keep) and
`artifacts/roast/nonlinear_full_nofilter_ab` (the control); `artifacts/roast/nonlinear_full`
untouched. `downsample` left at 100 everywhere — moving it is step 4.

**Fixed in passing: `controller.u` was not being logged at all.** `WaveformController` returned
`NoLog()`, and `simulate`'s logger records only the *fields* of a component's log dataclass, so a
field-less log model is skipped wholesale. Excitation trajectories therefore contained no `u`, and
`load_trajectory` — which identifies the predictor against `controller.u` — would `KeyError` on any
freshly generated set. The excitation data pipeline was broken at HEAD, silently, since the simulate
upgrade in `e8f6097`. `WaveformController` now carries a `WaveformControllerLog` with `u`, covered
by a test. **This blocked step 1 outright** and is the first thing to verify if a regenerated dataset
misbehaves. Note `data/experiment_excited_roast_regen` (~7 GB, generated during the A/B) predates the
fix and has no `controller.u`; it is not trainable as-is and step 1 will supersede it anyway.

**3. DONE — the payoff crosses over between 0.8 s and 1.0 s.** The 5-seed `u = 0` vs `[+2, 0, -2]`
mA probe was rerun at 0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.5 / 2.0 s
([`probe_payoff_crossover.py`](../scripts/probe_payoff_crossover.py), 353 s for 10 runs). It
reproduces the two points in Findings — 0.2 s: 1.027 vs 1.032, 1/5; 2.0 s: 0.969 vs 0.968, 3/5 — so
it is the same probe, now resolved:

| lookahead | median good/zero | improves in |
| --- | --- | --- |
| 0.2 s | 1.027 | 1/5 |
| 0.4 s | 1.011 | 1/5 |
| 0.6 s | 1.019 | 1/5 |
| 0.8 s | 1.016 | 1/5 |
| 1.0 s | **0.997** | **3/5** |
| 1.5 s | 0.967 | 3/5 |
| 2.0 s | 0.969 | 3/5 |

**This is a plateau, not a ramp.** The median never once dips below 1 across 0.2–0.8 s and the
improving count is pinned at 1/5, then both move together at 1.0 s. Nothing is gained by landing
just short of the crossover — 0.8 s is not a near miss.

Per-seed, the answer is the same but sharper. Only seeds 69 and 1026 are still responsive at
t = 12 s; both first pay off at **1.0 s** (0.687, 0.987) and only become decisive at **1.5 s**
(0.078, 0.678):

| seed | 0.2 s | 0.8 s | 1.0 s | 1.5 s | 2.0 s |
| --- | --- | --- | --- | --- | --- |
| 69 | 1.035 | 0.902 | **0.687** | **0.078** | **0.048** |
| 1026 | 1.196 | 1.016 | 0.987 | **0.678** | **0.139** |
| 1023 | 1.011 | 1.033 | 0.997 | 1.043 | 1.014 |
| 1024 | 0.989 | 1.014 | 1.034 | 1.020 | 1.042 |
| 1025 | 1.027 | 1.045 | 1.016 | 0.967 | 0.969 |

Seed 69's 0.968 at 0.4 s is inside the ±3 % jitter seeds 1023–1025 show in both directions; it is
not an early crossing. Seeds 1023–1025, fully seizing at 12 s, never benefit at *any* lookahead out
to 2 s — Findings' "the window to act closes" now holds across the whole grid, and those three sit
in the median as permanent 1.0× ballast. There is no gain past 1.5 s (0.967 → 0.969).

**What this costs the plan.** Steps 4 and 5 were written expecting the flip somewhere below 0.8 s.
It is not, so `downsample: 250` (0.50 s) and `400` (0.80 s) are both *inside the flat wrong-sign
plateau* — an MPC at either horizon still refuses to stimulate, for exactly the reason it refuses
now. The mildest setting that clears is `downsample: 500` (1.0 s at `horizon: 20`, `dt_mpc` 50 ms),
and it clears by nothing at all (median 0.997); for a payoff big enough to outweigh a `w_u = 10`
command penalty the target is `downsample: 750` (1.5 s, `dt_mpc` 75 ms). Both are past 400, which is
the condition step 5 was made conditional on — **step 5 is now mandatory, not optional.**

**4. Extend the horizon. TESTED 2026-08-11 the cheap way — it made the closed loop worse.**

Before paying for a coarser `downsample`, the free version was tried: keep `downsample: 100`, keep
the representation, keep the shipped artifact, and just raise `horizon` to 150 (1.5 s, past the
step 3 crossover). The one-line objection below — "the predictor is trained to a 20-step rollout" —
was measured rather than assumed, with
[`probe_rollout_horizon.py`](../scripts/probe_rollout_horizon.py) on 4 held-out 62-channel seeds
(5000-5003) pushed through the exact training decimation path:

| step | lookahead | NMSE, RAS input (10-200 ms holds) | NMSE, sustained input (2000 ms holds) |
| --- | --- | --- | --- |
| 1 | 0.01 s | 0.0299 | 0.0365 |
| 20 | 0.20 s | 0.2476 | 0.2666 |
| 40 | 0.40 s | 0.2979 | 0.3499 |
| 80 | 0.80 s | 0.3147 | 0.4264 |
| 100 | 1.00 s | 0.3241 | 0.4543 |
| 150 | 1.50 s | 0.3252 | **0.4877** |

Steps 1 and 20 of the RAS column reproduce step 2's filter-on A/B (0.0320, 0.2533) on different
held-out data, which validates the probe. **On RAS inputs the error saturates at ~0.32 from 0.6 s
and is flat to 1.5 s** — read alone, that licenses `horizon: 150` against the existing artifact with
no retrain, which is exactly how it was read, and that reading was wrong.

**The saturation is an artifact of the input distribution.** Re-run on excitation with `hold_ms:
[2000]` — the sustained regime the MPC actually commands — the error never saturates; it climbs
monotonically to 0.49 with no sign of flattening. The RAS curve certifies the model on inputs that
switch every 10-200 ms, and says nothing about the 1.5 s holds a long horizon will ask for. Caveat
on the comparison: `hold_ms: [2000]` also redraws amplitude only every 2 s, so within-trajectory
input variety drops and the plant sits in sustained states longer; part of the gap is a different
state distribution rather than hold length as such.

**Closed loop at `horizon: 150` (`shooting_depth: 150`, seed 69, `t_end: 12`): eegMS 146.1 against
32.8 at `horizon: 20` and 6.37 unstimulated.** Mean `|u|` rose 0.83 -> 1.23. It did not rediscover
"trigger early, then hold"; it found something confidently wrong. Cost: 3.6 h wall, median solve
7.24 s at 22 IPOPT iterations, peak RSS 338 MB, solver build 2.9 s.

**A model at NMSE 0.49 does not explain a 23x closed-loop degradation.** That gap is the real
lesson: the MPC is not suffering average-case error, it is selecting commands that exploit the
directions the model is wrong in. No average-case NMSE on randomly-drawn inputs — any hold length —
certifies a controller that optimises against the model. Step 1's sustained-hold data is confirmed
necessary and shown to be *insufficient*.

**Trap when raising `horizon`: `shooting_depth` must move with it.** The configs pin
`shooting_depth: 20`, which is single shooting only because it equals `horizon: 20`. Left at 20 with
`horizon: 150` it silently becomes 7-segment *multiple* shooting — the intractable case retracted in
Findings — and reads as "the long horizon is impossible". Set `shooting_depth == horizon`.

`w_u` needs no rescaling when the horizon changes: both cost terms sum over the same `h` steps, so
the y:u budget is preserved.

**The `downsample` route below is unchanged but now second in line**, since the failure above is not
about lookahead. Not by raising `horizon` to 200 — the NLP grows 10× and the predictor is trained to
a 20-step rollout. Retrain at a coarser `downsample` keeping `horizon: 20`: same NLP size, more
lookahead. But the coarsening is not free. Measured on `sim_000.npz` (62 ch, 20 s), bare striding
against `scipy.decimate`:

| `downsample` | fs | Nyquist | 20-step lookahead | alias RMS |
| --- | --- | --- | --- | --- |
| 100 (current) | 100 Hz | 50 Hz | 0.20 s | 1.0 % |
| 200 | 50 Hz | 25 Hz | 0.40 s | 2.2 % |
| 250 | 40 Hz | 20 Hz | 0.50 s | 3.0 % |
| 400 | 25 Hz | 12.5 Hz | 0.80 s | 11.1 % |
| 1000 | 10 Hz | 5 Hz | 2.00 s | **53.9 %** |

66 % of EEG variance sits above 5 Hz — 60.9 % of it in the 5–12 Hz seizure band alone — so at
`downsample: 1000` the strided signal is majority fold-back, with its variance inflated 2.35×.

The original reading of this table preferred `downsample: 250` — 500 ms of lookahead at 3 % alias,
raw EEG intact, no representation change. **Step 3 kills that option.** 250 and 400 both sit inside
the wrong-sign plateau, so the cheap end of this table buys a model that forecasts further and still
declines to act. The choice is now between two rows that do not appear above:

| `downsample` | fs | Nyquist | lookahead | alias (interpolated) | verdict |
| --- | --- | --- | --- | --- | --- |
| 500 | 20 Hz | 10 Hz | 1.00 s | ~20 % | clears by nothing (median 0.997) |
| **750** | 13.3 Hz | 6.7 Hz | 1.50 s | ~35–40 % | the real target |

At both, a *correct* anti-alias filter (step 2) deletes most or all of the 5–12 Hz seizure band, and
striding without one folds 20–40 % of the variance back. Raw EEG is unrepresentable either way —
which is precisely step 5's premise, and why step 5 now ships in the same change rather than being
held in reserve. Do not retrain at 250 or 400 expecting a closed-loop result; the only reason to
touch them is as a staging point for validating step 2's filter.

**5. Forecast an envelope, not the waveform** — **mandatory**, since step 3 puts the working `dt` at
50–75 ms, past the ~40 ms threshold this was gated on. At
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

**Skip:** `w_du` / `w_u_l1`, bigger networks, PCA. The `w_u` sweep already showed the
controller side is not a lever, and the probe above shows why.

**Skip — solver tuning. MEASURED 2026-08-11, IPOPT was never the constraint.** `MPCControllerLog`
now carries `n_iter` and `capped` (`capped` is `Maximum_Iterations_Exceeded` specifically, so
success / capped / other-failure is a clean trichotomy; the status *string* is deliberately not
logged, because the logger sizes each buffer from the first value's dtype and would silently
truncate a longer status to the first one's width). Across every closed-loop run on 2026-08-11:
`horizon: 20` converges in a median of **8-14** iterations against `max_iter: 100` with **zero**
capped solves; `horizon: 150` takes a median of 22 with 0.34 % capped. The commands the MPC applies
are converged local minima, not truncated iterates. Solver truncation is eliminated as an
explanation for anything in this document.

**Skip — reshaping the cost inside the current horizon. TESTED 2026-08-11, it is harmful.**
`MPCController` gained `w_y_terminal`, which replaces `w_y` on the final horizon step (`w_y: 0` with
`w_y_terminal: h` costs the terminal step alone at matched total y-weight). Against the uniform arm
at `horizon: 20`, `t_end: 12`, seeds 69 and 1026: eegMS 32.8 -> 133.3 and 104.0 -> 107.4, with mean
`|u|` rising 0.83 -> 1.12 and 0.96 -> 1.07. It spends more command and does more damage. This is
what step 3's plateau predicts — at `horizon: 20` the entire 0.2 s window is inside the wrong-sign
region, so no reweighting of it can recover the sign — but note what is new: the plateau was
measured *open-loop* with a fixed known-good command, and this is the closed-loop confirmation that
a better-shaped cost over the wrong window cannot help. Competing explanation not ruled out at
n=2: with `w_y = 0` the objective sees one step instead of a 20-step sum and is much noisier, so
some of the degradation may be variance. Both readings point the same way.

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

Revisit only after step 4. FC becomes estimable once the effective window lengthens, or restricted
to the `lTCI`-loading channels step 5 already wants, where 20 samples suffices. PSD needs
band-limiting to the 3–12 Hz bins and a gate that ramps with the mask (`mean(step_mask)`, or gating
once `L >= 10`) before any weight is set. Note the Welch segmentation itself is sound: `nperseg` is
`horizon`, so segments always align with rollout boundaries.

**Success criterion, revised 2026-08-11.** The original bar — the MPC rediscovers "trigger early,
then hold" on its own and beats the threshold controller on **duty cycle** (measured 95.1 %, poor)
rather than on suppression count, since 4/76 regions is near the ceiling — is the *second* bar, not
the first. Every MPC arm measured is worse than not stimulating, so the first bar is **beating
`ZeroController` on seizing regions at `t_end: 20`**: below 31-34, against a threshold controller
that reaches 4. Anything that only lowers `eegMS` without moving the region count is not
suppression.
