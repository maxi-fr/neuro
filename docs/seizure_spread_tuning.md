# Tuning the Seizure Spread: `K` and `sigma`

How the global coupling `K` and the noise std `sigma` were tuned so the EZ/PZ seizure
*propagates* over ~10 s — igniting in the EZ, reaching the PZ, then taking the left
hemisphere — instead of starting everywhere at once, which is what the previous settings did.

Source of truth:

- Knobs: [`src/neuro/connectome.py`](../src/neuro/connectome.py) (`K`),
  [`src/neuro/jansen_rit.py`](../src/neuro/jansen_rit.py) (`sigma`, `resting_state`)
- Measurement: [`src/neuro/seizure.py`](../src/neuro/seizure.py) (`spread_profile`, `spread_summary`)
- Search: [`scripts/sweep_seizure_spread.py`](../scripts/sweep_seizure_spread.py)
- Verification: [`notebooks/seizure_spread_search.py`](../notebooks/seizure_spread_search.py)
- Result: every config in [`configs/simulation/`](../configs/simulation/), documented in
  [`jansen_rit_seizure.yaml`](../configs/simulation/jansen_rit_seizure.yaml)

---

## 1. Result

| setting | before | after |
| ------- | ------ | ----- |
| `connectome.K` | 0.5357 | **0.60** |
| `params.sigma` | 500 (`JansenRitParams` default) | **280** (explicit in every config) |
| `dynamics.initial_state` | zeros (implicit) | **`rest`** |

Measured over 10 noise seeds of 30 s each, using the recruitment detector of §3:

| quantity | before | after |
| -------- | -----: | ----: |
| EZ recruited | ≤ 0.5 s (all seeds) | ≤ 0.5 s (all seeds) |
| PZ recruited, median (range) | 0.5 s (0.5 – 0.5) | **3.3 s (0.5 – 10.3)** |
| half the left hemisphere, median (range) | 0.5 s (0.5 – 0.5) | **6.3 s (0.8 – 15.8)** |
| 90 % of the left hemisphere, median (range) | 2.3 s (1.3 – 3.8) | **7.4 s (1.0 – 16.5)** |
| left hemisphere finally recruited | 92 – 95 % | 92 % (every seed) |
| right hemisphere finally recruited, median (max) | 8 % (**89 %**) | **0 % (5 %)** |
| EZ recruited before PZ | 9 / 10 seeds | **10 / 10 seeds** |

The whole propagation used to be over inside the first measurement window. It now unfolds on
the intended ~10 s scale, in the intended order, and stays in the left hemisphere — the old
setting went fully bilateral on one seed in ten.

The third row of the first table is not one of the two knobs that were asked about, but no
value of `K` or `sigma` does anything without it. See §4.

## 2. What the two knobs actually control

An isolated, *noiseless* Jansen–Rit node behaves like this:

| `A` | role | isolated node, `K = 0`, `sigma = 0` |
| --- | ---- | ----------------------------------- |
| 3.6 | EZ | oscillates at 13.2 mV peak-to-peak |
| 3.4 | PZ | flat (0.00 mV) |
| 3.25 | healthy | flat (0.00 mV) |

So the EZ is *past its own bifurcation*: it seizes on its own, within a few hundred ms,
whatever `K` and `sigma` are. **Neither knob can delay the seizure's onset**, and neither
should — the EZ igniting first is the definition of the regime.

Everything else is subcritical and can only be recruited by drive arriving from outside:

- **`K`** scales the deterministic, structure-following drive `K Σ_j w_ij S(y_j)`. Once a
  neighbour is seizing, `K` decides whether its output is enough to push a region over its own
  threshold. It follows the connectome, so it sets *which* regions go and in what order.
- **`sigma`** is the stochastic drive. A region sitting just below threshold escapes to the
  ictal limit cycle when a noise excursion carries it there — a Kramers-type escape, whose
  *waiting time* falls steeply as `sigma` rises. It sets *how long* recruitment takes.

Both feed the same sum, so they trade off along a ridge, which is exactly what the sweep shows
(§5). But they are not interchangeable: only the stochastic part buys a *slow* spread. Turning
`K` up far enough to recruit on coupling alone recruits everything at once.

## 3. Measuring "spread"

`spread_profile` (in [`src/neuro/seizure.py`](../src/neuro/seizure.py)) runs the plant and
reduces it to *when each region was recruited*:

- The peak-to-peak swing of each region's `y = x2 - x3` over a **1 s window**, hopped by
  0.25 s and timestamped at the window centre. The ictal limit cycle runs at 2–5 Hz, so 1 s
  always spans at least two cycles.
- A region counts as **seizing** above **5 mV** peak-to-peak. The median background is ≤ 2 mV at
  every `sigma` swept and the ictal cycle is ~14 mV, so one absolute cut is valid across the
  sweep — no per-`sigma` recalibration that could bias the comparison.
- Its **onset** is the first window that begins **1 s of uninterrupted** supra-threshold
  activity. The persistence rule matters: without it, single-window noise bursts and the
  settling transient of §4 both read as onsets, and every region looks recruited at t = 0.

`spread_summary` reduces that to five numbers — EZ onset, PZ onset, the time at which half the
left hemisphere is recruited, and the final extent in each hemisphere — and `SpreadSummary.score`
measures their distance from the target schedule (EZ ≤ 1.5 s, PZ ≈ 5 s, half the left hemisphere
≈ 10 s, ≥ 50 % of the left hemisphere, no right hemisphere).

Scores are averaged **per seed, then over seeds**, not computed from the seed-averaged schedule.
The two differ a lot here (§6), and only the former rewards a setting that hits the target on
*every* run.

## 4. The confound: the plant started at the origin

`JansenRitDynamics` defaults to an all-zero initial state, but the resting network sits at
`y ≈ 1.1 – 1.8 mV`. Relaxing from the origin therefore costs a synchronous, network-wide
voltage excursion of *ictal amplitude* in the first tens of ms — the network is kicked at
t = 0, and every region reads as seizing in the first window regardless of `K` and `sigma`.

This dominated the original complaint. The first sweep was run from zeros, and at the baseline
`K = 0.5357` the EZ, the PZ *and* 80–95 % of the left hemisphere were recruited inside the very
first window at every `sigma` from 250 to 550 — the knobs did nothing, because the startup
transient had already done the work. The 10-seed baseline row in §1 is the same story: `t_pz`
and `t_left_half` are 0.5 s on all ten seeds.

`resting_state` (in [`src/neuro/jansen_rit.py`](../src/neuro/jansen_rit.py)) fixes this: it
relaxes the healthy network noiselessly for 2 s and returns the fixed point (2 s is ample —
the slowest synaptic time constant is `1 / b = 20 ms`). Configs reach it with
`dynamics.initial_state: rest`, which every config in `configs/simulation/` now sets; the code
default stays `zeros`, so plants constructed directly in tests and notebooks are unaffected.

**Any experiment that measures *when* something happens needs this** — that includes stimulation
response timing, not just seizure onset.

## 5. The search

`scripts/sweep_seizure_spread.py` runs a `K × sigma` grid with several noise seeds per cell,
stores the full amplitude envelope, and ranks the cells. Four grids were run from rest, after
the zeros-start grid of §4:

| grid | `K` | `sigma` | seeds × duration | purpose |
| ---- | --- | ------- | ---------------- | ------- |
| coarse | 0.45 – 0.58 | 100 – 550 | 2 × 20 s | locate the transition |
| search | 0.55 – 0.60 | 280 – 400 | 3 × 20 s | map the ridge |
| confirm | 0.58 – 0.60 | 280 – 340 | 6 × 25 s | separate the top cells |
| ridge | 0.61 – 0.62 | 240 – 265 | 6 × 30 s | test going further |

The landscape (`notebooks/seizure_spread_search.py`, first figure) is a clean anti-diagonal
ridge — `K` and `sigma` buy the same drive — with three regions:

- **`K` ≤ 0.57, low `sigma`** — the seizure never leaves the EZ. Only the 3 EZ regions and
  occasionally one PZ region seize; `t_left_half` never arrives.
- **`K` ≥ 0.61** — the coupling alone is supercritical for healthy tissue. At `K = 0.61` and
  0.62, *both* hemispheres (88–89 % of the right) are recruited inside the first window, at
  every `sigma` tested down to 240. `sigma` no longer matters at all.
- **`K` ∈ [0.58, 0.60], `sigma` ∈ [280, 400]** — the usable band, where the coupling is just
  subcritical and recruitment waits on a noise excursion.

Inside the band, the seizure's *extent* turns out to be all-or-nothing: a run either stays in
the EZ (13–29 % of the left hemisphere) or takes essentially the whole left hemisphere
(92–95 %), with nothing in between. What `K` and `sigma` really tune is **how long the network
waits**, not how far the seizure gets.

## 6. Choosing the point, and the slowness/reliability trade-off

The scores of the top cells sit within noise of each other, so the choice was made on the
per-seed behaviour instead (confirm grid, 6 seeds; "reliable" = ≥ 90 % of the left hemisphere
recruited on *every* seed):

| `K` | `sigma` | reliable | median time to half the left hemisphere |
| --- | ------- | -------- | --------------------------------------- |
| 0.58 | 280 | 1 / 6 | never (majority of seeds) |
| 0.58 | 310 | 1 / 6 | never (majority of seeds) |
| 0.58 | 340 | 5 / 6 | 9.6 s |
| 0.59 | 280 | 3 / 6 | never (majority of seeds) |
| 0.59 | 310 | **6 / 6** | 7.8 s |
| 0.59 | 340 | **6 / 6** | 4.1 s |
| **0.60** | **280** | **6 / 6** | **7.5 s** |
| 0.60 | 310 | **6 / 6** | 4.5 s |
| 0.60 | 340 | **6 / 6** | 2.6 s |

This is the central trade-off, and it is intrinsic rather than a defect of the search: **a
slower spread means sitting closer to the escape threshold, and closer to the threshold the
escape sometimes does not happen at all.** The cells that spread more slowly than `K = 0.60,
sigma = 280` are precisely the ones that fail to spread on some seeds.

`K = 0.60, sigma = 280` is the chosen point: the slowest cell that recruits the left hemisphere
on every seed, at the lowest `sigma` among the reliable cells, and with `K` as high as it can go
before the bilateral cliff at 0.61 — which keeps the propagation route as structure-driven, and
therefore as reproducible, as the regime allows.

**The schedule is still a random variable.** Over 10 seeds, half the left hemisphere is reached
anywhere between 0.8 s and 15.8 s (median 6.3 s). That spread is the physics of a noise-driven
escape, not noise in the measurement, and it cannot be tuned away — only traded against
reliability by moving along the ridge. Runs that need a specific schedule should pin the seed.

`jansen_rit_seizure.yaml` does exactly that. Scanning seeds 50–61 at the chosen point:

| | seeds |
| --- | --- |
| half the left hemisphere within 4 s | 50, 51, 52, 53, 57, 58, 59, 61 |
| **≈ the target schedule (7 – 13 s)** | 54, **55**, 56 |
| never leaves the EZ | 60 |

Seed **55** is pinned there because it realises the target almost exactly: EZ at 0.5 s, PZ at
5.2 s, half the left hemisphere at 9.8 s, 92 % of the left hemisphere and none of the right.
The other configs keep the seeds their experiments already used.

## 7. Side effects

- **The healthy background is quieter.** Peak-to-peak of a healthy region drops from 1.6 mV
  (`sigma = 500`) to 0.9 mV (`sigma = 280`). This is a change to the model's "resting EEG"
  amplitude, which the paper's `sigma = 500` was originally chosen to set — see the
  `dt/sigma noise calibration coupling` note.
- **…and it no longer seizes by itself.** At `K = 0.5357, sigma = 500`, an *all-healthy*
  network (`A = 3.25` everywhere) still produces 0–2 regions that sustain > 5 mV over 20 s. At
  `K = 0.60, sigma = 280` it produces none across the same seeds. The new setting removes a
  false-positive source the old one had.
- **The run window went from 5 s to 20 s.** At 5 s the seizure only reaches onset plus early
  spread — the EZ ignites by ~0.5 s and the PZ follows at a median of 3.3 s, but half the left
  hemisphere is only reached at a median of 6.3 s. Every MPC and identification config therefore
  moved to `t_end: 20.0`, which covers the whole propagation on all but the slowest seeds. The
  `WaveformController.duration` in the excited configs moved with it, since the tES schedule must
  outlast the run.
- **Trained artefacts do not transfer.** `sigma` and `K` change the plant, so every NN predictor
  under `artifacts/` and every MPC benchmark predates the retune and was fitted to a different
  system; they must be refitted. The identification datasets were regenerated on 2026-07-30 as
  **25 trials × 20 s** (`data/experiment_excited{,_analytical,_reciprocal}`) — the same 500 s of
  data per set as the old 100 × 5 s, but with each trial spanning the full spread. Split 22 train
  / 3 test, preserving the old 90/10 ratio.
- **The old MPC benchmark numbers are not comparable.** The controller now faces a different
  problem: suppress a focus and block propagation, rather than damp an already-saturated network.
- **`data/experiment_excited_reduced_channels` was deleted and not regenerated**, so the
  `configs/nn_predictor/meeting_seven/{linear,nonlinear}_selected.yaml` predictors have no
  training data until a 25-channel set is rebuilt (the montage lives in
  `jansen_rit_seizure_excited.yaml`).
- **Disk is the binding constraint.** A 20 s trial writes 1.14 GB (the full `(T, 6, 76)` state
  array is 64 % of it), so 100 trials × 20 s × 3 sets would need ~340 GB. That is why the trial
  count dropped to 25 rather than the run length staying at 5 s.

## 8. Reproducing

```bash
# the grid behind §5 (writes artifacts/seizure_spread_<timestamp>/sweep.npz)
uv run python scripts/sweep_seizure_spread.py --k 0.58 0.59 0.60 --sigma 280 310 340 \
    --seeds 6 --duration 25

# the chosen point, and the old setting for comparison
uv run python scripts/sweep_seizure_spread.py --k 0.60 --sigma 280 --seeds 10 --duration 30
uv run python scripts/sweep_seizure_spread.py --k 0.5357 --sigma 500 --seeds 10 --duration 30 --from-zero

# inspect any of them (heatmaps, onset rasters, per-seed variability, live re-simulation)
uv run marimo edit notebooks/seizure_spread_search.py

# run the tuned seizure through the normal pipeline
uv run python scripts/run_simulation.py configs/simulation/jansen_rit_seizure.yaml
```

Each worker holds its own TVB dataset (~0.5 GB), so `--workers` is memory- rather than
core-bound; the default of 4 is deliberately conservative.
