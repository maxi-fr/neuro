# Handoff — predictability & controllability experiment design

**Repo:** `c:\Users\frank\closed-loop-neurostimulation` (branch `main`)
**Date of session:** 2026-08-13
**Session type:** `/grill-me` design interview. **No code was written and no files were changed.**

---

## Status — superseded, kept as the record of the design interview

**This handoff is historical.** The design was confirmed and is now implemented. The live document
is [`predictability_controllability_experiment.md`](predictability_controllability_experiment.md);
read that, not this, for anything current — the sections below describe the design as it stood
before implementation and a few of its numbers have since been *measured* and corrected there.

Both open points were ratified:

- Region-space labels use the existing `SpreadProfile` criterion unchanged (5 mV PTP, 1 s window,
  1 s persistence). **Confirmed.**
- Channel weighting (the doc's former "Open decision"): each predicted-tier metric is scored
  **twice**, over all 62 channels and over the four channels loading hardest on `lTCI`
  (`TP9`, `TP7`, `T7`, `P7`). **Confirmed.**

Shipped: `src/neuro/metrics.py` (registry **and** the scores), `src/neuro/ensembles.py`,
`scripts/run_predictability_experiment.py`, `notebooks/ensemble_explorer.py`,
`notebooks/metric_scoring.py`, plus `tests/test_windowed_metrics.py`,
`tests/test_metric_scores.py`, `tests/test_ensembles.py`.

Three things the implementation learned that this document gets wrong:

- **Cost.** The ~2.7 h estimate below was extrapolated from `probe_payoff_crossover.py`, which
  drives the full `Simulation` orchestrator. Measured on the real path: **0.46x realtime**, so
  ~31 min to generate. Scoring is the expensive half at ~43 min, and is cached.
- **Pairing.** The claim below that common random numbers makes the paired difference "far less
  noisy" holds only for `h` below ~0.5 s. By `h = 1 s` the ratio is ~1, and during ignition the
  paired spread is *larger*. `d_ctrl` is unaffected — it uses the unpaired `sigma_ens` by design.
- **Storage.** 19 GB is scalp only. Region LFP is stored as well, at 1 kHz float32 (+1.2 GB),
  because the observability axis needs region-space metrics.

---

## The design

### Framing

Predictability and controllability are treated as **independent axes**, not nested. The user's
opening proposal was "assume controllability if predictability is proven"; that was challenged with
repo evidence and the user accepted the two-axis framing. Each metric gets a coordinate on both
axes. The "predictable but uncontrollable" quadrant is an expected, publishable outcome.

Predictability is scoped to **intrinsic (model-free) predictability only** — seed ensembles, no
predictor training. This is therefore a metric-*selection* experiment; whether an MLP/ESN can
actually learn the winning metric stays with the existing predictor pipeline.

Controllability is measured by a **cheap open-loop probe**, not closed-loop MPC. This deliberately
avoids confounding metric quality with identification/solver failure.

### Plant and branch structure

Healthy and seizure plants differ **only** in the `A` vector (3.25 uniform vs 3.6 at EZ / 3.4 at
PZ). Everything else shared: `K: 0.60`, `sigma: 280`, `roast_3d` stimulation, `u_max: 2.0`,
`dt: 1e-4`, `initial_state: rest`. Pinned to the dynamics blocks of
`configs/simulation/jansen_rit_baseline.yaml` (healthy) and
`configs/simulation/nonlinear_full_mpc.yaml` (seizure).

**16 parent runs**: 8 healthy parents (distinct seeds) run to 4 s, 8 seizure parents run to 14 s.
Each seizure parent is snapshotted at all four seizure branch points, so one run serves four
branches.

| branch | plant | t_branch |
|---|---|---|
| healthy | A = 3.25 uniform | 4 s |
| pre-onset | EZ/PZ gains | 1 s |
| EZ ignited | EZ/PZ gains | 3 s |
| mid-spread | EZ/PZ gains | 7 s |
| saturated | EZ/PZ gains | 14 s |

Branch times follow the schedule encoded in `src/neuro/seizure.py` (EZ recruits ~1.5 s, PZ ~5 s,
left hemisphere half ~10 s).

**Per branch: 8 parents x 16 children x 2 arms**, 3 s rollout each.
**Total 1280 rollouts, ~2.7 h single-threaded** (see cost anchor below).

### Critical implementation gotcha — branching

"Same initial condition" is **not** just `x`. Per `src/neuro/jansen_rit.py` (~L344-L370) the plant
state is `x` (6, n_nodes) **plus** a `history` ring buffer of length `max_delay + 1` **plus** the
step counter `k` **plus** the RNG.

The public `initial_state=` constructor argument only sets `x` and fills `history` with a constant
sigmoid — correct from rest, **wrong for branching mid-seizure**.

Correct approach: `copy.deepcopy(dyn)` then reassign `dyn.rng = np.random.default_rng(child_seed)`.
Preserves `x`, `history` and `k` exactly, needs no API change.

### Stimulation arms

Two arms, **sharing child seeds** (common random numbers → paired differences):

- `u = 0`
- sustained hold `u = d1 = [2.0, 0.0, -2.0]` mA

`GOOD_COMMAND` is taken from `scripts/probe_payoff_crossover.py`. With `u_max = 2.0` it sits exactly
on the box corner and is zero-sum (KCL-valid).

The plant's noise is additive on `x5'` and state-independent (`src/neuro/jansen_rit.py:166`), so
common random numbers gives tight coupling between the two arms — this is what makes the paired
difference far less noisy than the unpaired one.

**The arm list must be a config parameter**, so `-d1` and a second basis direction `d2` can be added
later without redesign. This was an explicit concession — see "User overrides" below.

The `roast_3d` leadfield is `(3 electrodes, 76 regions, 3)`, so under KCL the admissible input space
is **exactly 2-dimensional**.

### Metrics

Computed on **raw 10 kHz scalp EEG** (62 channels — primary space, because a metric the controller
cannot measure is disqualified regardless of score). **Trailing/causal windows**, **50 ms hop**,
60-point grid over the 3 s rollout. Scores valid only for `h >= window`; the invalid region is
shaded in plots, not hidden.

**Predicted tier** (scored on all axes):

| metric | window |
|---|---|
| block PTP | 100 ms |
| 3-12 Hz band power | 500 ms |
| line length | 100 ms |
| synchronization R | 500 ms |
| eegMS | 100 ms |
| spectral centroid | 500 ms |

**Descriptive tier** (plots + separability only, never predicted at 62x62): FC matrices, full PSD,
topoplots, spread rasters. Scalar reductions of these (mean off-diagonal FC = `synchronization`,
spectral centroid) are what bridge into the predicted tier.

Rationale for the two-tier split: `src/neuro/nn_training.py` `_masked_fc` was measured to be "very
nearly a dataset constant" and unable to express `dFC/du`, because a 62x62 FC from a 20-sample
window has rank <= 19. An honest 62-channel FC needs ~5 s of window, which is not a control
timescale.

`src/neuro/metrics.py` should provide a uniform registry `(signals, fs, window, hop) -> (times,
values)` and **delegate** to the existing `src/utils/processing.py` (`compute_psd`, `band_energy`,
`synchronization`) rather than reimplementing them.

### Scores (per metric x branch)

- **Predictability** — `R2(h) = 1 - Var_within_ensemble / Var_across_parents`. Conditional variance
  reduction, scoped per phase. The across-parents denominator is *why* multiple parents exist.
  Also report raw `sigma_ens(h)` in native units with a **p5-p95** range reference (not min-max,
  which is non-robust and sample-size dependent) — the user asked for this explicitly.
- **Controllability** — signed `d_ctrl(h) = delta_bar(h) / sigma_ens(h)`, positive = toward healthy.
  The **unpaired** `sigma_ens` denominator is deliberate: a causal controller cannot know the noise
  realization, so the paired `sd(delta_i)` would credit effects no controller can act on.
  Secondary columns: paired `sd(delta_i)` for significance (distinguishes a true null from an
  underpowered one), and `delta_bar/Delta` for clinical magnitude.
- **Separability** — Cohen's d, healthy vs saturated, n = 8 per class. (Multiple parents are
  *required* for this to be defined at all; a single-parent design gives n = 1 per class.)
- **Observability** — per-metric scalp<->region correlation across the time grid. Region LFP is used
  only as the label/ground-truth generator, never as the primary space.
- **Feasibility** — minimum window length, as a hard practical filter. A 1 s-window metric is
  unusable at a 20 ms control rate no matter how well it scores.

Full **curves over h** are reported. `h_eval` defaults to **1.0 s** and must be adjustable (script
parameter + notebook slider). 1.0 s is not arbitrary: `probe_payoff_crossover.py` measured the
payoff sign flip between 0.8 s and 1.0 s, and characterized 0.2-0.8 s as "a plateau, not a ramp".

### Deliverables

| path | purpose |
|---|---|
| `src/neuro/metrics.py` | windowed metric registry, uniform signature |
| `src/neuro/ensembles.py` | parent runs, snapshot/branch, arm execution |
| `scripts/run_predictability_experiment.py` | generation CLI |
| `notebooks/ensemble_explorer.py` | trajectories, FC, PSD, topoplots, healthy vs seizure (visual tier) |
| `notebooks/metric_scoring.py` | R2, d_ctrl, separability table, 2D scatter |

Notebooks split **by purpose** (visual vs quantitative), not by experiment — splitting by experiment
would break the headline figure across two files, since it needs both axes.

**Storage:** full 10 kHz trajectories, ~19 GB (1280 x 30000 x 62 x 8 B). One npz per branch x arm,
opened with `mmap_mode='r'` (the repo already uses `use_mmap=True` in
`src/neuro/closed_loop_eval.py:101`). A **1 kHz derived cache built on first notebook run** keeps
interactive plotting responsive — 19 GB will not sit in a marimo kernel.

**Headline figure:** R2 vs signed d_ctrl scatter at `h_eval`, one point per metric, one panel per
phase, marker size = Cohen's d.

---

## User overrides (agent recommended X, user chose Y)

Respect these. They were put to the user with the trade-offs stated. Do not silently re-litigate.

| # | Topic | Recommended | User chose | Concern raised, and mitigation in place |
|---|---|---|---|---|
| 2 | Predictability notion | Both, staged (intrinsic then model) | **Intrinsic only** | Thesis will eventually need an intrinsic->realized link. Mitigation: ensemble outputs are saved so a later model-error study needs no regeneration. Agent agreed not to raise again. |
| 6 | Stim arms | 5 arms spanning the zero-sum plane | **2 arms `{0, +d1}`** | A null is then ambiguous between "metric uncontrollable" and "wrong direction probed" — the exact failure the `gamma` investigation turned on. Mitigation: arm list is a config parameter, so adding `-d1`/`d2` is a config line. |
| 7 | Predictability score | gap-normalized `sigma_ens/Delta` + horizon `tau` | **R2 vs global variance** | Global denominator mixes across-branch and within-branch variance. Mitigation: resolved at Q8 — the denominator became `Var_across_parents`, scoped per phase, which is the proper conditional R2. |
| 11 | Artifact | 1 kHz EEG + region PTP (~1 GB) | **Full 10 kHz (~19 GB)** | Disk is fine; notebook load is the real constraint. Mitigation: per-branch npz + mmap + 1 kHz derived cache. |
| 12 | Verdict rule | pre-register thresholds + null reading | **Interpret after results**; `h_eval` = 1.0 s default but configurable | Post-hoc thresholds are how the retracted readings in `docs/` happened. Mitigation: full curves over `h` are reported, so any threshold is applied in the open rather than cherry-picked. |

---

## Repo evidence this design is grounded in

Read these before second-guessing any decision — most objections a fresh agent would raise are
already answered in them.

- `scripts/probe_payoff_crossover.py` — `GOOD_COMMAND = (2.0, 0.0, -2.0)`, `SEEDS`, and the cost
  anchor: **353 s for 10 runs of ~14 s => ~2.5x realtime** for forward simulation, parallelizes
  trivially across seeds.
- `src/neuro/seizure.py` — `SpreadProfile`, EZ/PZ region names and gains, the 5 mV / 1 s
  persistence criterion, the spread schedule that sets the branch times.
- `src/neuro/jansen_rit.py` — the state/history/RNG structure (branching gotcha above), and
  state-independent additive noise at L166 (why common random numbers works).
- `src/utils/processing.py` — `compute_psd`, `band_energy`, `synchronization` already exist; reuse.
- `notebooks/healthy_vs_seizure_eeg.py` — existing marimo notebook covering some descriptive
  ground (PTP, band energy, topoplots). Check before duplicating plots.

Note `TODO.md` is modified and `docs/mpc_next_steps.md` is deleted in the working tree — both
pre-existing, unrelated to this session, and untouched by it.

---

## Suggested skills

- **`tdd`** — for `src/neuro/metrics.py`. The metric registry has a uniform signature and pure
  windowing semantics, which is close to ideal for test-first. The `h >= window` validity rule and
  the trailing-window convention are both easy to get subtly wrong and easy to pin with tests.
- **`codebase-design`** — before writing `src/neuro/metrics.py` and `src/neuro/ensembles.py`, if the
  seam between them is unclear. The registry interface is the one piece other code will depend on.
- **`astral:uv`** — dependency and script running. Repo convention is `uv add <pkg>` over editing
  `pyproject.toml`, and `uv run ...` for everything.
- **`astral:ruff`** / **`astral:ty`** — linting/formatting and type checking. `AGENTS.md` requires
  zero formatting, lint, type or test errors.
