# CasADi Jansen-Rit: refactor for MPC + system identification

## Context

`src/neuro/jansen_rit_casadi.py` provides differentiable `ca.SX` primitives
(`sigmoid`, `f_rhs`, `get_network_coupling`, `heun_step`) for the Jansen-Rit
network. Today only `A`, `K`, `W` are symbolic; `B, a, b, C1-4, e0, v0, r,
mean_input` are baked as numeric constants, and `A` is passed as a special-cased
argument separate from `params`. There is **no** NLP/`nlpsol`/`Opti`/shooting
code anywhere in the repo.

The user wants the model usable for two CasADi/IPOPT optimizations:

1. **MPC** — optimize the control schedule `{u_t}` over a horizon; parameters fixed.
2. **System identification** — optimize a *configurable subset* of parameters `θ`
   from a logged `(u(t), y(t))` dataset; controls/measurements fixed.

Both reduce to the **same** differentiable transition `x_{t+1} = F(x_history, u_t, θ)`
plus output `y_t = g(x_t)`. MPC fixes `θ` and frees `{u_t}`; sysid fixes `{u_t,y_t}`
and frees (part of) `θ`. The blocker is that "which quantities are symbolic" is
currently hard-coded. The fix is a single **configurable symbolic parameter
vector** threaded through param-struct-based primitives, then two thin drivers.

Decisions (from the user): build primitives **and** both drivers; keep everything
**N-agnostic**; **include delayed coupling**; make the sysid **free-parameter set
configurable**.

## Design overview

Reuse the UKF estimator's canonical augmented-state ordering so sysid results map
onto existing tooling (`src/neuro/estimator.py`): dynamic block first, then
`K (1)`, `vec(W) (N²)`, optional `eeg_gain (N)`, `A (N)`, `mean_input (N)`, ...
`θ` therefore spans **both dynamics params** (from `JansenRitParams`) **and the
observation gain** — the EEG gain is a first-class learnable parameter, matching
the estimator's `estimate_eeg_gains` per-node diagonal-leadfield block at
`g_start = 6N+1+N²`. Bounds mirror the estimator's projection step (`estimator.py`
~683-714): `K≥0`, `W≥0` zero-diagonal, `A∈[0,10]`, `I≥0`, `eeg_gain≥0`. For parameters in the `free` set without explicit UKF bounds (e.g., `B`, `a`, `b`), use sensible wide defaults like `±50%` of their nominal values or `[0, ∞]`.

Delays are handled with the **existing history-list mechanism**: `heun_step`
already takes `X_history_list[d] = state at t-d`. A roll-out keeps a sliding
window of length `max_delay+1`; the first `max_delay` entries are pre-horizon
history supplied as fixed NLP **parameters** (seeded from `x_hat` for MPC or the
dataset for sysid), later entries are rolled states. Delay length drives symbolic
graph size, not decision-variable count.

## Phase 1 — Refactor primitives (`src/neuro/jansen_rit_casadi.py`)

Foundational; both drivers depend on it.

1. **`JRSymbolicParams` dataclass** — an independent dataclass duplicating the `JansenRitParams` dynamics fields, plus `eeg_gain`, `K`, `w_weights`, and `delay_steps` (which remains a NumPy array, not symbolic). Fields are typed `ca.SX | float | FloatArray`. The dynamics functions read only the dynamics attributes; `measure` reads `eeg_gain`. Annotate args as the union.

2. **Param packing helpers** (canonical order = estimator's, incl. the gain block):
   - `build_param_symbols(base: JansenRitParams, free: Sequence[str], n_nodes) -> (theta: ca.SX, params: JRSymbolicParams, meta)` — creates `ca.SX.sym` for each free field (scalar for `K,B,a,b,...`; `(1,N)` for `A,mean_input,eeg_gain`; `(N,N)` for `W`), fills the rest from `base`, returns `theta = vertcat(free symbols)` in canonical order, and exposes the symbolic `eeg_gain` for `measure`.
   - `pack_theta(values, free, n_nodes) -> np.ndarray` / `unpack_theta(theta_vec, free, n_nodes) -> (JansenRitParams, eeg_gain)` — numeric round-trip for initial guess / reading results back (gain returned separately since it isn't a `JansenRitParams` field).
   - `theta_bounds(free, n_nodes) -> (lb, ub)` and `theta_nominal(base, free, n_nodes)` — bounds from the estimator convention (incl. `eeg_gain≥0`); nominals for scaling (see Conditioning).
   - `free` may include `"eeg_gain"` alongside any dynamics field.

3. **Fold `A`, `K`, `W` into the param struct.** New signatures (drop the
   special-cased `A`):
   - `f_rhs(X, coupling, u, params) -> ca.SX`  (reads `params.A`)
   - `get_network_coupling(X_history_list, D, params) -> ca.SX`  (reads `params.K, params.W`)
   - `heun_step(X_history_list, u, D, params, dt) -> ca.SX`
   Now *any* field may be symbolic; removes the `A`/rest asymmetry and the
   `# noqa: N803` on `A`. `D` stays a numeric `np.ndarray` (structural).

4. **CasADi output / measurement** (mirror `estimator._hx_step_jit` + `jansen_rit.output`):
   - `output(X) -> ca.SX` → `X[1,:] - X[2,:]` (node LFP, shape `1xN`).
   - `measure(X, gain, selected_channels) -> ca.SX` supporting **both estimator
     modes**: fixed full leadfield `gain @ (x2-x3)` (numeric `(n_ch, N)`), or a
     **learnable per-node diagonal gain** `eeg_gain * (x2-x3)` when `eeg_gain` is in
     `θ` (the `estimate_eeg_gains=True` mode), then channel-select. The symbolic
     gain flows in via the `θ` struct from Phase-1 packing. Needed by both drivers.

5. **CasADi control projection** (port `jansen_rit.project_control`):
   `project_control(u_elec, gamma_2d) -> ca.SX` → `u_elec @ gamma_2d`, `gamma_2d`
   numeric `(n_elec, N)`. Lets MPC decision vars be per-electrode currents.

6. **Roll-out helper** — the shared engine for both NLPs:
   `rollout(x0_history, controls, D, params, dt) -> (X_seq, Y_seq)` where
   `x0_history` is the pre-horizon window (params), `controls` is a list of
   per-step node inputs (decision vars for MPC / data for sysid). Maintains the
   sliding window, calls `heun_step` per step, collects states and `output`/`measure`.
   Supports **single shooting** (chain expressions) directly; for **multiple
   shooting** the driver instead creates per-step state vars and adds
   `gap_k = X_{k+1} - heun_step(window_k, ...)` constraints reusing `heun_step`.

7. **Update tests** (`tests/test_jansen_rit_casadi.py`) to the new param-struct
   signatures; add coverage for `pack/unpack` round-trip, `measure` vs
   `_hx_step_jit`, `project_control` vs numpy, and a symbolic-`θ` step still
   matching `_fx_step_jit`.

## Phase 2 — MPC controller (`src/neuro/mpc.py`)

`MPCController(Controller[MPCLog])` fitting `simulate.controller.Controller`
(`control.py` patterns: `update(t, ref, x_hat) -> (u, log)`, `from_config`).

- `__init__`: build the NLP **once** via `nlpsol`/`Opti`, parametric in
  `(x_hat→x0+history, θ numeric, y_ref)`. Decision vars = per-electrode controls
  over horizon `Nh` (+ state vars if multiple shooting). Cost =
  `Σ ‖measure(X_k) − y_ref‖² + R‖u_k‖² + R_du‖Δu_k‖²`; constraints = control box
  bounds, dynamics continuity, pre-horizon history params. Single-shooting default
  (small NLP); multiple-shooting flag for longer/stiffer horizons.
- Maintains a rolling buffer of past `x_hat` states for delayed-coupling lookups
  (same idea as `JansenRitDynamics`/UKF history). Warm-start from previous solve.
- `update`: set params, solve, return first control `u_0`.
- `from_config`: load connectome (`W, D, gamma, gain` via `connectome.py`), build
  `θ` from `JansenRitParams` configured with constant parameters via the config (no UKF interaction),
  horizon, cost weights, bounds.
- Verify it slots into `simulate.simulation.Simulation.run` loop unchanged.

## Phase 3 — System identification (`src/neuro/sysid.py` + `scripts/run_sysid.py`)

- **Data**: a logged orchestrator run with persistent excitation. Reuse the
  existing `WaveformController` (`control.py`, PRBS/RAS/multisine via
  `build_input_schedule`) to generate `u(t)`; read `u` and `universal_y_mea` from
  `log.npz` exactly as `scripts/run_ukf_feasibility.py` does. Reuse
  `select_regions_and_channels` for the reduced-model case.
- **Problem**: `free` set is configurable (CLI/config) and may include `eeg_gain`
  (jointly identifying source dynamics and the observation gain, as the UKF does
  with `estimate_eeg_gains`). Build `θ` decision vars + bounds from Phase-1
  helpers; fix the rest. Multiple-shooting over samples
  (states as vars + continuity) for conditioning, with optional segmenting for
  long records. Cost = `Σ_t ‖measure(X_t) − y_data_t‖² + λ·reg`. Solve with IPOPT.
- **Output**: fitted `JansenRitParams` (via `unpack_theta`). Validate by free-run
  multi-step prediction error vs persistence (mirror the reporting block in
  `run_ukf_feasibility.py` 210-235) and by comparing to UKF estimates.

## Conditioning (important for sysid convergence)

Parameters span very different scales (`A~3.25`, `mean_input~90`, `W∈[0,1]`,
`K~0.75`, and `eeg_gain` set by the leadfield magnitude). Scale decision variables
to O(1) by dividing by `theta_nominal` so IPOPT sees a well-conditioned problem. Note the prior UKF finding that *z-scoring
the signals* kills `A`'s observability — so scale **parameters**, and scale
channels by their std (as `run_ukf_feasibility.py` 125-130 does), never z-score
away amplitude. Apply the estimator's bounds as hard constraints.

## Critical files

- `src/neuro/jansen_rit_casadi.py` — primitives refactor (Phase 1).
- `tests/test_jansen_rit_casadi.py` — update to new API.
- `src/neuro/mpc.py` *(new)* — MPC controller (Phase 2).
- `src/neuro/sysid.py` *(new)* + `scripts/run_sysid.py` *(new)* — fitting (Phase 3).
- Reuse (no edits): `estimator.py` (param layout, bounds, `_hx_step_jit`),
  `connectome.py` (`weights/delays/gamma/gain`), `control.py`
  (`Controller`, `WaveformController`, `build_input_schedule`), `jansen_rit.py`
  (`JansenRitParams`, `output`, `project_control`, plant for synthetic data).

## Verification

1. `uv run pytest tests/test_jansen_rit_casadi.py` — primitives still match the
   numba reference under the new param-struct API (incl. symbolic-θ step vs
   `_fx_step_jit`, `measure` vs `_hx_step_jit`, pack/unpack round-trip).
2. `uv run ty check` + `uv run ruff check/format` clean (pre-commit will enforce).
3. **MPC smoke test**: single node initialized in the `A=3.6` limit-cycle regime;
   MPC should reduce output peak-to-peak vs `ZeroController`; assert solver
   converges and `u` respects bounds.
4. **SysID recovery test**: generate synthetic data with known `θ` (plant +
   PRBS `WaveformController`), fit on 1-3 nodes with zero delay first, assert
   recovered `θ ≈ true` within a strict tolerance (e.g., 1%) to ensure global minimum convergence; then repeat with nonzero delays.
5. End-to-end: run `scripts/run_sysid.py` against an existing capture `log.npz`
   and report free-run prediction error vs persistence.

## Risks / notes

- **Delay × N × horizon graph size**: with real delays at `dt=1e-4`, the
  history window is hundreds of steps; the symbolic graph for coupling lookups
  grows accordingly. Mitigate with the reduced model, a coarser MPC `dt`, or
  zero-delay where acceptable. Decision-variable count is unaffected.
- **`get_network_coupling` is an O(N²) Python loop building SX** — we will stick with this simple loop without delay grouping or vectorization optimizations.
- **Identifiability**: a broad free set (B, a, b, ...) is ill-conditioned and may
  be unidentifiable from `y` alone; persistent excitation + regularization +
  starting from the minimal set is the safe path.
- **`A` × `eeg_gain` scaling ambiguity**: both scale output amplitude, so freeing
  both at once leaves `A·gain` only jointly observable. Mitigate by adding an L2 regularization penalty to the cost function, pulling `eeg_gain` and other free parameters toward their nominal values.
- **Naming**: `jansen_rit_casadi.heun_step` collides with `jansen_rit.heun_step`;
  importers must alias. Consider renaming if it becomes a friction point.
