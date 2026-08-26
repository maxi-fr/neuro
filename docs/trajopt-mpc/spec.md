# Replacing the CasADi MPC with trajopt

## Problem Statement

The controller side of the codebase — `MPCController`, `LinearMPCController`, `MPCNlp`, and the
three `MPCSolver` implementations — is a hand-built CasADi symbolic-optimization stack: a
hand-rolled multiple/single-shooting NLP builder, a hand-rolled DFT matrix for the PSD hinge cost
(CasADi has no FFT), and three solver wrappers (`IpoptMPCSolver`, `SqpMPCSolver`,
`SqpFallbackMPCSolver`) built directly on `ca.nlpsol`/`ca.qpsol`. It reads its model from a
CasADi-specific artifact bridge (`build_symbolic_model`, the `SymbolicModel`/`ObservableModel`
protocols) that the unified-predictor refactor (`docs/unified-predictor/spec.md`) marks for
deletion, keeping it alive only through decision 8's throwaway torch→CasADi adapter.

`trajopt` — already a git dependency (`pyproject.toml:19`) pulling JAX and Equinox transitively —
is a JAX/Equinox trajectory-optimization framework that overlaps almost entirely with what this
stack hand-builds: `Objective`/`CostFunction` for the cost, `ConstraintList` for box and linear
constraints, `transcription/{ipopt,osqp,clarabel,single_shooting}.py` for NLP formulation, native
JAX solvers (`ilqr`, `al`, `altro`, `boxqp`, `pn`) as an alternative to NLP transcription entirely,
and — notably — `trajopt.simulate.TrajOptMPC`, a `simulate.Controller` subclass whose `update()`
loop (`with_measurement` → `problem.solve` → shift) is close enough to what our own controllers
hand-roll today to serve as this refactor's concrete reference, even though decision 2 below means
we don't reuse it directly.

Three internal knowledge-base benchmarks (`knowledge-base/Notes/sqp_ipopt_fallback_benchmark.md`,
`narx_mpc_benchmark.md`) already settled one formulation question this refactor must not
relitigate: single shooting decisively beats every multiple-shooting or output-lifted formulation
tested at this problem's scale (960-dim autoregressive state, up to 62 EEG channels) — 40–70x
faster in the worst comparisons — because state/output-lifted defect constraints scale with
channel count and shift-register width in a way this system is never small enough to avoid. Best
observed closed-loop latency (hybrid SQP, single shooting) is still ~850ms against the 20ms
real-time deadline; closing that gap is explicitly **not** a goal of this refactor.

## Sequencing

This spec assumes `docs/unified-predictor/spec.md` has landed: predictors are torch modules with
numpy-readable checkpoints, the Observable predictor is one-Frame-per-step (not the incumbent
one-shot Control-Horizon forecast), and the throwaway torch→CasADi bridge (decision 8) exists only
as an interim stopgap. Building this refactor on top of that bridge would mean writing a
torch→Equinox adapter and immediately deleting the torch→CasADi one it was built to replace — pure
waste. The Predictor protocol (opaque state, raw-units `prime`/`step`/`rollout`/`absorb`) was
deliberately designed so this conversion is a mechanical weight copy; this spec is that conversion
cashing in.

## Solution

Replace `MPCController`/`LinearMPCController` with our own new `simulate.Controller` implementation
built directly on trajopt's lower-level primitives (`Problem`, `Objective`, `ConstraintList`,
`MPCState`, solver backends) — not `trajopt.simulate.TrajOptMPC` itself. `TrajOptMPC.update` is the
concrete reference for the receding-horizon loop shape (`with_measurement` → `problem.solve` →
extract first control → `shift`), but it isn't reusable as-is: its `x_hat` parameter already
assumes the full model state (decision 2), and the "previous state" it persists between ticks
(`self.state.x0`, populated by `MPCState.shift()`) is the prior solve's second knot — a model
prediction, not the measurement-corrected state `absorb` produces. A wrapper composing `TrajOptMPC`
would need a second, parallel piece of persistent state to hold the true absorbed value, duplicating
what `TrajOptMPC` already tracks incorrectly for our purposes — building directly on the primitives
`TrajOptMPC` itself is built on removes that duplication instead of papering over it. Our work is
building the pieces `TrajOptMPC.from_config` shows how to assemble: an `AbstractModel` per predictor
kind, an `Objective` encoding the current cost terms, and a `ConstraintList` encoding the box bounds
and Kirchhoff constraint — plus the config seam and checkpoint loader that construct them — wired
into our own controller's `update()` instead of into `TrajOptMPC`.

Given the single-shooting result above, `transcription/single_shooting.py` is the default
transcription and multiple-shooting support is not carried forward; `NarxMPC`
(`src/neuro/control/narx_mpc.py`) is dropped, not ported, as it never won a benchmark at this
channel count. Solver selection across trajopt's NLP-transcription and native-JAX backends is
evaluated empirically (see Testing), not chosen up front — this is the one open axis actually
worth measuring, since JIT/vmap solvers are a genuinely new option we haven't had before, even
though hitting the 20ms deadline is out of scope.

## Implementation Decisions

### 1. Predictor → `AbstractModel`

Each torch Predictor (waveform MLP, Observable) becomes a `trajopt.dynamics.base.
DiscreteDynamics`/`EuclideanModel` subclass: `n` = the Predictor's opaque state width (the
shift-register vector), `m` = control count, `discrete_dynamics(x, u, t, dt)` = one
`step`. Weights load from the Predictor's numpy-readable checkpoint into Equinox buffers — the
permanent replacement for the torch→CasADi bridge, not another throwaway adapter.

### 2. Our own `Controller[L]`, modeled on `TrajOptMPC`

`TrajOptMPC.update(t, ref, x_hat)` calls `state.with_measurement(x_hat)` treating `x_hat` as
already the full model state — it does not know about `absorb`. A naive fix (an outer wrapper that
calls `absorb` itself and forwards the result as `x_hat`) turns out not to compose: the "previous
state" a wrapper would need to call `absorb(state, y_new, u_prev)` against isn't available from
`TrajOptMPC` afterwards. `TrajOptMPC.state.x0` is set by `MPCState.shift()`, and per
`trajopt/problem.py`'s `.shift()`, `new_x0` is `new_X[0]` — the *previous solve's second knot*, the
model's own one-step-ahead prediction, not the measurement-corrected state `absorb` produced. Using
it as the "previous state" for the next `absorb` call would silently diverge from ground truth
under any model mismatch, defeating the point of absorbing a measurement at all.

So instead of wrapping/subclassing `TrajOptMPC`, we implement our own `simulate.Controller[L]`
subclass directly, structured after `TrajOptMPC.update`'s loop but with our own persistent state:

- Keep the true absorbed predictor state and `u_last` as private instance attributes — the same
  pattern `MPCController` already uses today (`self._state`, `self._u_last`), not derived from any
  post-solve trajectory.
- On each tick: `self._state = predictor.absorb(self._state, x_hat, self._u_last)` (our
  controller's `x_hat` keeps the ABC's existing raw-measurement meaning, unlike `TrajOptMPC`'s),
  then call `state.with_measurement(self._state)` → `problem.solve(...)` → extract the first
  control → `.shift(dt)` for warm-starting the next solve, mirroring `TrajOptMPC.update` line for
  line where the steps do carry over.
- `Problem`, `MPCState`, `Objective`, `ConstraintList`, and the solver backends are used directly —
  `TrajOptMPC` itself is not instantiated; it's a reference for how those pieces compose, not a
  dependency.

`Controller[L]` (`simulate/controller.py`) is a plain mutable Python class, not a frozen
`eqx.Module`, so owning ordinary instance attributes for this state is architecturally identical to
what `MPCController` does today — no new pattern is needed, just the same one restated on trajopt's
primitives instead of CasADi's.

### 3. Cost: quadratic terms are native, hinge/L1 terms are custom `CostFunction`s

EEG power + control effort maps directly onto `trajopt.costs.base.QuadraticCostFunction` /
`Objective.tracking` (state and control weights, optionally terminal). The spectral PSD hinge and
Observable log-reference hinge become custom `CostFunction` subclasses (`evaluate(x, u, t) ->
scalar`, JAX-autodiff'd — no more hand-rolled DFT matrix; `jnp.fft` replaces it directly). The
Observable hinge cost must keep reading `ObservableGeometry` as the shared source of truth with
the training-time `Loss`, per the unified-predictor spec's Loss↔Cost correspondence (Further
Notes) — this doesn't change, only its host type does. L1 sparsity's epigraph-slack formulation is
solver-transcription-specific (meaningful for NLP transcription, not for the native JAX solvers);
whether to keep a smooth L1 surrogate for the native-solver path or restrict L1 to
transcription-based solvers is a decision the solver-comparison benchmark (see Testing) should
resolve with data, not upfront.

This native/custom split is **not** a solver-compatibility requirement — checked directly against
`trajopt/expansions.py` and `trajopt/transcription/osqp.py`. Every solver backend, including OSQP
and Clarabel, consumes costs through `Objective.cost_expansion`, which for any `CostFunction` that
isn't already `QuadraticCost`/`DiagonalCost` calls `jax.grad`/`jax.hessian` on `evaluate` at the
current knot to build a local second-order Taylor expansion — the same shape (`Q`, `R`, `H`, `q`,
`r`) either way.

### 4. Constraints

Box bounds on `u` map onto `trajopt.constraints.bounds`; Kirchhoff sum-to-zero maps onto
`trajopt.constraints.linear`. No shooting-defect constraints are needed since single shooting
condenses state out of the decision variables entirely — the `shooting_depth >= horizon` memory
constraint (multiple shooting being unsolvable in the current NLP) becomes moot rather than
needing a fix, since multiple shooting isn't carried forward.

### 5. Config and wiring stay class-path dispatch

`TrajOptMPC.from_config`'s `{class_path, ...}`-dict pattern for `problem`/`model` is the template
our own controller's `from_config` follows, matching the existing pattern
(`configs/simulation/*mpc*.yaml` currently dispatches `class_path:
neuro.control.nonlinear_mpc.MPCController`). Migrating a simulation config means pointing
`class_path` at our new controller (e.g. `neuro.control.trajopt_mpc.TrajOptMPCController`) and at a
`Problem`-building factory function per predictor kind (waveform/Observable) that assembles the
`AbstractModel` + `Objective` + `ConstraintList` from checkpoint + geometry + the existing
cost-weight fields (`w_y`, `w_u`, `w_u_l1`, `w_psd`, `u_max`, etc.) — no new config tree, consistent
with decision 10 of the unified-predictor spec (dispatch by config type, not a `model_type` field).

### 6. Removals

`nonlinear_mpc.py`, `linear_mpc.py`, `nlp.py`, `solvers.py`, `narx_mpc.py`,
`nn_predictor_casadi.py`, `observable_casadi.py`, and
`artifacts.py::build_symbolic_model` are deleted (the latter two and the artifact bridge are
already slated for removal by the unified-predictor spec regardless). CasADi drops as a dependency
— it's a **direct** pyproject dependency, not purely a controller concern, so confirm
`src/neuro/transforms.py`'s CasADi usage isn't load-bearing for something outside the controller
before removing it from `pyproject.toml`.

## Testing Decisions

Prior art for what "equivalent behavior" means here is the unified-predictor spec's Seam 5
(symbolic rollout parity to ~1e-10 against a checkpoint) and this repo's own solver benchmarks
— tiny synthetic models plus an
explicit open-loop/closed-loop latency table, not trained-weight accuracy.

- **Model parity.** Each `AbstractModel` adapter's rollout matches the torch Predictor's `rollout`
  to floating-point tolerance, on tiny synthetic weights — the trajopt-side equivalent of
  `test_nn_predictor_casadi.py`/`test_observable_casadi.py`.
- **Cost parity.** Custom `CostFunction`s (PSD hinge, Observable hinge) evaluate to the same
  values as today's CasADi cost graph on fixed inputs, both scoring the same
  `ObservableGeometry`-derived reference.
- **Controller wiring.** Our own controller (decision 2) reproduces today's `MPCController.update`
  control sequence on a fixed scripted trajectory — same persistent-state pattern (absorbed
  predictor state, `u_last`), same output contract, `TrajOptMPC` used only as the reference
  implementation, not present at runtime.
- **What this must not assert.** Same exclusion as the unified-predictor spec: predictive
  accuracy, control-sensitivity magnitude, and closed-loop seizure burden stay out of unit tests —
  they're a probe-script/benchmark concern, not a wiring-correctness one.

## Out of Scope

- **`NarxMPC`.** Dropped, not ported — it lost every benchmark comparison at this channel count.
- **Multiple shooting.** Not carried forward; single shooting is the only formulation this spec
  scopes in, per the benchmark evidence.
- **Any change to the unified-predictor Predictor protocol itself**, or to the Observable
  geometry / Loss formulation — this spec only consumes both as already defined.
- **The `simulate` package's `Controller`/`Dynamics` ABCs** — no changes needed; our controller
  implements `Controller[L]` directly (decision 2), and `TrajOptDynamics` is reused unmodified for
  the plant side if a trajopt model stands in as the simulated plant. `TrajOptMPC` itself is not
  instantiated anywhere in this refactor — it's a design reference only.
- **Solver comparison (new benchmark note).** Follow up: a
  `knowledge-base/Notes/trajopt_solver_benchmark.md` benchmarking every trajopt backend for nonlinear MPC—
  NLP transcription (Ipopt × single shooting) and native JAX solvers (ALTRO) — against the existing CasADi SQP+IPOPT baseline, on the same open-loop/closed-loop

## Further Notes

- The two connections back to the unified-predictor spec are decisions 1 and 2 here: the
  Observable predictor's one-Frame-per-step move is what collapses `ObservableModel` into a plain
  `AbstractModel`, and the Predictor protocol's `absorb`/opaque-state design is what makes it
  possible to replicate `TrajOptMPC`'s loop in our own controller with a single `absorb` call
  ahead of `with_measurement`, per decision 2.
- `trajopt`'s own `Problem.solve` already tracks a `single_shooting` flag on the solver object
  (`trajopt/problem.py:99`) and special-cases dual-variable shapes for it, suggesting single
  shooting is a first-class, not bolted-on, path in the library — consistent with it being our
  only formulation.
- `CONTEXT.md` should gain no new glossary terms from this refactor — `Cost`, `Control Horizon`,
  and `Rollout` already exist and mean the same thing on the trajopt side; only their
  implementation moves.
