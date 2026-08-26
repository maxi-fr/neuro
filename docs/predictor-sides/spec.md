# Predictor Training/Inference Sides

## Problem Statement

The unified `Predictor` protocol bundles two different jobs into one torch module: the
training-time forward pass and the runtime NumPy interface (`prime`, `step`, `rollout`,
`prime_many`, `rollout_many`, `absorb`, `is_ready`, `initial_state`). The JAX/Equinox MPC has since
landed as a separate inference side — `absorb`/`is_ready`/`initial_state` plus
`discrete_dynamics`/`output` — which reimplements the same State Absorption and opaque-state layout
in a second framework, with a third hand-rolled float64 copy in the tests. Four consequences:

1. **The torch runtime is dead weight.** Every runtime call converts between NumPy and torch, and
   most of those methods are only exercised by tests — the controller uses the JAX adapters' own
   `absorb`, never the torch one.
2. **The opaque state has three owners.** The torch module, the JAX adapter and the float64 test
   reference each encode the shift-register/lift layout independently; they can only drift.
3. **Free-run scoring scores the wrong model.** Evaluation runs the torch module's NumPy runtime,
   not the deployed JAX model the controller actually steps.
4. **Checkpoint reads go through an intermediate dataclass layer** that duplicates what `save`/`load`
   already know, purely so a torch-free reader and the float64 reference have a neutral structure.

Split the interface into two sides, each native to its framework, and delete the overlap.

## Solution

Two sides, one exchange:

- **Training side (torch).** `forward` (batched standardized Rollout), `save`/`load`, and the
  Ridge-Fittable capability. Every NumPy runtime method is deleted.
- **Inference side (jax).** `discrete_dynamics`/`output` (jax), `absorb`/`is_ready`/`initial_state`
  (host NumPy State Absorption), a stateless jax `rollout` for free-run scoring, and
  `save`/`load`.

The exchange is the existing `.npz` checkpoint, produced and consumed through per-model
`to_checkpoint`/`from_checkpoint` pairs composed with shared serialization primitives. Free-run
evaluation moves to the inference side: the training run hands the fitted torch model to the jax
model in memory and scores *that*. Cross-side parity — the jax `rollout` equals the decoded torch
`forward` — replaces the float64 reference as the correctness pin.

## Implementation Decisions

### 1. Two ABCs replace the single Predictor protocol

The single runtime protocol is split into two abstract base classes the models inherit directly.
Abstract members are methods only; attributes stay documented contract rather than abstract
properties, because abstract properties collide with the equinox field machinery on the inference
side and with `__init__` instance attributes on the training side.

```text
TrainingPredictor (torch, abstract)
    forward(x) -> Tensor                       # batched standardized Rollout over the trained Span
    save(stem) -> None
    to_checkpoint() -> (meta, arrays)
    from_checkpoint(meta, arrays) -> Self      # classmethod
    load(stem) -> Self                         # classmethod

InferencePredictor (jax, abstract)
    discrete_dynamics(x, u, t, dt) -> x'       # jax, one position: a sample or a Frame
    output(x) -> raw output                    # jax
    absorb(state, y, u) -> state'              # host NumPy, State Absorption
    is_ready(state) -> bool                    # Priming completeness
    initial_state() -> state
    rollout(y_hists, u_hists, u_futures) -> (B, positions, outputs)   # jax, stateless
    save(stem) -> None
    to_checkpoint() -> (meta, arrays)
    from_checkpoint(meta, arrays) -> Self      # classmethod
    load(stem) -> Self                         # classmethod
```

Attributes such as channel/control/output counts, `dt`, `m`, `n`, and `ne` are part of the
contract by documentation, not abstract enforcement. The training ABC lives with the torch
modules; the inference ABC lives with the jax models and folds in the controller's existing
priming-seam protocol. The dependency-light type module keeps only the NumPy aliases, the
activation registry, the layer alias and `RidgeFittable`.

### 2. Persistence is per-model, with shared primitives only

The checkpoint dataclasses and the dispatch loaders are deleted. Each concrete model — torch and
jax — owns `to_checkpoint` (build the `(meta, arrays)` pair) and `from_checkpoint` (rebuild from
that pair); `save` and `load` are thin wrappers over the shared `save_checkpoint`/`load_checkpoint`
primitives. A `load_meta` primitive reads only the JSON meta block, so config validation never
loads weights. The `.npz` layout and the `model_type` string dispatch are unchanged.

The evaluation handoff is in memory: the training run builds the jax model with
`from_checkpoint(*trained.to_checkpoint())`, never touching disk.

### 3. The NumPy runtime methods are deleted from the training side

`prime`, `step`, `rollout`, `prime_many`, `rollout_many`, `absorb`, `is_ready`, `initial_state`,
`encode`, and `decode` leave the torch modules. The controller keeps its State Absorption seam on
the jax side only, so the opaque-state layout now has a single owner. The torch standardizer
properties survive — Ridge fitting and `to_checkpoint` read them.

### 4. Free-run scoring moves to the inference side

`rollout_batches` swaps `prime_many`/`rollout_many` for the stateless jax `rollout`; the scoring
math (`nmse`, `window_energy`, log-energy error) is unchanged. This applies to the waveform arm's
sample-grid Rollout scoring. The observable arm's held-out standardized MSE stays a training-side
score, because it measures the training `forward`, not the deployed Frame recursion. The move
creates a training→inference dependency edge; it is accepted as the honest expression of "score
what you deploy."

### 5. Ridge-Fittable stays a capability Protocol

The waveform module attaches its closed-form members per instance only at depth 0, so capability
depends on a constructor argument. Inheritance cannot express that, so `RidgeFittable` remains a
`runtime_checkable` Protocol: `design_normal_equations(trajectories) -> (G, P)` and
`install_readout(A) -> None`, bias column last, unchanged. The build-time capability check survives
exactly as today.

### 6. One observable training module

The one-shot observable predictor is removed. The remaining torch observable module is the folded
one: the batched `forward` plus standardizer buffers and `save`/`load`. The stepwise runtime it
used to carry lives only on the jax side now.

### 7. Standardizer and geometry stay shared vocabulary

`Standardizer` and the Observable/spectral geometry classes do not move. They are shared
vocabulary both sides already import. The change is where each side holds them: torch keeps
float32 buffers (with NumPy properties for Ridge and checkpoint writing), jax keeps plain
`center`/`scale` arrays. `Standardizer`'s array-key convention becomes the contract between torch
`to_checkpoint` and jax `from_checkpoint`; there is no jax `Standardizer`.

### 8. Validation and scripts read metadata, not checkpoints

Config validation reads `load_meta` and dispatches on the `model_type` string instead of loading a
dataclass and type-switching on it. Scripts call the concrete `load` of the side they need.

### 9. The float64 reference is deleted

The hand-rolled float64 NumPy runtime is removed. Correctness is pinned by cross-side parity: the
jax `rollout` (float64) against the decoded torch `forward` (float32), at the tolerance the
existing parity tests already use.

## Testing Decisions

A good test asserts behavior at a seam a caller actually uses, on tiny synthetic modules with
random weights, and never reaches into private state or asserts on trained accuracy. The six seams
below were chosen deliberately; each has prior art in this repo.

### Seam 1 — The two-sided checkpoint contract

Torch `save` → jax `load` reproduces the decoded torch `forward` via the jax `rollout`; jax `save`
→ torch `load` round-trips weights, buffers and metadata. This is the whole exchange tested at the
`.npz` boundary. Prior art: `tests/test_checkpoint_reader.py`, `tests/test_kinds.py`.

### Seam 2 — Cross-side numerical parity

For both model kinds, across activations, depths, residual and horizon: the jax `rollout`
(raw in → raw out) equals the decoded torch `forward`, at float32↔float64 tolerance. This replaces
the float64 reference as the pin. Prior art: `tests/test_trajopt_kinds.py`,
`tests/test_predictor_module.py::test_forward_matches_torch_rollout`.

### Seam 3 — Training result contract

`train(cfg, data_files)` still returns the trained torch Predictor, its `candidates`, and the
free-run scores, and `save`/`load` round-trip. Both gradient-descent and Ridge arms still train both
model kinds. The torch modules no longer expose the runtime methods, and that absence is asserted
as part of the contract. Prior art: `tests/test_unified_trainer.py`,
`tests/test_predictor_train.py`, `tests/test_observable_train.py`.

### Seam 4 — Inference contract at the ABC

The jax models are `InferencePredictor` subclasses; the controller's absorb → is_ready → solve →
shift loop runs against a `load`ed checkpoint; the stateless jax `rollout` is the evaluation entry.
Prior art: `tests/test_mpc.py`, `tests/test_full_cost.py`.

### Seam 5 — Ridge capability, unchanged

Depth-0 modules fit through `RidgeFittable`; depth>0 fails at build time. Must keep passing after
the torch modules shed their runtime. Prior art: `tests/test_ridge_trainer.py`.

### Seam 6 — Free-run scoring math, unchanged

`nmse`, `window_energy`, `evaluate_rollouts`, and `evaluate_log_energy` yield the same numbers for
a fixed model; only the subject of `rollout_batches` changes. The pure-scoring unit tests stay
verbatim. Prior art: `tests/test_metrics.py`, `tests/test_batched_rollout.py` (rewritten against the
jax `rollout`).

### What pytest must not assert

Predictive accuracy, control-sensitivity magnitude, and closed-loop seizure burden are not unit
tests. Tests assert wiring, equivalence and invariants; performance stays the job of probe scripts
and closed-loop benchmarks.

## Out of Scope

- **The JAX/Equinox controller itself** — the loop, costs, and transcriptions are unchanged; this
  spec only formalizes the predictor interface split they already consume.
- **Changing the `.npz` layout** — it is kept byte-for-byte compatible so existing checkpoints
  remain loadable.
- **Merging the two config trees** — dispatch still happens at the seams.
- **A further "everything steps one position" observable refactor** — this spec only folds the
  one-shot module away; deeper runtime unification belongs to a later ticket.
- **Closed-loop gates and performance benchmarks.**

## Further Notes

- The prior spec's `SymbolicModel`/`ObservableModel` note is now stale: the symbolic side is gone,
  and the "stay unrelated so it fails at build time" guarantee is carried by the abstract methods
  of the two ABCs rather than by keeping two model classes apart.
- The single source of truth for the opaque state is now the inference side alone. Cross-side
  parity is the regression net that keeps the training `forward` honest against it.
- `CONTEXT.md` needs no new terms: `Predictor`, `State Absorption`, `Priming`, `Rollout`,
  `Observable`, `Frame`, `Trainer`, and `Ridge-Fittable` still describe the split interface.
