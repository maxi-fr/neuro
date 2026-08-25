# Unified Predictor Protocol and Training/Sweep Seams

## Problem Statement

The predictor side of the codebase has three trained models — the autoregressive MLP, the
observable-space MLP, and the ESN — that share almost all of their runtime surface (`prime`,
`rollout`, `absorb`, `is_ready`, `initial_state`) yet are wired to three separate training
implementations and two bespoke sweep scripts, with no seam between them. Three consequences:

1. **Training is duplicated, not shared.** The two torch training loops (waveform and observable)
   each reimplement the batch/curriculum/early-stopping machinery; the ESN's closed-form fit lives
   in a script. Adding a model, or adding a fit algorithm, means copying the loop again.
2. **The observable projection lives in the wrong layer.** The Frame targets are computed in the
   observable trainer by standardizing the EEG, inverting the standardizer, and reducing — a round
   trip that re-does work the data pipeline already did. The NumPy/torch/CasADi copies of the same
   spectrogram-and-Frame machinery are three implementations of one function.
3. **The framework-free artifact has become the wrong abstraction.** The Predictor is the torch
   module; the NumPy artifact duplicates its weights and its runtime in a second framework, purely
   so evaluation and the CasADi controller can read it. Moving the ESN to torch (so it can later be
   gradient-descent trained) and moving the controller to a torch→CasADi weight adapter removes the
   reason for that duplication.

The design must also survive the coming JAX/Equinox MPC replacement: the predictor contract settled
now is what that conversion reads, so it has to be minimal and framework-agnostic, not
CasADi-shaped.

## Solution

Define one minimal runtime **Predictor** protocol — predict the next output(s) — and let everything
else hang off it. Trainers become generic algorithms (gradient-descent, Ridge) that ask a Predictor
which fits it supports, instead of being paired to a model kind. Sweeps become a seam with two
implementations (Optuna, Grid) that minimise an objective named in config and drawn from the
Trainer's candidates. The observable projection moves into the data pipeline, computed raw-direct
on the trajectories. The framework-free artifact and its NumPy runtime are removed; persistence is
implementation-specific checkpoints whose weights are stored as NumPy arrays, so the torch-free
control path can load them.

The protocol is runtime-only and raw-units in/out:

```text
prime(y_hist, u_hist) -> state
step(state, u) -> (state', output)     # one position: a sample (waveform) or a Frame (observable)
rollout(state, u_future) -> (n_positions, n_outputs)
prime_many(y_hists, u_hists) -> (B, state)
rollout_many(states, u_futures) -> (B, n_positions, n_outputs)
absorb(state, y, u) -> state           # State Absorption
is_ready(state) -> bool                # Priming completeness
initial_state() -> state
# identity: channel count, control count, output count, dt, priming_steps, horizon
```

`__init__`, `save`, `load`, geometry, output-space tag and `model_type` are **not** on the protocol
— they are implementation-specific. Standardizers live inside the module as buffers, so callers
exchange raw units only.

## Implementation Decisions

### 1. The Predictor protocol is runtime-only and raw-units

The protocol carries the members above and nothing else. State is opaque: the MLP predictors carry
a shift-register, the ESN a reservoir vector, and callers never inspect it. Raw units at the
boundary: `prime` takes raw history, `rollout` returns raw predictions, `absorb` takes raw
measurements. Standardizers are module buffers, invisible through the protocol. `step`'s "position"
is a sample for the waveform predictors and a Frame for the observable predictor; that granularity
is an implementation detail, and the interface only sees "advance one position, emit that
position's output."

For the observable predictor, `step`'s control input is one Frame-mean `(n_controls,)`; `rollout`
and `absorb` take raw samples, and `rollout` aggregates raw controls into Frame means internally
via the shared geometry helper. The observable predictor is rebuilt from scratch as a
one-Frame-per-step module, not the incumbent one-shot forecast over the Control Horizon.

### 2. No framework-free artifact; persistence is a numpy-readable checkpoint

`MLPArtifact`, `ESNArtifact` and `ObservableArtifact`, and their NumPy `prime`/`rollout`, are
deleted. A trained Predictor is a torch module; its `__init__`, `save` and `load` are its own
business and persist whatever "everything we need to know later" is — weights, standardizer
buffers, and implementation-specific attributes (the observable predictor's geometry, provenance,
`model_type`). Weights are stored on disk as NumPy arrays plus a metadata block, so the torch-free
controller and validation paths read them without importing torch. The protocol does not require a
save/load contract.

### 3. Geometry is data-pipeline metadata, not an interface member

The Observable geometry that shapes the observable predictor's targets stays in the data pipeline
and config. It determines the predictor's output width and position count, which become plain
constructor arguments; the geometry itself rides along as a plain attribute on the observable
module and in its checkpoint, not as a protocol member. The controller reads it from there when it
needs to hinge a Cost against the log reference.

### 4. Trainers are generic algorithms, not model kinds

A Trainer is named by the fit it performs — gradient-descent or Ridge — and asks the Predictor which
fits it supports. Gradient-descent serves any torch module over the base protocol. Ridge serves any
module that is Ridge-Fittable. A model can support either or both: a depth-0 MLP supports both; the
ESN supports Ridge now and gradient-descent once it is a torch module. The config names the fit
with `training.fit`; a fit the model does not support fails at build time.

### 5. RidgeFittable is a capability protocol, not a base-protocol member

```text
design_normal_equations(trajectories) -> (G (f, f), P (f, c))   # bias column last
install_readout(A (c, f)) -> None
```

The Ridge Trainer is `G, P = model.design_normal_equations(trajs); A = ridge(G, P, λ); model.install_readout(A)`
with no knowledge of which model it holds (`ridge` leaves the bias column unregularized). The capability is named around the readout, not
"linearity": a depth-0 MLP is linear end-to-end, the ESN is nonlinear end-to-end with only a linear
readout, and a depth-0 observable MLP is linear end-to-end — all three are Ridge-Fittable. `is_linear` is rejected as the name because it cannot express
that, and a bare boolean would still leave the Trainer needing per-kind knowledge of how to extract
features and where to write the result.

- depth-0 MLP: `design_normal_equations` folds the one-step input features (with the control-window
  shift alignment) and the next-step targets into `G`/`P`; `install_readout` writes the single
  layer. This extracts today's warm-start least-squares fit out of the gradient-descent Trainer.
- ESN: `design_normal_equations` streams the fixed-reservoir states `[h; 1]` and their one-step-ahead
  targets into `G`/`P`; `install_readout` writes `W_out`.
- depth-0 observable MLP: `design_normal_equations` harvests the per-Frame lifted state `z_m`;
  `install_readout` writes the shared readout.

### 6. Sweep is a seam with two implementations

`OptunaSweep` serves the two NN predictors; `GridSweep` serves the ESN (outer reservoir-size ×
ridge-λ grid, inner Optuna over the continuous reservoir hyperparameters — the incumbent hybrid,
not a pure grid). The objective is a named string in config drawn from the Trainer's `candidates`;
every candidate is recorded on every trial so a finished study can be re-ranked. Trainer candidates
are per-model — waveform `{log_energy, val_loss, rollout_nmse}`, observable `{val_loss,
val_log_mse}`, ESN `{rollout_nmse, log_energy}` — and `closed_loop` is a sweep-level candidate
available for all three kinds.

### 7. ESN moves to torch

Reservoir generation (sparse random matrix, spectral-radius rescale) stays a one-time scipy
preprocessing step whose outputs are copied into torch buffers. The Predictor itself — `absorb`,
`readout`, `step`, `rollout` — becomes torch. The ridge readout solve uses `torch.linalg`. The
closed-form Trainer stays; the move is what later lets gradient-descent serve the ESN too.

### 8. Interim controller bridge: torch → CasADi weights

The current MPC builds its symbolic models from artifacts; with artifacts gone it needs a source. A
thin adapter rebuilds the existing CasADi bridges from a torch module's buffers plus its metadata
instead of an artifact. This is throwaway — it keeps the controller green until the Equinox
conversion lands and deletes it.

### 9. Data pipeline owns the observable projection

The Frame targets are computed in the data-preparation module, raw-direct on the loaded trajectories
(no standardize→inverse round trip), on the same window grid the waveform targets use. The NumPy
spectrogram-and-Frame machinery becomes the single target path; the differentiable Loss terms stay
where they are, because they score predictions, not targets.

### 10. Config stays two trees; dispatch is by config type

`NNPredictorConfig` and `ESNPredictorConfig` are kept. The Trainer and Sweep dispatch on which tree
they are handed, not on a `model_type` read off the module. `training.fit` names the algorithm;
`sweep.objective` becomes a named string instead of a fixed literal.

### 11. Removals

The three artifacts, their NumPy runtime, the artifact-dispatch entry point, and the
`SymbolicModel`/`ObservableModel` protocols. Evaluation moves onto the torch module. The
simulation-validation and closed-loop-evaluation paths stop reading `.npz` artifacts and read
checkpoints instead.

## Testing Decisions

A good test asserts behaviour at a seam a caller actually uses, on tiny synthetic modules with
random weights, and never reaches into private state or asserts on trained accuracy. The six seams
below were chosen deliberately; each has prior art in this repo.

### Seam 1 — Predictor protocol

Every torch predictor satisfies the protocol; `rollout_many` returns `(B, n_positions, n_outputs)`;
raw units round-trip at the boundary with standardizers internal. Prior art:
`tests/test_batched_rollout.py`, `tests/test_predictor_module.py`, `tests/test_esn.py`.

### Seam 2 — Trainer

`train(cfg, data_files)` returns the Predictor plus `candidates` and a `save` that round-trips.
Gradient-descent serves waveform and observable; Ridge serves depth-0 MLP, depth-0 observable, and
ESN; `candidates` contains the config-named objective. Prior art: `tests/test_predictor_train.py`,
`tests/test_observable_train.py`.

### Seam 3 — RidgeFittable

Depth-0 MLP ridge fit reproduces the incumbent warm-start least-squares; ESN
`design_normal_equations` reproduces the incumbent harvest; depth-0 observable fits the shared
readout on harvested `z_m`; `install_readout` writes weights back; a non-fittable model handed to
the Ridge Trainer fails at build time. Prior art: `tests/test_esn.py` (harvest/ridge),
`tests/test_predictor_train.py` (warm start).

### Seam 4 — Sweep

The two sweep implementations pick the config-named objective from the Trainer's `candidates` and
record every candidate per trial; tests assert the selection wiring, not Optuna internals. Prior
art: `tests/test_config.py` (sweep config validation).

### Seam 5 — Torch ↔ CasADi parity (interim)

The weight adapter's symbolic rollout equals the checkpoint's float64 rollout to ~1e-10 across
activations and horizons (the float32 module matches the checkpoint to ~1e-5); checkpoint save/load
round-trips weights, standardizers and recorded metadata. Prior art:
`tests/test_nn_predictor_casadi.py`, `tests/test_observable_casadi.py`, and
`tests/test_predictor_module.py::test_control_path_never_imports_torch`, which must keep passing
with the new modules added to its parametrisation.

### Seam 6 — Data pipeline

Raw-direct Frame targets equal today's round-trip `build_targets` output to floating-point
tolerance, and the resolved Frame count matches the geometry's own derivation across a range of
horizons. Prior art: `tests/test_observable_train.py`, `tests/test_predictor_losses.py`.

### What pytest must not assert

Predictive accuracy, control-sensitivity magnitude, and closed-loop seizure burden are not unit
tests — they depend on trained weights and the data budget. Tests assert wiring, equivalence and
invariants; performance remains the job of probe scripts and closed-loop benchmarks.

## Out of Scope

- **The JAX/Equinox MPC itself** — the torch→Equinox conversion and any change to the `simulate`
  Controller interface. This spec only requires that the Predictor protocol be minimal enough for
  that conversion to read.
- **CasADi removal** — the interim adapter keeps it alive; deleting it belongs to the MPC
  replacement.
- **Gradient-descent training of the ESN** — enabled by the torch move, not performed here.
- **Merging the two config trees** — they stay separate; dispatch happens at the seams.
- **Closed-loop gates and performance benchmarks** — none are introduced here.

## Further Notes

- The two connections to the MPC replacement are the reason for decisions 1 and 2: the protocol is
  what makes the torch→Equinox conversion a mechanical weight-copy, and the persistence decision
  (numpy-readable checkpoints, implementation-specific save/load) is what the Equinox loader will consume.
  The Loss↔Cost correspondence keeps the Observable geometry as the single source of truth shared
  by the training Loss and the controller Cost, which is why decision 3 demotes geometry to shared
  metadata rather than deleting it.
- The `SymbolicModel`/`ObservableModel` split recorded in the type-alias module is deliberately
  superseded by the single protocol; the "unrelated so it fails at build time" concern moves to the
  capability protocol (decision 5), which preserves the build-time guarantee.
- `CONTEXT.md` gains `Trainer`, `Ridge-Fittable` and `Sweep`; the rest of the glossary (`Predictor`,
  `Rollout`, `State Absorption`, `Priming`, `Observable`, `Frame`) is unchanged and still describes
  the minimal protocol.
