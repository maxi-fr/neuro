# Architecture

A descriptive inventory of the modules in `src/`, compiled ahead of a refactor. Each entry states what exists today: responsibility, interface, seam, adapters, and dependencies. It does not propose changes or flag refactor candidates — that's a separate pass.

Vocabulary (see `/codebase-design`):

- **Module** — anything with an interface and an implementation; scale-agnostic (a function, class, file, or package).
- **Interface** — everything a caller must know to use the module correctly: signatures, invariants, ordering constraints, error modes, required config.
- **Seam** — the location where the interface lives; the place you could swap what satisfies it without editing callers.
- **Adapter** — a concrete thing satisfying the interface at a seam. One adapter present means the seam is unproven; two or more means it's real and load-bearing.

Some modules have no seam at all — a single concrete implementation with no substitution point. These are marked `none`/`n/a` rather than forced into the template.

---

## Predictor / Modeling

### Trajectory Dataset Preparation

**Files:** `src/neuro/predictor/data.py`

**Responsibility:** Load raw simulation trajectories, split them by file into train/validation, fit standardizers on the training split, and build sliding-window regression datasets for multi-step predictor training.

**Interface:**

- `load_trajectory(data_file, n_steps, downsample, dt, cutoff_hz=None) -> (u_data, y_data)`
- `split_data_files(data_files, train_split) -> (train_files, val_files)`
- `extract_windows_flattened(data, window_size) -> FloatArray`
- `build_dataset_for_trajectory(u_data, y_data, n_y, n_u, N) -> (X, Y)`
- `fit_standardizers(data_files, *, n_steps_cfg, downsample, dt, train_split, scaler, global_scaling, cutoff_hz=None) -> TrajectorySplit`
- `prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon, dt, train_split, *, scaler, global_scaling, cutoff_hz=None) -> Datasets`
- invariants / ordering constraints: `split_data_files` always keeps at least one file per side; standardizers are fit only on the training split, then applied to both.
- error modes: `ValueError` if fewer than 2 trajectory files are supplied (no file left to hold out).

**Seam:** none — a set of plain functions and two result dataclasses (`TrajectorySplit`, `Datasets`), no adapter/interface split.
**Adapters:** n/a (single implementation, no seam)

**Depends on (modules):** Causal Filtering, Standardization.
**Requires (config/data):** trajectory `.npz` files (each holding `sensor_0.y_mea` and `controller.u`).

**Depended on by:** Autoregressive State-Space Predictor (`predictor/train.py`), Observable-Space Predictor (`predictor/observable_train.py`), ESN Predictor (`esn_training.py`, via `fit_standardizers`).

## Predictor Loss Terms

**Files:** `src/neuro/predictor/losses.py`

**Responsibility:** Compute the weighted training loss for the autoregressive state-space predictor from a configurable set of curriculum-scheduled loss terms.

**Interface:**

- `Loss` Protocol — `name`, `weight`, `span_steps`, `start_epoch` properties, `__call__(pred, true, ctx) -> (Tensor, dict[str, float])`
- `LossContext` dataclass (`y_center`, `y_scale`, `fs`, `epoch`) with `to_raw(x)`
- `build_losses(specs, fs) -> list[Loss]`
- `total_loss(losses, pred, true, ctx) -> (Tensor, dict[str, float])`
- `spectrogram`, `pool_bins`, `frame_kernel`, `smooth_frames` — torch helpers shared by the spectral loss terms
- invariants: a loss term only contributes once `ctx.epoch >= loss.start_epoch`; `ctx.epoch is None` (validation) always trusts the whole rollout span.
- error modes: `TypeError` from `build_losses` on a spec type with no registered factory.

**Seam:** `Loss` Protocol, defined in this file.
**Adapters:**

- `CurriculumMSE` — plain MSE over a curriculum-lengthening trusted rollout prefix
- `StftLoss` — log-spectrogram matching loss on hopped, pooled, frame-smoothed segments
- `EegMsLoss` — log-space mean-square windowed-power matching loss

**Depends on (modules):** Predictor Configuration Schemas & Loading, Spectral Cost Support.
**Requires (config/data):** `LossSpecs`/`LossSpec` config objects, sampling rate `fs`.

**Depended on by:** Autoregressive State-Space Predictor (`predictor/train.py`).

### Autoregressive State-Space Predictor

**Files:** `src/neuro/predictor/module.py`, `src/neuro/predictor/train.py`, `src/neuro/predictor/artifact.py`

**Responsibility:** Define, train, and freeze a torch MLP that predicts future EEG samples autoregressively, one raw sample at a time, over a fixed horizon.

**Interface:**

- `AutoregressiveMLP(*, n_y, n_u, horizon, n_channels, n_controls, hidden_size, depth, activation)` — `nn.Module`; `forward(x) -> (B, horizon*C)`; `to_artifact(dt, downsample, y_std, u_std) -> MLPArtifact`; `from_artifact(art) -> Self`
- `train(cfg, data_files, *, seed_offset=0) -> TrainingResult`
- `TrainingResult.save(artifact_dir)` — writes `model.npz` and `training_stats.json`
- `MLPArtifact` — the frozen, framework-free numeric twin: `encode`/`decode`, `prime`/`prime_many`, `forward_1step`, `rollout`/`rollout_many`, `meta`, `load`/`save`
- invariants / ordering constraints: the control window shifts into the MLP input *before* the forward call during training/rollout construction, but *after* priming in `MLPArtifact.rollout` — opposite order, same rule (both windows end at step `t` when `y_{t+1}` is predicted), documented at both call sites.
- error modes: `train` raises `ValueError` if `cfg.training.losses` is `None` (an `observable` config should go through `train_observable` instead); `_fit` raises `ValueError` if train or validation loss goes `NaN`.

**Seam:** `predictor/module.py` (the torch module `predictor/train.py` trains).
**Adapters:**

- `AutoregressiveMLP` — the only adapter currently present.

**Depends on (modules):** Trajectory Dataset Preparation, Predictor Loss Terms, Predictor Artifacts & Rollout Evaluation, Observable Metrics & Ensemble Scoring, Provenance.
**Requires (config/data):** `NNPredictorConfig`/`TrainingConfig`, trajectory `.npz` files.

**Depended on by:** Symbolic Stepping Predictor (`nn_predictor_casadi.py` loads `MLPArtifact`); Observable-Space Predictor (`observable_module.py` reuses `activation_module`/`to_numpy` from `module.py`, `observable_train.py` reuses `float32_tensor`/`lr_schedule`/`shuffled_batches` from `train.py`); `neuro/observable.py` reuses `ACTIVATIONS`/`Layers`/`mlp_forward` from `artifact.py`.

### Observable-Space Predictor

**Files:** `src/neuro/predictor/observable_module.py`, `src/neuro/predictor/observable_train.py`, `src/neuro/observable.py`

**Responsibility:** Define, train, and freeze a torch model that forecasts standardized log-power (Observable) on a coarser Frame grid over the Control Horizon, together with the Observable geometry math (reducing raw EEG to Frames, averaging controls per Frame) shared identically by training and by the CasADi forecaster.

**Interface:**

- `ObservableMLP(*, n_y, n_u, horizon, n_channels, n_controls, geometry, fs, z_dim, lift_hidden, lift_depth, transition_hidden, transition_depth, activation)` — `nn.Module`; `forward(x) -> (B, n_frames, C*n_values)`; `to_artifact(...) -> ObservableArtifact`; `from_artifact(art) -> Self`
- `train_observable(cfg, data_files, *, seed_offset=0) -> ObservableTrainingResult`
- `prepare_observable_data(cfg, data_files) -> ObservableData` (also builds the control-blind sensitivity arm)
- `log_observable(y, geometry, fs) -> FloatArray` — raw log-Observable Frame grid
- `control_means(geometry, horizon, fs) -> FloatArray` — the fixed Frame-averaging operator over controls
- `ObservableArtifact` — `prime`/`prime_many`, `lift_state`, `forecast(state, u_future) -> (n_frames, n_channels, n_values)`, `meta`, `load`/`save`
- `envelope_log_reference` / `load_log_reference` — reduce a healthy reference envelope onto the same Frame grid so it is commensurable with a forecast
- invariants / ordering constraints: the recursion advances once per Frame rather than once per sample, so BPTT/forecast depth is the Frame count, not the Control Horizon; `ObservableArtifact.prime` is byte-identical in layout to `MLPArtifact.prime`.
- error modes: `train_observable`/`prepare_observable_data` raise `ValueError` if `cfg.observable` is `None`; `_fit`-equivalent `fit()` raises `ValueError` on `NaN` loss; `log_observable`/`envelope_log_reference` raise `TypeError` on an unsupported geometry/envelope pairing.

**Seam:** `predictor/observable_module.py` (the torch module `observable_train.py` trains).
**Adapters:**

- `ObservableMLP` — the only adapter currently present.

**Depends on (modules):** Trajectory Dataset Preparation; Autoregressive State-Space Predictor (reuses `activation_module`/`to_numpy` from `predictor/module.py`, `mlp_forward`/`ACTIVATIONS`/`Layers` from `predictor/artifact.py`, and `float32_tensor`/`lr_schedule`/`shuffled_batches` from `predictor/train.py`); Predictor Configuration Schemas & Loading, Provenance, Spectral Cost Support, Standardization.
**Requires (config/data):** `NNPredictorConfig` with an `observable` block, trajectory `.npz` files.

**Depended on by:** Symbolic Observable Forecaster (`observable_casadi.py` uses `ObservableArtifact` and `control_means`).

### ESN Predictor

**Files:** `src/neuro/esn.py`, `src/neuro/esn_training.py`

**Responsibility:** Build and ridge-fit a leaky-integrator echo-state-network predictor from harvested normal equations (closed-form, not gradient descent), and freeze it into a framework-free artifact exposing the same prime/rollout surface as the MLP predictor.

**Interface:**

- `generate_reservoir(reservoir_size, spectral_radius, density, input_scaling, in_dim, seed) -> (W_res, W_in)`
- `harvest_normal_equations(trajectories, y_std, u_std, w_res, w_in, leak_rate, priming_steps, noise_sigma, seed) -> (G, P)`
- `solve_ridge(G, P, ridge_lambda) -> W_out`
- `ESNPredictor` — the core NumPy recurrence: `readout(h)`, `absorb(h, z, v)`, `step(h, v)`
- `ESNArtifact` — `encode`/`decode`, `prime`/`prime_many`, `rollout`/`rollout_many`, `meta`, `load`/`save`
- `prepare_training_data(cfg, data_files) -> ESNTrainingData` — train/val trajectory split, fitted standardizers, `in_dim`; no windowing, since the ESN consumes whole trajectories rather than sliding windows
- invariants / ordering constraints: `solve_ridge` regularizes every state dimension except the bias column; `harvest_normal_equations` pairs reservoir state `h[t]` — entered *before* absorbing `(z[t], v[t])` — with target `z[t]`, so the fit is a genuine one-step-ahead prediction rather than a reconstruction.
- error modes: `generate_reservoir` raises `ValueError` if the computed spectral radius is zero or non-finite.

**Seam:** none — `ESNPredictor`/`ESNArtifact` is the only reservoir engine in the repo.
**Adapters:** n/a (single implementation, no seam)

**Depends on (modules):** Trajectory Dataset Preparation (via `fit_standardizers`); Provenance, Standardization.
**Requires (config/data):** `ESNPredictorConfig`, trajectory `.npz` files.

**Depended on by:** Symbolic Stepping Predictor (`esn_predictor_casadi.py` loads `ESNArtifact`).

### Symbolic Stepping Predictor (CasADi)

**Files:** `src/neuro/esn_predictor_casadi.py`, `src/neuro/nn_predictor_casadi.py`

**Responsibility:** Wrap a trained predictor artifact (ESN or MLP) as a CasADi-symbolic single-step state-transition function the MPC solver can differentiate through, presenting one interface across both predictor kinds.

**Interface (shared by both adapters):**

- `state_shape`, `history_depth`, `n_controls`, `n_channels`, `native_horizon`, `is_linear`, `free_syms`
- `initial_state() -> FloatArray`
- `absorb(state, y, u) -> FloatArray` — NumPy state update, outside the symbolic graph
- `is_ready(state) -> bool`
- `step(history, u) -> ca.SX | ca.MX` — one symbolic transition step
- `output(x) -> ca.SX | ca.MX` — decode state to raw EEG
- `f_step`, `f_out` — cached compiled `ca.Function` wrappers around `step`/`output`
- invariants / ordering constraints: `step`/`output` build a pure CasADi graph, compiled once via `cached_property`; `absorb` always runs in NumPy and never enters the differentiated graph.
- error modes: `ESNSymbolicModel.absorb`/`is_ready` assert the state is a `ReservoirState` (`AssertionError` otherwise); no declared error modes on the NN side.

**Seam:** the member set both classes implement (this is the domain's `SymbolicModel` — see `AGENTS.md`'s note that it and `ObservableModel` share seven members on purpose so a model missing the stepping members fails at build time). No `Protocol` class is declared in these files; the two classes are structurally identical and used interchangeably by the MPC. Formalized in `neuro.types.SymbolicModel` (see Type Aliases & Symbolic Model Protocols).
**Adapters:**

- `ESNSymbolicModel` (`esn_predictor_casadi.py`) — wraps `ESNArtifact`; state is a reservoir vector tagged with a step counter (`ReservoirState`, an `ndarray` subclass).
- `NNSymbolicModel` (`nn_predictor_casadi.py`) — wraps `MLPArtifact`; state is the flattened `[y-window | u-window]` shift register; also exposes `mlp_forward_ca`, the symbolic twin of `predictor/artifact.py`'s `mlp_forward`.

**Depends on (modules):** ESN Predictor (`ESNArtifact`), Autoregressive State-Space Predictor (`MLPArtifact`); Standardization.
**Requires (config/data):** a saved `ESNArtifact` or `MLPArtifact` `.npz`.

**Depended on by:** Symbolic Observable Forecaster (`observable_casadi.py` imports `mlp_forward_ca` from `nn_predictor_casadi.py`); Controller (`nonlinear_mpc.py`, `linear_mpc.py`).

### Symbolic Observable Forecaster (CasADi)

**Files:** `src/neuro/observable_casadi.py`

**Responsibility:** Wrap a trained `ObservableArtifact` as a single CasADi-symbolic function that forecasts the whole Control-Horizon log-Observable Frame grid in one shot, rather than stepping sample by sample.

**Interface:**

- `state_shape`, `fs`, `n_controls`, `n_channels`, `n_values`, `native_horizon`, `geometry`
- `n_frames(horizon) -> int`
- `initial_state() -> FloatArray`
- `absorb(state, y, u) -> FloatArray`
- `is_ready(state) -> bool`
- `forecast(x0, u_seq) -> ca.SX | ca.MX`, shape `(C * n_values, n_frames)`
- `f_forecast` — cached compiled `ca.Function`
- invariants / ordering constraints: deliberately carries no `f_step`/`f_out` — there is no per-sample state to step, so the MPC branches on which model it received rather than guarding an optional member (stated in the class docstring); Frame `m` structurally consumes only control means that landed before its Segment ended.
- error modes: none declared beyond shape mismatches surfaced by CasADi itself.

**Seam:** none — `ObservableSymbolicModel` is the only adapter. It is intentionally structurally parallel to, but type-unrelated from, the Symbolic Stepping Predictor's shared interface, so a model missing the stepping members fails at construction time rather than silently. Formalized in `neuro.types.ObservableModel` (see Type Aliases & Symbolic Model Protocols).
**Adapters:**

- `ObservableSymbolicModel` — the only adapter present.

**Depends on (modules):** Symbolic Stepping Predictor (imports `mlp_forward_ca` from `nn_predictor_casadi.py`); Observable-Space Predictor (`ObservableArtifact`, `control_means`); Standardization.
**Requires (config/data):** a saved `ObservableArtifact` `.npz`.

**Depended on by:** Controller (`nonlinear_mpc.py`, via the MPC NLP Builder).

### Simulation Ensemble Generation & Scoring

**Files:** `src/neuro/ensembles.py`

**Responsibility:** Run branched Jansen-Rit plant simulations under fixed stimulation arms to build a validation ensemble (parent/child rollouts across healthy/seizure operating points), and score the stored rollouts against the shared metrics library.

**Interface:**

- `EnsembleConfig`, `Branch`, `Arm` — declarative description of what to generate
- `build_plants(config_path) -> PlantPair`
- `run_parent(...)`, `run_child(...)` — single-rollout simulation primitives
- `generate(out_dir, cfg, plants) -> manifest`, `generate_iter(...) -> Iterator[manifest]`
- `score_ensemble_dir(out_dir, *, hop_s, cutoffs, rebuild=False) -> ScoreArchive`
- `ScoreArchive.ensemble(...)`, `ScoreArchive.state(...) -> Ensemble`
- invariants / ordering constraints: `EnsembleConfig` rejects any branch whose `t_branch` is shorter than `SPREAD_WINDOW_S` (not enough pre-branch history for the seizure state); child seeds are shared across the arms of one branch (common-random-number coupling); stores are written as on-disk memmaps so the full (tens-of-GB) EEG store is never fully resident.
- error modes: `ValueError` from `EnsembleConfig.__post_init__` (branch too early) and from `build_plants` (config's `A` gain vector does not match `neuro.seizure.build_seizure_a_gains`).

**Seam:** none in this template's sense — a standalone simulation-and-scoring pipeline, not an adapter of any interface shared with the rest of this cluster.
**Adapters:** n/a

**Depends on (modules):** Connectome, EEG Measurement, Jansen-Rit Plant, Observable Metrics & Ensemble Scoring, Seizure Spread — no dependency on any predictor/control module.
**Requires (config/data):** a plant simulation config (`configs/simulation/jansen_rit_seizure_excited.yaml` by default).

**Depended on by:** validation/analysis scripts and notebooks.

**Note:** this file's responsibility — synthetic plant-data generation and metric scoring for validation — is distinct from the predictor-training/modeling responsibility of the rest of this section. It is described here because it was reviewed as part of the predictor-cluster pass, but it shares no interface or seam with any other module in this section.

---

## Control & Stimulation

### Controller

**Files:** `src/neuro/control/linear_mpc.py`, `src/neuro/control/nonlinear_mpc.py`, `src/neuro/control/threshold.py`, `src/neuro/control/waveform.py`, `src/neuro/control/zero.py`

**Responsibility:** turn a per-step EEG measurement (and reference) into a per-electrode control vector, at the `simulate.controller.Controller` seam.

**Interface:**

- `__init__(dt, ...)` — build the controller and (for the MPC adapters) compile its solver once.
- `from_config(config: dict[str, Any]) -> Self` — construct from a validated config dict (each adapter has a private `pydantic` `StrictConfig` schema), resolving any on-disk artifacts (predictor model, waveform schedule).
- `update(t, ref, x_hat) -> tuple[FloatArray, <ControllerLog>]` — ingest the current time, reference and measurement; return `(u, log)` where `u` is the emitted `(n_controls,)` (or `(n_u,)`) control and `log` is an adapter-specific frozen dataclass of diagnostics.
- invariants / ordering constraints: `update` is called once per control step in increasing `t`; the MPC adapters carry internal state (`self._state`, `self._u_last`, `self._u_prev`) that assumes strictly sequential calls and is not reentrant. `LinearMPCController`/`MPCController` return an all-zero control with `warmup=True` in `MPCControllerLog` until `model.is_ready(self._state)`.
- error modes: constructors raise `ValueError` on malformed config (e.g. `u_max` size mismatch against `n_controls`, unknown `formulation`/`solver` literal, `LinearMPCController` given a nonlinear predictor).

**Seam:** the `Controller[LogT]` base class from the external `simulate` package; each file below is one adapter satisfying it.
**Adapters:**

- `linear_mpc.py` (`LinearMPCController`) — receding-horizon MPC restricted to a linear (0-hidden-layer) `SymbolicModel`, solved as a QP (OSQP "sparse" or qpOASES "dense") via CasADi's `qpsol`; supports quadratic + L1 control-effort penalties and a Kirchhoff sum-to-zero equality.
- `nonlinear_mpc.py` (`MPCController`) — receding-horizon MPC for a general `SymbolicModel` or `ObservableModel` predictor, formulated as a multiple/single-shooting NLP via `MPCNlp` and solved by one of the MPC Solvers backends; adds a spectral/PSD hinge cost option on top of quadratic and L1 effort terms.
- `threshold.py` (`AmplitudeThresholdController`) — non-MPC trigger: fires a fixed-amplitude, fixed-duration burst whenever a trailing peak-to-peak EEG amplitude crosses a threshold.
- `waveform.py` (`WaveformController`) — open-loop playback of a precomputed per-electrode tES schedule (`build_input_schedule` generates `ras`/`prbs`/`multisine` excitation waveforms); ignores `ref`/`x_hat`.
- `zero.py` (`ZeroController`) — always emits an all-zero control vector, ignoring every input.

**Depends on (modules):** MPC NLP Builder and MPC Solvers (for `nonlinear_mpc.py`); Predictor Artifacts & Rollout Evaluation (predictor loading), Observable-Space Predictor / Symbolic Observable Forecaster (Observable-path support), Spectral Cost Support (PSD envelope), Predictor Configuration Schemas & Loading (`StrictConfig`), Type Aliases & Symbolic Model Protocols.
**Requires (config/data):** `dt`; a predictor artifact path (`linear_mpc`, `nonlinear_mpc`); `u_max` bounds; optionally a healthy-PSD reference npz (`nonlinear_mpc`, `w_psd > 0`); a precomputed schedule or its generating parameters (`waveform`).
**Depended on by:** whatever assembles the closed-loop simulation (the `simulate` runner that calls `Controller.update` each step, e.g. Closed-Loop Evaluation).

### MPC NLP Builder

**Files:** `src/neuro/control/nlp.py`

**Responsibility:** build the symbolic CasADi NLP (decision variables, cost, constraints, bounds) that `MPCController` hands to a solver — not itself a `Controller` adapter.

**Interface:**

- `MPCNlp.build(model, *, horizon, shooting_depth, n_controls, u_max, w_y, w_y_terminal=None, w_u=0.0, w_u_l1=0.0, w_psd=0.0, psd_envelope=None, log_reference=None) -> MPCNlp` — the sole entry point; returns a frozen `MPCNlp` dataclass (`nlp: dict`, `lbx`, `ubx`, `lbg`, `ubg`).
- Internal helpers (module-private, not part of the seam): `_l1_epigraph`, `_sum_to_zero`, `_spectral_hinge_cost`, `_observable_hinge_cost`, `_rollout_cost`, `_observable_cost`.
- invariants / ordering constraints: two mutually exclusive branches selected by `isinstance(model, ObservableSymbolicModel)`. On the Observable branch, `w_y`/`w_y_terminal` must be unset and `shooting_depth >= horizon` (no per-sample state to shoot on); `w_psd > 0` and `log_reference` are required (it has no other output term). On the rollout (`SymbolicModel`) branch, `w_psd > 0` requires `psd_envelope`. `shooting_depth` partitions the horizon into `(horizon - 1) // shooting_depth` segments with defect (continuity) constraints between them; `shooting_depth >= horizon` collapses to single shooting (no defects).
- error modes: `ValueError` for every invariant above, and when `_spectral_hinge_cost`'s `horizon < envelope.window` or its channel count disagrees with the model's output count.

**Seam:** none — this is a builder function/dataclass, not a module with interchangeable adapters. Only one implementation exists and none is expected; it is internal to `MPCController`.
**Adapters:** n/a

**Depends on (modules):** Symbolic Observable Forecaster (`ObservableSymbolicModel`, branch discriminator), Spectral Cost Support (`LOG_FLOOR`, `PsdEnvelope`), Type Aliases & Symbolic Model Protocols.
**Requires (config/data):** a `SymbolicModel`/`ObservableModel`, horizon/shooting geometry, control bounds, cost weights, and (conditionally) a `PsdEnvelope` or log-reference array.
**Depended on by:** Controller (`nonlinear_mpc.MPCController._build_solver`).

### MPC Solvers

**Files:** `src/neuro/control/solvers.py`

**Responsibility:** solve a built `MPCNlp` numerically and report the outcome in a uniform result type, decoupling `MPCController` from the choice of numerical method.

**Interface:**

- `MPCSolver` (a `Protocol`) — `solve(x0: FloatArray, w0: FloatArray) -> MPCSolveResult`.
- `MPCSolveResult` — frozen dataclass: `u_opt, cost, success, n_iter, capped, fallback`.
- Each adapter exposes a `build(mpc_nlp: MPCNlp, **solver_options) -> Self` classmethod that compiles the CasADi `nlpsol` once, plus `.solve(x0, w0)`.
- invariants / ordering constraints: `build` must run once per `MPCNlp` (compiling the solver is expensive); `solve` may be called repeatedly with new `(x0, w0)`. `SqpMPCSolver.solve` and `SqpFallbackMPCSolver.solve` never raise — a solver exception is caught and reported as `success=False` with `cost=nan`.
- error modes: `IpoptMPCSolver.solve` propagates exceptions from the underlying `ca.nlpsol` call uncaught; the SQP-based adapters swallow them instead (see above).

**Seam:** `MPCSolver` Protocol, defined in this file.
**Adapters:**

- `IpoptMPCSolver` — pure IPOPT (`ca.nlpsol` with the `"ipopt"` plugin); success is `Solve_Succeeded` or `Solved_To_Acceptable_Level`.
- `SqpMPCSolver` — standalone SQP (`"sqpmethod"` plugin) with a choice of QP subsolver (`qpoases`/`osqp`/`qrqp`) and Hessian approximation (`limited-memory`/`exact`); exceptions are caught and turned into a failed `MPCSolveResult`.
- `SqpFallbackMPCSolver` — composes one `SqpMPCSolver` and one `IpoptMPCSolver`: tries SQP first, and on failure re-solves with IPOPT warm-started from the SQP iterate (or from the original `w0` if the SQP iterate isn't finite), summing iteration counts and setting `fallback=True`.

**Depends on (modules):** MPC NLP Builder (`MPCNlp`, input type only).
**Requires (config/data):** a built `MPCNlp`; per-adapter solver options (iteration caps, CPU time budget, QP subsolver choice, etc.).
**Depended on by:** Controller (`nonlinear_mpc.MPCController._build_solver`, which selects one of the three adapters by its `solver: Literal["ipopt", "sqp", "sqp_fallback"]` config field).

### Stimulation Model

**Files:** `src/neuro/stimulation/base.py`, `src/neuro/stimulation/analytical.py`, `src/neuro/stimulation/null.py`, `src/neuro/stimulation/roast_3d.py`, `src/neuro/stimulation/yu_dynamic.py`

**Responsibility:** project a controller's per-electrode current vector onto a per-node drive on the simulated plant — the actuation layer between controller output and plant input.

**Interface:**

- `StimulationModel` (ABC): attributes `n_controls: int`, `control_labels: StrArray`; abstract method `project(u: FloatArray) -> FloatArray` returning the node-level drive, shape `(n_nodes,)`, for currents `u` of shape `(n_controls,)`.
- Module-level helpers shared by adapters (`base.py`): `check_n_controls(u, n_controls)` (raises on a control-vector size mismatch), `select_rows(labels, wanted)` (case-insensitive, order-preserving row lookup, raises `ValueError` listing missing electrodes), `assert_region_order(file_labels, node_labels)` (raises unless a file-backed projection's region order matches the plant's connectome order).
- `StimulationConfig` — a pydantic discriminated union (`model` field) over `_NullConfig | _AnalyticalConfig | _Roast3DConfig | _DynamicYuConfig`, the config schema for selecting/constructing an adapter.
- invariants / ordering constraints: adapters that load a field-projection file (`Roast3DStim`, `DynamicYuStim`) require it to carry region labels matching the plant's node order and (for `Roast3DStim`) valid, nonzero region normals; `assert_region_order`/shape checks enforce this at construction, not at `project` time.
- error modes: `check_n_controls` raises `ValueError` on a wrong-length `u`; missing files raise `FileNotFoundError`; malformed/missing NPZ keys or mismatched array shapes raise `ValueError`/`KeyError` at construction.

**Seam:** `StimulationModel` ABC in `base.py`.
**Adapters:**

- `null.py` (`NullStim`) — unstimulated plant; one nominal control electrode whose `project` always returns zeros over `n_nodes` (kept at `n_controls=1` rather than 0 so downstream sizing/Kirchhoff logic is unaffected).
- `analytical.py` (`AnalyticalStim`) — closed-form Coulomb point-source potential per electrode in a homogeneous medium (`compute_gamma`/`_coulomb_potential_fn`), softened by a `spread` radius; `project` is a linear combination `u @ gamma`.
- `roast_3d.py` (`Roast3DStim`) — loads a precomputed ROAST 3D FEM electric-field projection NPZ (via ROAST Field-Projection I/O), reduces each region's field along its cortical normal and scales by a fixed `polarization_length_mm`, yielding a linear `u @ gamma` projection to somatic deflection (mV).
- `yu_dynamic.py` (`DynamicYuStim`) — loads the same kind of field-projection NPZ (E-field and potential) but recomputes the projection per `project(u)` call rather than precomputing a linear `gamma`: superposes the vector E-field and scalar potential over electrodes, then applies a smooth (`tanh`) sigmoidal polarity transition around the montage's median potential — nonlinear in `u`, unlike the other three adapters.

**Depends on (modules):** ROAST Field-Projection I/O supplies the MAT-to-NPZ conversion that `roast_3d.py`'s and `yu_dynamic.py`'s NPZ format assumes (no direct import — linked only by shared file format); Geometry (`analytical.py`, sensor positions); Predictor Configuration Schemas & Loading (`StrictConfig`), Type Aliases & Symbolic Model Protocols.
**Requires (config/data):** electrode list and geometry (`analytical.py`); a field-projection NPZ file at a configured path (`roast_3d.py`, `yu_dynamic.py`); the plant's `region_labels`/node count, supplied by the caller at construction (not from config).
**Depended on by:** whatever builds the plant/simulation pipeline from a `StimulationConfig` (e.g. Jansen-Rit Plant).

### ROAST Field-Projection I/O

**Files:** `src/neuro/stimulation/roast_io.py`

**Responsibility:** convert a ROAST-generated MATLAB field-projection file into the NPZ format `Roast3DStim`/`DynamicYuStim` load — an offline data-preparation utility, not a runtime adapter.

**Interface:**

- `load_roast_field_projection_mat(mat_path) -> tuple[FloatArray, FloatArray | None, list[str], list[str], list[str], FloatArray]` — reads the v7.3 (HDF5) MAT file, transposing MATLAB's column-major arrays, and returns `(projection_E, projection_V, channel_labels, roast_labels, region_labels, region_normals)`.
- `convert_roast_field_projection_to_npz(mat_path=..., npz_path=...) -> None` — end-to-end conversion: load, validate (`_validate`), and `np.savez` the NPZ consumed by the stimulation adapters; prints a summary including any electrode-label substitutions between the requested and ROAST-run labels.
- invariants / ordering constraints: the MAT file's declared `normalsFrame` metadata must equal `"mni_ras"`; the last (return) electrode's `projection_E` row must be all zero (the reference ground).
- error modes: `ValueError` on frame mismatch, shape mismatches (`_validate`), or a nonzero return row; `KeyError` if none of `projection_E`/`leadfield_E`/`leadfield_3d` is present.

**Seam:** none — a single conversion utility with one implementation, invoked as an offline script/tool rather than through a swappable interface.
**Adapters:** n/a

**Depends on (modules):** none (uses `h5py`, `numpy` directly).
**Requires (config/data):** a ROAST-run `-v7.3` MAT file at `mat_path`.
**Depended on by:** `roast_3d.py`'s and `yu_dynamic.py`'s docstrings/error messages reference `convert_roast_field_projection_to_npz` as the tool that produces their input NPZ, but neither imports this module directly — linked only by the NPZ file format they share, not a code dependency.

---

## Plant, Simulation & Clinical Domain

### Jansen-Rit Plant

**Files:** `src/neuro/jansen_rit.py`

**Responsibility:** The whole-brain Jansen-Rit neural-mass network as a steppable plant: given a control input, advance the 6-state-per-node network by one time step and expose the observed LFP.

**Interface:**

- `JansenRitParams(A, B, a, b, C1..C4, e0, v0, r, mean_input, sigma)` — frozen dataclass of biophysical scalars/vectors; `.to_numba_tuple(n_nodes)` broadcasts `A` for the JIT kernels; `.from_config(dict)` builds it from a validated config.
- `JansenRitDynamics(dt, params, conn, stim=None, seed=None, initial_state=None, *, enforce_zero_sum_current=True, log="none")` — the plant itself, implementing `simulate.dynamics.Dynamics[L]` (`x`, `n_inputs`, `dynamics(t, x, u) -> x_next`, `_make_log()`; stepping/ZOH/logging machinery lives in the base class, not here).
- `JansenRitDynamics.from_config(dict) -> Self` — builds params, `Connectome`, and a `StimulationModel` from a raw config dict.
- `simulate_network(*, dyn, duration, control_current=0.0, stim_window=None, t0=0.0) -> (t, x_traj)` — steps a prebuilt plant forward and returns its whole trajectory; `x_traj` shape `(6, N, n_samples)`.
- `lfp(x_traj) -> y` — observed output `x2 - x3`.
- `resting_state(conn, dt, settle_s=2.0) -> FloatArray` — runs a noiseless plant to its fixed point, shape `(6, N)`.
- invariants / ordering: `dyn` owns its own monotonic clock (state, RNG, delay history); resuming a plant across multiple `simulate_network` calls requires passing the correct `t0`, otherwise the plant's delay-history indexing silently desyncs. Per-electrode `control_current` must sum to ~0 (Kirchhoff) unless `enforce_zero_sum_current=False`.
- error modes: `ValueError` on `initial_state` shape mismatch; `ValueError` from `_assert_zero_sum_current` when currents don't sum to zero and enforcement is on; `ValueError` in `simulate_network` when `control_current` electrode count doesn't match the plant's controls.

**Seam:** `simulate.dynamics.Dynamics` (external `simulate` package) — the abstract plant-stepping interface used throughout the orchestrator (`update`/`evaluate`, integrator-optional `dynamics(t, x, u)` kernel, `_make_log()`).
**Adapters:**

- `neuro.jansen_rit.JansenRitDynamics` — the whole-brain Jansen-Rit network (this repo's only plant adapter reachable through simulation configs; `simulate.dynamics.LinearDynamics`, in the external `simulate` package, is a second adapter of the same seam but outside this repo).

**Depends on (modules):** Connectome (structural weights/delays/coupling), Stimulation Model (current-to-node projection), Predictor Configuration Schemas & Loading (`StrictConfig`, `parse_array`).
**Requires (config/data):** `dt`, biophysical params, a `Connectome`, optionally a `StimulationModel`, RNG seed.

**Depended on by:** EEG Measurement (`lfp` for the EEG projection), Seizure Spread (`simulate_network`, `lfp`, `JansenRitDynamics` type), Simulation Ensemble Generation & Scoring, the `simulate` orchestrator via `class_path: neuro.jansen_rit.JansenRitDynamics` in simulation configs (e.g. `configs/simulation/jansen_rit_baseline.yaml`).

### TVB Reference Simulator

**Files:** `src/neuro/tvb_reference.py`

**Responsibility:** Build and run an independent, third-party (TVB library) Jansen-Rit whole-brain simulation of the same EZ/PZ network, to validate the hand-rolled plant's dynamics and EEG projection against a trusted external implementation.

**Interface:**

- `build_reference_simulator(connectome, *, dt_ms=0.1, K=0.54, nsig=1e-4, ez=..., pz=..., a_ez=..., a_pz=..., a_bg=..., seed=69, with_eeg=True) -> tvb.simulator.simulator.Simulator` — configures a TVB `Simulator` (model, coupling, integrator, monitors) from this repo's `Connectome`.
- `run_reference(sim, duration_s) -> ReferenceResult` — runs the TVB simulator and returns `ReferenceResult(t, region_y, eeg)`; `region_y` is `y1 - y2` per region, `eeg` is the mirror-corrected 62-channel projection (or `None` without an EEG monitor).
- `reference_eeg_leadfield(sim) -> FloatArray` — the EEG monitor's region-level leadfield `(62, n_regions)`.
- `_spatialized_gain(...)` — builds the per-region excitatory-gain column for EZ/PZ/background regions (private helper).
- invariants: `dt_ms` is expected to match the hand-rolled engine's `dt` (0.1 ms == `1e-4` s) for the two to be comparable; `with_eeg=False` makes `run_reference`'s `eeg` field `None` and `reference_eeg_leadfield` raise.
- error modes: `ValueError` from `reference_eeg_leadfield` when the simulator has no EEG monitor.

This module is not an adapter of the `Dynamics` seam: it does not implement `dynamics()`/`_make_log()` or get stepped by the `simulate` orchestrator. It wraps and drives TVB's own `simulator.Simulator` directly through TVB's API, with its own `run_reference` entry point. It shares structural input (`Connectome`) and EEG geometry (`neuro.eeg`, `neuro.geometry`) with the Jansen-Rit plant but is a standalone reference/validation harness, not a second adapter behind the same interface.

**Seam:** none (not consumed through a shared interface within this repo; called directly by validation code).
**Adapters:** n/a — single concrete module, not a seam.

**Depends on (modules):** Connectome (structural data), EEG Measurement (projection/region-mapping file constants, mirror-partner permutation), Geometry (`SENSORS_FILE`), TVB (`tvb.datatypes.*`, `tvb.simulator.*`, external).
**Requires (config/data):** a `Connectome`; TVB's bundled connectivity/projection/sensor/region-mapping data files (loaded via `.from_file()`).

**Depended on by:** validation notebooks/scripts (e.g. `notebooks/tvb_validation.py`); not imported by any other `src/neuro` module.

### Connectome

**Files:** `src/neuro/connectome.py`

**Responsibility:** Hold the structural (anatomical) data a whole-brain network is built on — region connection weights, tract lengths, coordinates, hemisphere labels, conduction speed/delays — shared by every network-level module.

**Interface:**

- `Connectome(K, weights, tract_lengths, centres, region_labels, hemispheres, speed, delays, region_index)` — frozen dataclass; `__post_init__` validates every array's shape against `n_nodes = len(region_labels)`.
- `.delay_steps(dt) -> npt.NDArray[np.int64]` — conduction delays as integer step counts for integration step `dt`.
- `.from_config(dict) -> Connectome` — loads TVB's bundled structural connectivity (`Connectivity.from_file()`), overrides `speed`, and derives `delays`.
- error modes: `ValueError` in `__post_init__` if any array's shape doesn't match `(n_nodes, n_nodes)` / `(n_nodes, 3)` / `(n_nodes,)` as appropriate.

This is a data-holding module: its "interface" is mostly a validated bundle of arrays plus one small derived-quantity method (`delay_steps`), not a behavioral seam with alternative implementations.

**Seam:** none — single concrete dataclass, not a point of substitution.
**Adapters:** n/a (`connectome.py` is the only source of `Connectome` instances, all built from TVB's bundled connectivity file).

**Depends on (modules):** Predictor Configuration Schemas & Loading (`StrictConfig`); TVB (`tvb.datatypes.connectivity.Connectivity`, external).
**Requires (config/data):** `speed`, `K` (both configurable; structural arrays themselves come from TVB's bundled connectivity file, not from config).

**Depended on by:** Jansen-Rit Plant (`JansenRitDynamics`, `resting_state`), TVB Reference Simulator (`build_reference_simulator`), Seizure Spread (region indices for EZ/PZ, hemisphere masks), Simulation Ensemble Generation & Scoring, Stimulation Model.

### Geometry

**Files:** `src/neuro/geometry.py`

**Responsibility:** Coordinate-frame utilities for scalp electrode/sensor geometry — converting TVB's sensor file into the connectome's coordinate frame and into MNI RAS for external head models.

**Interface:**

- `SENSORS_FILE` — the TVB sensor data filename constant, shared by `neuro.eeg` and `neuro.tvb_reference`.
- `EXTRACEPHALIC_ELECTRODES_MM` — hand-calibrated positions (mm) for two off-head return electrodes.
- `sensor_positions_mm(sensors_file=SENSORS_FILE, radius=SCALP_RADIUS_MM) -> (labels, positions)` — scalp electrode positions in the connectome's (anterior, left, superior) frame, on a sphere of radius `radius`.
- `centres_to_mni_ras(coords) -> FloatArray` — converts connectome-frame coordinates to MNI RAS; described in the module's own docstring as "the registration seam every external head model is reached through."
- error modes: none explicit; relies on TVB's `SensorsEEG.from_file` for data loading.

Despite the docstring's use of the word "seam," this is a pure coordinate-transform module (two functions plus two constants) with a single implementation each — no alternative adapters exist behind these functions in this codebase.

**Seam:** none (in this document's Module/Adapter sense) — single implementation per function.
**Adapters:** n/a

**Depends on (modules):** TVB (`tvb.datatypes.sensors.SensorsEEG`, external).
**Requires (config/data):** TVB's bundled `eeg_unitvector_62.txt.bz2` sensor file.

**Depended on by:** EEG Measurement (`SENSORS_FILE`), TVB Reference Simulator (`SENSORS_FILE`), Stimulation Model (`analytical.py`: `EXTRACEPHALIC_ELECTRODES_MM`, `SENSORS_FILE`, `sensor_positions_mm`).

### EEG Measurement

**Files:** `src/neuro/eeg.py`

**Responsibility:** Build the EEG forward/leadfield operator from TVB's surface projection and region mapping, and expose it as a measurement component that collapses Jansen-Rit network state to scalp EEG.

**Interface:**

- `build_eeg_leadfield() -> (leadfield, channel_labels)` — collapses TVB's `(62, 16384)` surface projection to region-level `(62, 76)` by summing each region's mapped vertex columns; mirror-corrects channel order (see `_mirror_partner_permutation`).
- `focal_channels(leadfield, region, k=4) -> indices` — the `k` channels loading hardest on a given region.
- `_mirror_partner_permutation(locations)` — corrects TVB's sensor-file/leadfield left-right mirror mismatch (private helper, documented in detail because the bug it fixes is non-obvious).
- `EEGMeasurement(n_nodes=None, selected_channels=None)` — a `__call__(t, x, u) -> FloatArray` measurement component mapping the raw network state `x` (reshaped to `(6, n_nodes)`) through the leadfield to an EEG channel vector, optionally subset to `selected_channels`.
- `EEGMeasurement.from_config(dict) -> Self` — builds it from a validated config.
- error modes: `selected_channels` given by string label raises `KeyError` if the label isn't a known channel (no explicit re-raise/message).

**Seam:** a sensor/measurement seam consumed by the `simulate` orchestrator's `sensors.measurement` slot (external `simulate.sensor.GaussianSensor` wraps a `measurement` callable of this shape); within this repo only one measurement adapter exists.
**Adapters:**

- `neuro.eeg.EEGMeasurement` — network-state-to-scalp-EEG via the leadfield (only measurement adapter present in this repo).

**Depends on (modules):** Geometry (`SENSORS_FILE`), Jansen-Rit Plant (`lfp`, used as `regional_lfp`); TVB (`tvb.datatypes.projections.ProjectionSurfaceEEG`, `region_mapping.RegionMapping`, `sensors.SensorsEEG`, external).
**Requires (config/data):** TVB's bundled projection matrix and region-mapping files; optional `n_nodes` truncation and `selected_channels` list.

**Depended on by:** TVB Reference Simulator (`_PROJECTION_FILE`, `_REGION_MAPPING_FILE`, `_mirror_partner_permutation`), Simulation Ensemble Generation & Scoring (`build_eeg_leadfield`, `focal_channels`), simulation configs via `class_path: neuro.eeg.EEGMeasurement` (e.g. `configs/simulation/jansen_rit_baseline.yaml`, wired under `sensors.measurement`).

### Seizure Spread

**Files:** `src/neuro/seizure.py`

**Responsibility:** Define the EZ/PZ seizure regime (which regions are excitable, by how much) and measure how a seizure spreads through a run — per-region recruitment onsets, seizure burden, and a scored comparison to a target spread schedule.

**Interface:**

- `build_seizure_a_gains(connectome) -> FloatArray`, `ez_pz_indices(connectome) -> list[int]` — construct the per-region excitatory-gain vector and EZ+PZ region indices for the seizure regime.
- `SpreadProfile(times, ptp, onsets, threshold)` — frozen dataclass of the per-region amplitude envelope and recruitment onsets; `.from_ptp(times, ptp, threshold, persist_s)` derives onsets from a stored envelope; `.seizing()`, `.n_seizing(nodes=None)`, `.recruited_by(t, nodes=None)`, `.seizure_state()`, `.burden()` read off recruitment/suppression metrics.
- `spread_profile_from_lfp(y, dt, *, window_s, hop_s, threshold) -> SpreadProfile` — measures recruitment from an already-recorded region LFP.
- `spread_profile_from_states(x_traj, dt, ...) -> SpreadProfile` — same, from a logged state trajectory of shape `(n_samples, 6, n_nodes)`.
- `spread_profile(dyn, duration, ...) -> SpreadProfile` — runs a `JansenRitDynamics` plant itself (via `simulate_network`, in hop-sized chunks) and measures recruitment as it goes; mutates `dyn`.
- `SpreadSummary(t_ez, t_pz, t_left_half, frac_left, frac_right)` — reduces a `SpreadProfile` to the EZ→PZ→hemisphere spread schedule; `.score(duration)` scores distance from the target schedule (lower is better).
- `spread_summary(profile, connectome) -> SpreadSummary`.
- invariants / ordering: `SpreadProfile.from_ptp` requires the envelope to span at least the persistence window, else it raises rather than silently marking every region as never-seizing.
- error modes: `ValueError` from `SpreadProfile.from_ptp` and `spread_profile_from_lfp` when the run/envelope is shorter than the persistence/spread window.

**Seam:** none — this is a domain-metrics module (one implementation of "how to measure spread"), not a substitutable interface.
**Adapters:** n/a

**Depends on (modules):** Jansen-Rit Plant (`JansenRitDynamics`, `lfp`, `simulate_network`), Seizure Calibration (default thresholds/windows), Connectome (region indices, hemisphere masks, for `spread_summary`).
**Requires (config/data):** a `Connectome` (for region/hemisphere lookups); calibration constants from Seizure Calibration.

**Depended on by:** Closed-Loop Evaluation (`spread_profile_from_lfp`, `spread_profile_from_states`), Simulation Ensemble Generation & Scoring (`build_seizure_a_gains`, `spread_profile_from_lfp`, constants).

### Seizure Calibration

**Files:** `src/neuro/seizure_calibration.py`

**Responsibility:** Hold the spread-detection threshold/window constants shared by the simulator (Seizure Spread) and the analysis layer (Observable Metrics & Ensemble Scoring), as a leaf module with no `neuro` imports.

**Interface:**

- `SPREAD_WINDOW_S`, `SPREAD_HOP_S`, `SPREAD_PERSIST_S`, `SEIZURE_PTP_MV` — module-level float constants; no functions or classes.
- The module's own comment states its reason for being a separate leaf: `neuro.metrics` needs these constants without reaching through `neuro.seizure -> neuro.jansen_rit -> neuro.config`, which would otherwise create a cycle (`neuro.config` would become unable to import `neuro.metrics`).

This is a pure constants module with no behavioral interface; it doesn't map onto the Module/Interface/Adapter vocabulary beyond "a module other modules import constants from."

**Seam:** none.
**Adapters:** n/a

**Depends on (modules):** none (deliberately leaf: imports nothing from `neuro`).
**Requires (config/data):** none.

**Depended on by:** Seizure Spread (thresholds/windows), Observable Metrics & Ensemble Scoring (`SEIZURE_PTP_MV`, `SPREAD_WINDOW_S`).

### Closed-Loop Evaluation

**Files:** `src/neuro/closed_loop_eval.py`

**Responsibility:** Run a controller+plant closed-loop simulation across multiple seeds and score seizure-suppression proficiency, for use as an optimization objective (e.g. hyperparameter tuning).

**Interface:**

- `evaluate_closed_loop_suppression(trial_dir, eval_cfg) -> (score, summary)` — loads a base simulation config, points its controller at a trained predictor artifact (`trial_dir/model.npz`), runs one `simulate.simulation.Simulation` per seed in `eval_cfg.seeds`, and reduces each run's logged LFP/state trajectory to a `SpreadProfile`.
  - `score` = mean seizure burden across seeds + `eval_cfg.amplitude_weight * mean stimulation amplitude` (to minimize).
  - `summary` dict keys: `score`, `seizure_burden`, `suppressed_seeds`, `total_seeds`, `mean_amplitude`, `mean_delivered_charge`, `mean_seizing_regions`.
- invariants / ordering: the base sim config's `dynamics.log` is coerced to `"lfp"` if unset, so exactly one of `("dynamics", "lfp")` / `("dynamics", "x")` is always present in the logger's signals to compute the spread profile from; each seed run is logged to a scratch temp directory (`use_mmap=True`) so full trajectories don't stay resident.
- error modes: `FileNotFoundError` if `eval_cfg.simulation_config` or the model artifact don't exist; `RuntimeError` if the simulation's logger is missing after a run; `TypeError` if the controller has no `u_max` attribute.

**Seam:** none — a single orchestration function tying `simulate.simulation.Simulation`, Simulation Config Validation, and Seizure Spread together for one evaluation strategy (burden + amplitude scoring); no alternative scoring implementations exist here.
**Adapters:** n/a

**Depends on (modules):** Seizure Spread (`spread_profile_from_lfp`, `spread_profile_from_states`), Simulation Config Validation (`validate_simulation_config`); `simulate.config.load_config`, `simulate.simulation.Simulation` (external orchestrator).
**Requires (config/data):** a base simulation config file (`eval_cfg.simulation_config`), a trained predictor artifact (`model.npz` in `trial_dir`), an `eval_cfg` carrying `seeds`, `t_end`, `seizure_ptp_mv`, `max_seizing_regions`, `amplitude_weight`.

**Depended on by:** `scripts/sweep_nn_predictor.py` (Optuna-style hyperparameter sweep).

**Note on thin entries:** none among the eight modules in this section lack a describable interface entirely, but Connectome, Geometry, and Seizure Calibration are worth flagging for how thin their Interface/Seam sections are relative to the template's expectations — each is a single-implementation structural/data/constants module with no substitution point, described above rather than stretched to invent a seam.

---

## Data Processing, Infra & Support

### Standardization

**Files:** `src/neuro/transforms.py`

**Responsibility:** Fit and apply an affine (z-score) standardization to channel-wise data, generically over NumPy arrays or CasADi symbolic expressions.

**Interface:**

- `zscore(x, center, scale) -> TMath`
- `unzscore(z, center, scale) -> TMath`
- `Standardizer.fit(x, *, kind="standard"|"robust", global_scaling=False) -> Standardizer`
- `Standardizer.transform(x) -> TMath`
- `Standardizer.inverse_transform(z) -> TMath`
- `Standardizer.arrays(prefix) -> dict[str, FloatArray]`
- `Standardizer.from_arrays(mapping, prefix) -> Standardizer`
- invariants: `center`/`scale` are fit once from `(rows, C)` data and then frozen (the dataclass is `frozen=True`); a zero `scale` is replaced with `1.0` at fit time so `transform` never divides by zero.
- ordering: `fit` (or `from_arrays`) must run before `transform`/`inverse_transform`.
- error modes: none raised directly; malformed `arrays` mappings (missing keys) raise a `KeyError` from `from_arrays`.

**Seam:** `TMath` generic in `zscore`/`unzscore`/`Standardizer.transform` — the same formula runs whether `x` is a NumPy array or a CasADi `SX`/`MX` symbol.
**Adapters:**

- NumPy `FloatArray` — concrete numeric standardization.
- CasADi `ca.SX` / `ca.MX` — symbolic standardization, used when a symbolic model bridge builds a standardizer into its graph.

**Depends on (modules):** none.
**Requires (config/data):** none beyond the arrays passed to `fit`/`from_arrays`.
**Depended on by:** Trajectory Dataset Preparation, ESN Predictor, Symbolic Stepping Predictor, Symbolic Observable Forecaster, Observable-Space Predictor.

### Causal Filtering

**Files:** `src/neuro/filtering.py`

**Responsibility:** Design and apply causal Butterworth filters, and expose two of them as online `Estimator` components that low-pass a plant's measurement.

**Interface:**

- `design_lowpass_sos(fs, cutoff) -> FloatArray` — order-4 low-pass SOS, shape `(2, 6)`.
- `design_bandpass_sos(fs, band) -> FloatArray` — order-4 band-pass SOS, shape `(4, 6)`.
- `causal_filter(signals, sos) -> FloatArray` — applies `sos` along the last axis, `zi` seeded from sample 0 to remove the zero-state settling transient.
- `group_delay_s(sos, fs, *, freq_hz=0.0) -> float` — measured (not closed-form) group delay, computed per-section to stay numerically conditioned for narrow low-passes.
- `lowpass_filter(y, fs, cutoff_hz) -> FloatArray` — axis-0 low-pass from zero state.
- `antialias_filter(y, fs, downsample) -> FloatArray` — no-op at `downsample == 1`, otherwise a low-pass at the decimated Nyquist rate.
- `LowPassEstimator(dt, cutoff_hz)` / `.from_config(config)` / `.update(t, y_mea, u) -> (x_hat, LowPassEstimatorLog)`
- `AntiAliasEstimator(dt, downsample)` — subclass of `LowPassEstimator`, cutoff fixed to the decimated Nyquist rate.
- invariants: `causal_filter` and the `Estimator` subclasses are causal by construction (never zero-phase) — this is deliberate, since the filter must be reproducible sample-by-sample online.
- error modes: none raised directly; malformed `sos`/`fs` propagate as `scipy.signal` errors.

**Seam:** `simulate.estimator.Estimator[TLog]` (external) — the generic online-estimator interface `LowPassEstimator`/`AntiAliasEstimator` satisfy via `update`.
**Adapters:**

- `LowPassEstimator` — low-passes at an explicit `cutoff_hz`.
- `AntiAliasEstimator` — low-passes at the Nyquist rate implied by a `downsample` factor; reproduces the same filter Trajectory Dataset Preparation's `load_trajectory` applies before striding, so the controller's zero-order hold matches the predictor's training decimation.

**Depends on (modules):** `simulate.estimator.Estimator` (external base class).
**Requires (config/data):** plant sample time `dt`; `cutoff_hz` or `downsample`.
**Depended on by:** Observable Metrics & Ensemble Scoring (`causal_filter`, `design_bandpass_sos`, `design_lowpass_sos`, `group_delay_s`); Simulation Config Validation (recognizes `AntiAliasEstimator`/`LowPassEstimator` by class path).

### Observable Metrics & Ensemble Scoring

**Files:** `src/neuro/metrics.py`

**Responsibility:** Define candidate control observables (windowed and window-free reductions of multi-channel signals) and score how well they track and can be driven toward suppressing a modeled seizure state.

This one file carries two distinct responsibility clusters that are read and written together but do not share a common type; both are described here rather than split, since they live in one file:

**Interface — observable registry (candidate metrics):**

- `windowed(signals, fs, reduce, *, window_s, hop_s=DEFAULT_HOP_S) -> (times, values)` — causal trailing-window slider; the shared machinery every `Metric` runs through.
- `Metric(name, window_s, reduce)` — a registry entry; `__call__` scores signals per channel.
- `METRICS: dict[str, Metric]` — `block_ptp`, `line_length`, `eeg_ms`, `band_power`, `fc_strength`, `spectral_centroid`.
- `RawSeries(name, transform, latency)` and `RAW_SERIES: dict[str, RawSeries]` — window-free observables (`waveform`, `envelope`).
- `envelope(signals, fs, *, band=SEIZURE_BAND_HZ, smooth_hz=ENVELOPE_SMOOTH_HZ) -> FloatArray` — causal band-pass/rectify/low-pass amplitude envelope (deliberately not a Hilbert transform, which would leak the future).
- `envelope_latency_s(fs, ...) -> float`, `latency_s(name, fs) -> float`, `sample_at(signals, fs, times) -> FloatArray`, `raw_series_grid(duration_s, *, hop_s) -> FloatArray`.
- `seizure_state(region_lfp, fs, *, window_s, hop_s, threshold) -> (times, state)` — the ground truth the metrics are scored against.
- invariants: windows are strictly causal (`(t - window_s, t]`); `threshold`/`window_s` in `seizure_state` are calibrated together against Seizure Calibration constants and must not be changed independently.

**Interface — ensemble scoring:**

- `Ensemble(times, values, n_replicates)` — one metric's value across one branch/arm on the lookahead grid; rejects (in `__post_init__`) a row count that is not a whole number of `n_replicates`-sized states.
- `variance_ratio(values, n_replicates) -> FloatArray`, `sigma_ens(ens) -> FloatArray`
- `StateReadout` / `state_readout_r2(metric, state, *, n_bins=10) -> StateReadout`
- `Separability` / `separability(healthy, saturated) -> Separability`
- `Controllability` / `controllability(zero, stim, *, direction, gap) -> Controllability`
- `state_predictability_r2(ens, explained_var) -> FloatArray`, `state_predictability_snr_db(ens, explained_var) -> FloatArray`
- `coupling(zero_metric, stim_metric, zero_state, stim_state) -> FloatArray`
- `score_store(store, fs, *, metrics, channel_sets, n_replicates, hop_s, cutoff_hz=None, pool=True) -> (times, ensembles)` — single-pass scorer over a `(n_rollouts, n_channels, n_samples)` memory-mapped store.
- `state_store(store, fs, *, hop_s, window_s) -> (times, state)`
- `score_raw_store(store, fs, *, times, series, channel_sets, n_replicates) -> ensembles`
- invariants: `zero`/`stim` ensembles passed to `controllability`, and all four ensembles passed to `coupling`, must be row-aligned (same trajectories, replicates, seeds).
- error modes: `Ensemble.__post_init__` raises `ValueError` on a non-whole-number row count.

**Seam:** `Metric.reduce` / `RawSeries.transform` callables — the registry dispatch point through which `score_store`/`score_raw_store` are metric-agnostic.
**Adapters:** `_block_ptp`, `_line_length`, `_eeg_ms`, `_band_power`, `_fc_strength`, `_spectral_centroid` (registered as `Metric`s); `waveform`, `envelope` (registered as `RawSeries`).

**Depends on (modules):** Causal Filtering (`causal_filter`, `design_bandpass_sos`, `design_lowpass_sos`, `group_delay_s`); Seizure Calibration (constants); Signal Processing Utilities (`band_energy`, `compute_psd`).
**Requires (config/data):** a `(n_rollouts, n_channels, n_samples)` rollout store and matching `fs` for the store-scoring entry points.
**Depended on by:** Predictor Configuration Schemas & Loading (`METRICS`, `DEFAULT_HOP_S`, for `EegMsGeometry`'s defaults), Autoregressive State-Space Predictor.

### Spectral Cost Support

**Files:** `src/neuro/spectral.py`

**Responsibility:** Compute per-window periodograms and mean-square power on the geometry the MPC's spectral cost replicates symbolically, and load the healthy-baseline power envelopes that cost is scored against.

**Interface:**

- `compute_periodograms(y, *, fs, window, hop) -> FloatArray` — one-sided, density-scaled, periodic-Hann periodograms, shape `(n_windows, n_channels, window // 2 + 1)`, DC bin included and never averaged across segments.
- `hinge_penalty(power, reference) -> float` — mean squared one-sided log excess, `0.0` exactly when `power` never exceeds `reference`.
- `windowed_mean_square(y, *, window, hop) -> FloatArray` — time-domain twin of `compute_periodograms`, shape `(n_windows, n_channels)`.
- `MsEnvelope(power, fs, window, hop)` / `.load(path)` — reads `Pref_ms` from the `.npz` written by `scripts/build_healthy_psd.py`; raises `ValueError` if that key is absent.
- `PsdEnvelope(power, fs, window, hop)` / `.load(path)` — reads `Pref`; raises `ValueError` if the loaded bin count does not match `window // 2 + 1`.
- invariants: `MsEnvelope` and `PsdEnvelope` are measured independently (not derived from one another by Parseval) because the spectral cost leaves DC unscored while a time-domain mean square includes the offset.
- error modes: `ValueError` from both `.load` classmethods on a malformed or mismatched envelope file.

**Seam:** none — `MsEnvelope`/`PsdEnvelope` share the same four fields and load convention by design (companion envelopes written together), not because they satisfy a common protocol.
**Adapters:** n/a

**Depends on (modules):** none.
**Requires (config/data):** an `.npz` envelope file produced by `scripts/build_healthy_psd.py`.
**Depended on by:** Simulation Config Validation (`PsdEnvelope.load`, geometry cross-checks), Predictor Loss Terms, Observable-Space Predictor, MPC NLP Builder.

### Simulation Config Validation

**Files:** `src/neuro/validation.py`

**Responsibility:** Catch silent, cross-component mismatches in a simulation config (rates, anti-alias filters, predictor horizon/geometry, plant identity) before the run is built.

**Interface:**

- `validate_simulation_config(config: Mapping[str, Any]) -> None` — the sole entry point; raises `ConfigConsistencyError` on a hard mismatch, emits a `warnings.warn` on a soft one (horizon exceeded, off-plant predictor).
- `ConfigConsistencyError(ValueError)` — raised for: estimator/plant/sensor rate mismatch; anti-alias cutoff disagreeing with the predictor's training decimation; controller `dt` disagreeing with the predictor's native `dt`; spectral reference `dt`/window/hop mismatch; Observable geometry (fs, segment length, hop, channel count, reduced reference shape) mismatch.
- invariants: only checks couplings that fail *silently*; channel/electrode count mismatches are deliberately left to the components themselves, which already raise within the first few steps.
- error modes: `ConfigConsistencyError` (hard stop) vs. `warnings.warn` (soft, e.g. horizon or plant-fingerprint mismatch) — the two are used for different severities and are not interchangeable.

**Seam:** none — a single validation entry point over a raw config mapping, not a pluggable interface.
**Adapters:** n/a

**Depends on (modules):** Predictor Artifacts & Rollout Evaluation (`load_any_artifact`); Observable-Space Predictor (`ObservableArtifact`, `envelope_log_reference`, `load_envelope`); Provenance (`plant_fingerprint`, `TrainingProvenance`); Spectral Cost Support (`PsdEnvelope`).
**Requires (config/data):** the full simulation config mapping (`dynamics`, `sensors`, `estimator`, `controller` blocks).
**Depended on by:** the simulation entry point / CLI (not covered in this document).

### Provenance

**Files:** `src/neuro/provenance.py`

**Responsibility:** Fingerprint the plant a training run or predictor artifact was generated on, and warn when an excitation schedule's block boundaries don't line up with a predictor's decimation grid.

**Interface:**

- `plant_fingerprint(config: Mapping[str, Any]) -> str` — SHA-256 over the `dynamics` (minus `seed`/`log`) and `sensors` blocks.
- `data_plant_fingerprint(data_dir: Path) -> str | None` — recovers the generating config from a trajectory directory's single `*.yaml`, or `None` if it can't.
- `check_excitation_alignment(data_dir, downsample) -> None` — `warnings.warn`s if a `WaveformController`'s `ras`/`prbs` hold durations aren't whole multiples of the decimated step.
- `TrainingProvenance(cutoff_hz, plant_fingerprint)` — `.meta` (serializable dict), `.from_meta(mapping)` (round-trip through an artifact's `meta`).
- `training_provenance(data_files, cutoff_hz) -> TrainingProvenance`
- invariants: `_generating_config` only succeeds when a data directory holds exactly one `*.yaml`; an `experiments:` batch config's first entry is treated as the full run, the rest as seed-only overrides.
- error modes: none raised directly; `check_excitation_alignment` only warns, never raises.

**Seam:** none — plain functions and one data-carrying dataclass, no dispatch point.
**Adapters:** n/a

**Depends on (modules):** none (`numpy`, `yaml` only).
**Requires (config/data):** a simulation config mapping, or a trajectory directory holding one generating `*.yaml`.
**Depended on by:** Simulation Config Validation (`plant_fingerprint`, `TrainingProvenance`); Predictor Configuration Schemas & Loading (`check_excitation_alignment`, via `resolve_data_files`); Autoregressive State-Space Predictor, ESN Predictor, Observable-Space Predictor.

### Predictor Artifacts & Rollout Evaluation

**Files:** `src/neuro/artifacts.py`

**Responsibility:** Load a predictor artifact from disk by its recorded model type, build the matching symbolic model bridge, and score free-run rollout accuracy against held-out trajectories.

**Interface — artifact loading/dispatch:**

- `load_any_artifact(artifact_path) -> PredictorArtifact` — reads `meta["model_type"]` from the `.npz` and dispatches to `MLPArtifact.load` / `ESNArtifact.load` / `ObservableArtifact.load`; raises `ValueError` on an unrecognized `model_type`.
- `load_rollout_artifact(artifact_path) -> RolloutArtifact` — as above, but raises `TypeError` if the loaded artifact is an `ObservableArtifact` (it forecasts the Observable, never a waveform).
- `build_symbolic_model(art) -> SymbolicModel | ObservableModel` — branches on artifact type to build `ESNSymbolicModel`, `ObservableSymbolicModel`, or `NNSymbolicModel`.

**Interface — rollout scoring:**

- `rollout_batches(art, trajectories, steps, *, stride=25, start=None) -> Iterator[(y_pred, y_true)]` — primes and rolls out a whole t0 grid per trajectory in one batched call, shape `(n_windows, steps, n_channels)`.
- `accumulate_rollout_errors(art, trajectories, steps, *, stride=25, start=None) -> (sq_err, power, pred_power)`
- `nmse(sq_err, power) -> FloatArray` — the repo's single NMSE definition (`inf` where the reference is silent); every reported NMSE goes through this.
- `RolloutNMSE(pooled, per_step)` / `evaluate_rollouts(art, val_trajs, horizon, step_stride=25) -> RolloutNMSE`
- `window_energy(y, window_steps, hop_steps) -> FloatArray`
- `LogEnergyError(pooled, per_position)` / `evaluate_log_energy(art, val_trajs, horizon, *, window_steps, hop_steps, step_stride=25) -> LogEnergyError` — scores the quantity the MPC actually costs (log-ratio of windowed energy), resolved per window position rather than pooled first, so over/under-prediction can't cancel across windows; raises `ValueError` if `horizon < window_steps` or if no trajectory is long enough for one window.

**Seam:** `RolloutArtifact = MLPArtifact | ESNArtifact` — both expose `priming_steps`, `prime_many`, `rollout_many`, which `rollout_batches` calls through without branching on which.
**Adapters:** `MLPArtifact` (Autoregressive State-Space Predictor), `ESNArtifact` (ESN Predictor).

**Depends on (modules):** ESN Predictor, Symbolic Stepping Predictor, Observable-Space Predictor, Symbolic Observable Forecaster, Autoregressive State-Space Predictor; Type Aliases & Symbolic Model Protocols (`SymbolicModel`/`ObservableModel` as `build_symbolic_model`'s return type).
**Requires (config/data):** an artifact `.npz` path; validation trajectories as `list[tuple[FloatArray, FloatArray]]` (u, y pairs).
**Depended on by:** Simulation Config Validation (`load_any_artifact`), Controller.

### Observable Geometry

**Files:** `src/neuro/config.py` (subset: `ObservableGeometry` and its subclasses)

**Responsibility:** Define the single source of truth for the sample grid an Observable is reduced onto (Segment length, hop, per-channel width), shared by the offline target, the training loss, and the MPC cost.

**Interface:**

- `ObservableGeometry.segment_steps(fs) -> int`, `.hop_steps(fs) -> int`, `.n_frames(span_steps, fs) -> int`, `.frame_supports(span_steps, fs) -> tuple[(int, int), ...]`, `.n_values(fs) -> int`, `.check_span(span_steps, fs) -> None`.
- invariants: `check_span` must be called (and pass) before a geometry is used at a given span — it is the single place that raises `ValueError` for a span too short to hold one Frame, or a Frame Kernel wider than the frame count it would consume.
- error modes: `ValueError` from `check_span` (all subclasses); `StftGeometry` additionally validates in a Pydantic `model_validator` that `band_hz` is increasing.

**Seam:** `ObservableGeometry` base class (a `ClassVar[str] kind` plus the methods above); `neuro.types.ObservableModel.geometry` and `ObservableSpec.geometry()` both return one at this seam.
**Adapters:**

- `StftGeometry` — spectrogram geometry in samples (`n_segment`, `n_hop`, optional `band_hz`, bin pooling, a Frame Kernel of configurable width/shape).
- `EegMsGeometry` — trailing mean-square window geometry in seconds, defaulting to whatever Observable Metrics & Ensemble Scoring's `eeg_ms` metric uses when `window_s`/`hop_s` are left `None`.

**Depends on (modules):** Observable Metrics & Ensemble Scoring (`METRICS["eeg_ms"].window_s`, `DEFAULT_HOP_S`, as `EegMsGeometry`'s defaults).
**Requires (config/data):** an `fs` and a span length in steps at call time.
**Depended on by:** `ObservableSpec.geometry()`, `StftSpec.geometry()`, `EegMsSpec.geometry()` (same file); Simulation Config Validation (`_check_observable_geometry`, via `ObservableArtifact.geometry`); Type Aliases & Symbolic Model Protocols (`ObservableModel`, typing only).

### Optuna Search Parameters

**Files:** `src/neuro/config.py` (subset: `ParamSpec` and its members)

**Responsibility:** Represent one Optuna hyperparameter search dimension as a validated, discriminated config value that knows how to suggest itself from a trial.

**Interface:**

- `.suggest(trial: optuna.Trial, name: str) -> Any` — the only method; each variant calls the matching `trial.suggest_*`.
- `ParamSpec = Annotated[CategoricalParam | IntParam | FloatParam | LogUniformParam, Field(discriminator="type")]` — a Pydantic discriminated union keyed on each variant's literal `type` field.
- invariants: `IntParam`/`FloatParam`/`LogUniformParam` (via `_RangeParam`) reject `high < low` at construction; `LogUniformParam` additionally requires `low`/`high > 0`.
- error modes: Pydantic `ValidationError` on a bad discriminator value or an inverted range.

**Seam:** the shared `.suggest(trial, name)` method, dispatched through the `type` discriminator rather than `isinstance`.
**Adapters:** `CategoricalParam`, `IntParam`, `FloatParam`, `LogUniformParam`.

**Depends on (modules):** none (`optuna.Trial`, type-checking only).
**Requires (config/data):** an `optuna.Trial` at suggest-time.
**Depended on by:** `NNSweepConfig.model`/`.training`, `ESNSweepConfig.model` (same file, as `dict[str, ParamSpec]`); whatever runs the Optuna sweep.

### Predictor Configuration Schemas & Loading

**Files:** `src/neuro/config.py` (remainder)

**Responsibility:** Strictly-validated Pydantic schemas for the NN- and ESN-predictor pipelines' YAML configs, plus the functions that load, resolve, and cross-check them.

**Interface:**

- `StrictConfig` — base class: `extra="forbid"`, `frozen=True`, `protected_namespaces=()`; every schema in this module inherits it.
- `ModelConfig`, `SimulationConfig`, `TrainingConfig`, `LossSpecs` (+ `LossSpec`/`SecondsSpanSpec`/`CurriculumMSESpec`/`StftSpec`/`EegMsSpec`), `ObservableSpec`, `NNSweepConfig`, `ClosedLoopEvalConfig`, `NNPredictorConfig` — the NN pipeline's schema tree, cross-validated by several `@model_validator(mode="after")` hooks (warmup < epochs, exactly one of `losses`/`observable` configured, span/bin-range feasibility at the resolved `fs`, sweep keys existing and not overlapping the base config).
- `ESNModelConfig`, `ESNTrainingConfig`, `ESNSweepConfig`, `ESNPredictorConfig` — the parallel ESN pipeline schema tree.
- `parse_array(val) -> Any` — resolves a config value that is (or points to) an `.npy`/`.npz`/`.npv` array.
- `load_config(path) -> NNPredictorConfig`, `load_esn_config(path) -> ESNPredictorConfig` — read + validate YAML; raise `FileNotFoundError` if `path` doesn't exist.
- `resolve_data_files(config, data_path_override=None) -> list[str]` — resolves and globs `.npz` training files; raises `ValueError` if no `data_path` is given or no files are found; also runs `check_excitation_alignment`.
- `resolve_artifact_dir(artifact, default_prefix) -> Path` — defaults to a timestamped `artifacts/<prefix>_<timestamp>` directory and creates it.
- `expand_dotted_dict(flat) -> dict[str, Any]` — expands `{'a.b.c': v}` into nested mappings (for CLI/sweep overrides).
- invariants: every schema is `frozen=True` and rejects unknown keys; `NNPredictorConfig` requires exactly one of `training.losses` / `observable`; at least one configured loss must have `start_epoch == 0`.
- error modes: `pydantic.ValidationError` from schema construction/validators; `FileNotFoundError` from the two `load_*` functions; `ValueError` from `resolve_data_files` and several cross-field validators.

**Seam:** none beyond the two called out above (Observable Geometry, Optuna Search Parameters) — this is otherwise a declarative schema tree plus loader functions, not a pluggable interface.
**Adapters:** n/a for this subset.

**Depends on (modules):** Observable Metrics & Ensemble Scoring (`DEFAULT_HOP_S`, `METRICS`, via `EegMsGeometry`); Provenance (`check_excitation_alignment`).
**Requires (config/data):** a YAML config file on disk for `load_config`/`load_esn_config`; an `.npz` data directory for `resolve_data_files`.
**Depended on by:** the training/sweep entry points; Predictor Loss Terms, Observable-Space Predictor, Controller, Connectome, Jansen-Rit Plant, Stimulation Model (all via `StrictConfig`).

### Type Aliases & Symbolic Model Protocols

**Files:** `src/neuro/types.py`

**Responsibility:** Pure type-level module — array aliases and two structural `Protocol`s that symbolic model bridges must satisfy; no runtime behavior of its own.

This file has no independent "how it works," only "what a caller must implement or accept." It is documented here as the seam definitions it hosts, since both are genuinely dispatched over elsewhere (`neuro.artifacts.build_symbolic_model`).

- `FloatArray = npt.NDArray[np.float64]`, `IntArray = npt.NDArray[np.intp]`, `StrArray = npt.NDArray[np.str_]` — the repo's array-shape aliases (see `AGENTS.md`: use these, not raw `npt.NDArray[...]`).

**Seam 1 — `SymbolicModel` Protocol:** state/step/output over a CasADi graph for autoregressive rollout (`state_shape`, `n_controls`, `n_channels`, `native_horizon`, `is_linear`, `f_step`, `f_out`, `step`, `output`, `initial_state`, `absorb`, `is_ready`).
**Adapters:** `NNSymbolicModel`, `ESNSymbolicModel` (Symbolic Stepping Predictor; referenced from `neuro.artifacts.build_symbolic_model`).

**Seam 2 — `ObservableModel` Protocol:** forecasts an Observable over the Control Horizon in one shot rather than stepping (`state_shape`, `n_controls`, `n_channels`, `n_values`, `fs`, `native_horizon`, `geometry`, `n_frames`, `forecast`, `f_forecast`, `initial_state`, `absorb`, `is_ready`). Deliberately shares seven member names with `SymbolicModel` but stays an unrelated Protocol: it omits the stepping members (`step`, `f_step`, `is_linear`, `output`, `f_out`) on purpose, so a model missing them fails at build time rather than silently degrading the autoregressive path.
**Adapters:** `ObservableSymbolicModel` (Symbolic Observable Forecaster).

**Depends on (modules):** Observable Geometry (typing only, `TYPE_CHECKING`).
**Requires (config/data):** none.
**Depended on by:** Predictor Artifacts & Rollout Evaluation (`build_symbolic_model`'s return type); every concrete `SymbolicModel`/`ObservableModel` adapter.

### Plotting

**Files:** `src/utils/plotting.py`

**Responsibility:** Render multi-channel EEG/LFP signals, receding-horizon multistep prediction fans, and PSDs to Matplotlib figures.

**Interface:**

- `plot_signals(signals, dt_ms, *, channel_names=None, channels_to_plot=None, stacked=True, offset_scale=1.5, title=..., color=..., ax=None) -> (fig, ax)` — stacked waterfall or overlaid channel view; raises `ValueError` if `signals` isn't 2-D.
- `plot_multistep_predictions(y_true, y_pred, dt, *, channels=None, channel_names=None, stride=None, anchors=None, anchors_in_seconds=False, t_start=None, t_end=None, connect_to_truth=True, show_anchor_markers=True, cmap=None, title=..., ylabel=..., true_kwargs=None, pred_kwargs=None, figsize=None, axes=None) -> (fig, axes)` — one stacked subplot per channel, prediction "fans" branching off the true trace at each anchor; raises `ValueError` on inconsistent `y_true`/`y_pred` shapes/ndim or non-positive `dt`.
- `plot_psd(signals, dt_ms, *, channel_names=None, channels_to_plot=None, plot_mean=False, max_freq=None, normalize=False, nperseg=None, ax=None) -> (fig, ax)` — per-channel or mean±SD spectrum; raises `ValueError` if `signals` isn't 2-D.
- invariants: all three accept an optional pre-existing `ax`/`axes` to draw into, else create a new figure; channel selection/naming defaults are handled by a shared private helper (`_filter_channels`).
- error modes: `ValueError` on malformed input shapes (each function validates its own).

**Seam:** none — a fixed set of plotting functions, not a pluggable interface.
**Adapters:** n/a

**Depends on (modules):** Signal Processing Utilities (`compute_psd`, used by `plot_psd`).
**Requires (config/data):** in-memory arrays only; no files.
**Depended on by:** not traced in this document.

### Signal Processing Utilities

**Files:** `src/utils/processing.py`

**Responsibility:** Shared low-level signal-processing primitives (PSD, band energy, transient trimming, synchrony) used by both the metrics/plotting layers.

**Interface:**

- `compute_psd(signals, dt_ms, nperseg=None) -> (freqs, psd)` — Welch PSD, `nperseg` defaults to ~1 Hz resolution (`min(n_samples, round(fs))`); raises `ValueError` if `signals` isn't 2-D; returns empty arrays if `n_samples == 0`.
- `band_energy(signals, dt_ms, *, band=(0.0, 50.0), nperseg=None, normalize=True) -> FloatArray` — detrends, computes PSD, trapezoidal-integrates over `band`, optionally normalizes so the strongest channel is `1.0`; raises `ValueError` if `signals` isn't 2-D.
- `steady_window(signals, dt_ms, transient_ms) -> FloatArray` — drops a leading transient along the last axis.
- `synchronization(activity) -> float` — mean off-diagonal Pearson correlation across nodes; `NaN` if fewer than two channels.
- error modes: `ValueError` from `compute_psd`/`band_energy` on non-2-D input.

**Seam:** none — plain functions, no dispatch point.
**Adapters:** n/a

**Depends on (modules):** none.
**Requires (config/data):** in-memory arrays only.
**Depended on by:** Observable Metrics & Ensemble Scoring (`compute_psd`, `band_energy`); Plotting (`compute_psd`).

### Thesis Plot Persistence

**Files:** `src/utils/save_plots.py`

**Responsibility:** Persist a Matplotlib figure (or named dict of figures) to disk as pickle/PGF/PNG alongside JSON metadata and optional raw data, formatted for direct LaTeX inclusion.

**Interface:**

- `ThesisPlotSaver(textwidth_pt=418.25, base_dir="artifacts")` — on construction, configures global `matplotlib.rcParams` for serif/LaTeX-sized fonts, switching to the `pgf` backend with `text.usetex=True` only if `pdflatex` is found on `PATH` (else warns and falls back to the default backend).
- `.calculate_dimensions(fraction=1.0, subplots=(1, 1)) -> (width_in, height_in)` — golden-ratio figure size scaled to the LaTeX text width, so embedded fonts don't distort.
- `.save(fig, name, metadata=None, *, data=None, overwrite=False) -> None` — writes `<dir>/<name>/<name>.{pkl,pgf,png,json}` (plus `.npz` if `data` is given); `metadata` is stamped with `git_commit` and `generated_at`; if `overwrite=False` and the target directory exists, appends a `_NN` suffix instead of overwriting; a `dict[str, Figure]` is saved as one file set per key.
- invariants: PGF export failure is caught and only warns (`RuntimeWarning`), never raises — PNG/pickle export is expected to succeed regardless of whether `pdflatex` is present.
- error modes: none raised directly by `.save`; PGF save failures are swallowed into a warning.

**Seam:** none — a single concrete saver, not a pluggable interface.
**Adapters:** n/a

**Depends on (modules):** Type Aliases & Symbolic Model Protocols (`FloatArray`, typing only); shells out to `git rev-parse HEAD` (falls back to `"not_a_git_repository"` if unavailable).
**Requires (config/data):** an on-disk `base_dir` (created as needed); `pdflatex` on `PATH` for true LaTeX-rendered PGF output (optional).
**Depended on by:** thesis-figure-generating scripts.
