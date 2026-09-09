# Dynamics Output Functions and Output Costs Specification

## Problem Statement

Optimal control problems in `trajopt` historically evaluated objectives and constraints directly on state coordinates $\mathbf{x} \in \mathbb{R}^n$ and control coordinates $\mathbf{u} \in \mathbb{R}^m$. In closed-loop neurostimulation, however, Predictor models maintain state representations that pack internal execution memory:
- Autoregressive models pack sliding history windows of standardized measurements and applied Control Currents into states with dimensions up to $\mathbf{x} \in \mathbb{R}^{1580}$.
- Biophysical whole-brain Jansen-Rit models pack differential equation states, delayed axonal coupling buffers, and simulation step indices into states with dimensions exceeding $\mathbf{x} \in \mathbb{R}^{600}$.

In both paradigms, the physical quantities of clinical interest—sensor-space Raw EEG, regional Local Field Potentials (LFPs), and spectral Observable log-power Frames—are lower-dimensional observations $\mathbf{y} = g(\mathbf{x}, \mathbf{u}, t) \in \mathbb{R}^p$.

Lacking first-class output function support in `trajopt`, the codebase implemented several ad-hoc workarounds:
1. Padded state-space weights: Tracking costs on the Waveform Predictor required constructing dense or diagonal weight vectors of dimension $n$ (e.g. 960) filled with zeros except for the newest measurement slice, scaling diagonal entries by squared standardizer scales to counteract state standardization, and mapping physical target references into standardized state coordinates.
2. Bespoke state-space tracking costs: The Jansen-Rit Predictor required a dedicated 55-line cost class (`JansenRitTrackingCost`) to manually unpack ODE states, compute LFPs, project through the Leadfield gain matrix, and evaluate quadratic tracking error.
3. Auxiliary decoding wrappers: Whole-horizon spectral costs (`SpectralHingeCost`, `ObservableFrameHingeCost`) required separate adapter classes (`StateOutputs`, `JansenRitStateOutputs`) to extract and rescale output trajectories from state trajectories.

Following `trajopt` ADR 0008, `trajopt` supports native dynamics output functions $\mathbf{y} = g(\mathbf{x}, \mathbf{u}, t) \in \mathbb{R}^p$ on `AbstractModel` and composable `OutputCost` adapters. This repository should adopt these capabilities to eliminate architectural boilerplate and streamline optimal control problem assembly.

## Solution

1. Equip all Predictor models implementing `DiscreteDynamics` with the `trajopt` output interface: declare static output dimension `p` and implement `output(self, x, u=None, t=0.0) -> jax.Array`.
2. Adopt `trajopt.costs.output.OutputCost` to evaluate quadratic tracking objectives and Observable Frame hinge penalties directly on output space $\mathbb{R}^p$.
3. Eliminate ad-hoc output extraction modules (`StateOutputs`, `JansenRitStateOutputs`) and bespoke tracking cost classes (`JansenRitTrackingCost`), delegating output decoding directly to the model's native `output` and `evaluate_output` methods.

## Implementation Decisions

### 1. Dynamics Models Output Contract

Every discrete Predictor model implementing `DiscreteDynamics` will define its compile-time output dimension `p: int` and implement pointwise output evaluation:

- **Autoregressive Shift-Register Predictor**:
  - Declares output dimension $p$ matching the unstandardized output width ($n_{\text{outputs}}$).
  - Evaluates `output(x, u, t)` by slicing the newest standardized block from the state window, un-standardizing it via the model's center and scale parameters, and returning the physical signal.
  - For Waveform Predictors, this returns the physical Raw EEG sample vector of shape $(n_{\text{channels}},)$.
  - For Observable Predictors, this returns the physical STFT Observable log-power Frame of shape $(n_{\text{channels}} \cdot n_{\text{values}},)$.

- **Nullspace-Reduced Predictor**:
  - Inherits output dimension $p$ from the underlying base Predictor.
  - Evaluates `output(x, u, t)` by projecting reduced controls $\mathbf{v} \in \mathbb{R}^{m-1}$ through the null-space basis $Z$ into physical Control Currents $\mathbf{u} = Z \mathbf{v}$, then delegating to the base model's `output`. When the base model exhibits no control feedthrough, control projection is omitted.

- **Jansen-Rit Whole-Brain Predictor**:
  - Declares output dimension $p$ matching the sensor-space channel count ($n_{\text{channels}}$).
  - Evaluates `output(x, u, t)` by extracting somatic pyramidal and inhibitory post-synaptic potentials from the ODE state slice, calculating regional Local Field Potentials, and projecting them through the Leadfield matrix to produce sensor-space Raw EEG.

### 2. Output Tracking Cost Refactoring

- **Jansen-Rit Quadratic Tracking**:
  - The custom `JansenRitTrackingCost` class is retired.
  - The tracking objective is formulated as a standard `DiagonalCost.tracking` defined on output space $\mathbb{R}^p$, penalized against the physical reference target.
  - The cost is wrapped in `OutputCost(model, stage_tracking)` for stage costs and `stage_tracking.as_terminal()` for terminal costs.

- **Waveform Quadratic Tracking**:
  - Waveform tracking in problem assembly is refactored from full state space $\mathbb{R}^n$ ($n = 960$) to output space $\mathbb{R}^p$ ($p = n_{\text{channels}}$).
  - Eliminates zero-padded $Q$ state vectors, manual index slicing of the newest buffer element, and state-standardization unscaling operations.
  - Uses `DiagonalCost.tracking` with diagonal weight vector $Q \in \mathbb{R}^p$ set to $2 w_y / \text{horizon}$ against the unstandardized healthy reference mean, wrapped in `OutputCost`.

### 3. Observable Frame Hinge Cost Simplification

- For Observable Predictors stepping on the Frame grid, each knot state transition produces an STFT Observable log-power Frame.
- The stagewise hinge penalty against the healthy reference envelope is formulated as a generic hinge CostFunction operating directly on $\mathbb{R}^p$, wrapped in `OutputCost(model, hinge_on_output)`.
- Terminal knot pricing uses `stage_cost.as_terminal()` or an explicit terminal instance on $\mathbb{R}^p$, preserving exact whole-horizon scoring in combination with initial-knot exclusion.

### 4. Retirement of Auxiliary Decoding Adapters

- `StateOutputs` and `JansenRitStateOutputs` are deprecated and removed.
- Whole-horizon spectral costs (`SpectralHingeCost` and `ObservableFrameHingeCost`) accept the dynamical model directly instead of an auxiliary decoding container.
- Stage trajectory decoding within `stage_costs(X, U, t)` is performed by vectorizing model output evaluation across the stage states via `jax.vmap(model.output)(X)` or `model.evaluate_output(trajectory)`.

### 5. Architectural Clarifications and Exclusions

- **Whole-Horizon Waveform STFT Remains in Stage Costs**:
  Evaluating Fourier transforms across a waveform rollout requires gathering time segments along the trajectory axis ($N_{\text{segment}}$ knots) and hopping along the time grid. Because `trajopt.AbstractModel.output` is strictly a pointwise map from one knot state to one output ($x_k \mapsto y_k$), STFT generation cannot be embedded into `output()` on the Waveform Predictor. Whole-horizon spectral objectives on waveform trajectories remain whole-horizon functionals evaluated within `stage_costs`.
- **Output Constraints and Simulation Measurement Bridges**:
  `OutputConstraint` and `TrajOptMeasurement` are not adopted at this time, as the repository's control limits operate exclusively on Control Currents and the biophysical Plant simulation operates via independent measurement pipelines.

## Testing Decisions

### Test Characteristics and Integrity
- Tests must assert external numerical behavior and optimization invariants, never internal implementation details.
- Parity with established baselines: Controller-commanded Control Currents and reported Cost values on fixed benchmark trajectories must match incumbent golden values within tight float tolerances ($10^{-4}$ or better).
- Dynamic consistency: Linearization Jacobians derived via automatic differentiation through `output_state_jacobian` must match finite-difference checks.

### Modules to Test
1. **Predictor Output Seams**:
   - Verify `output()` on `WaveformMLPModel`, `ObservableMLPModel`, and `JansenRitModel` matches canonical decoding from known states.
   - Verify `NullspaceReducedModel.output()` agrees with the base model under expanded Control Currents.
2. **Cost Equivalence**:
   - Verify `OutputCost` wrapping quadratic output tracking matches the incumbent Waveform and Jansen-Rit tracking cost evaluations on identical trajectories.
   - Verify that `SpectralHingeCost` and `ObservableFrameHingeCost` produce identical whole-horizon penalty values when decoding through `model.output`.
3. **End-to-End Optimal Control and Parity**:
   - Verify `build_waveform_problem`, `build_observable_problem`, and `build_jansen_rit_problem` assemble successfully and solve to optimality with configured solvers.
   - Run existing golden MPC solve tests (`test_reproduces_mpc_controller_control_sequence`, `test_migrated_config_reproduces_incumbent_end_to_end`, `test_jansen_rit_mpc_problem_solve`) to ensure zero regression in closed-loop execution.

### Prior Art
- `tests/test_mpc.py`: Golden control sequence and reported cost parity tests against pinned CasADi and single-shooting baselines.
- `tests/test_cost.py`: Unit verification of spectral hinge, Observable Frame hinge, and smooth L1 control penalty reductions.
- `tests/test_jansen_rit_jax.py`: Verification of Jansen-Rit state packing, Heun integration, tracking cost evaluation, and closed-loop MPC stepping.

## Out of Scope

1. **Output Constraints (`OutputConstraint`)**:
   Adding sensor-space voltage bounds or regional seizure-threshold state bounds to the OCP constraint set is out of scope.
2. **Simulation Observation Map Replacement (`TrajOptMeasurement`)**:
   Refactoring the ground-truth biophysical Plant or `simulate` sensor pipeline to use `trajopt` observation maps is out of scope.
3. **Predictor Checkpoint Schema Changes**:
   Serialization formats and checkpoint schemas (`to_checkpoint`, `from_checkpoint`) remain unchanged.

## Further Notes

- Adopting `OutputCost` simplifies problem builders and eliminates hundreds of unused zeros in quadratic weight structures, reducing memory overhead during compilation and Jacobian evaluation.
- Capitalize `CONTEXT.md` domain glossary terms throughout docstrings and comments (Plant, Observable, Frame, Control Horizon, Control Current, Seizure Suppression).
