# Healthy Operating Point Reference Tracking

## Problem Statement

Quadratic tracking Costs evaluate the squared deviation of the Observable from zero over the Control Horizon:
$w_y \sum_k \|y_k\|^2$. However, zero is not the healthy operating point of the neural Plant. In the Jansen-Rit
biophysical neural mass model, the resting Local Field Potential (LFP) $x_2 - x_3$ sits at approximately $+1.5\text{ mV}$
with a healthy fluctuation standard deviation of only $0.16\text{ mV}$. Consequently, approximately $99\%$ of the
quadratic tracking Cost evaluated during healthy background activity reflects the static operating point offset, while
only $1\%$ represents the dynamic fluctuations the Cost is intended to shape.

Under hard Control Budget bounds on per-electrode Control Current, the receding-horizon controller expends its
authority attempting to null an unattainable DC offset. Furthermore, the quadratic penalty treats healthy resting-state
dynamics with the same severity as pathological seizure activity. This issue is general to any Observable with a
nonzero operating point. In contrast, spectral and Observable hinge Costs do not suffer from this defect because they
score one-sided log excess of power spectral density over a healthy reference envelope and explicitly skip the DC bin.

## Solution

Generalize quadratic tracking Costs across biophysical and learned Predictor formulations to track a healthy operating
point reference:

$$w_y \sum_k \|y_k - y_{\text{ref}}\|^2$$

where $y_{\text{ref}}$ is the empirical per-region or per-channel mean evaluated over unperturbed healthy background
simulations.

Healthy operating point reference vectors are computed and stored in the healthy reference artifact alongside existing
spectral power envelopes. A unified healthy reference container encapsulates the artifact and supplies the appropriate
reference slices to each Cost. Problem builders and configuration schemas are unified so that a single reference
parameter supplies all active Costs (quadratic tracking, spectral hinge, and Observable hinge), with strict validation
requiring the reference whenever quadratic tracking is active.

## Implementation Decisions

1. **Empirical mean vectors stored in healthy reference artifact.**
   The reference generation pipeline computes the per-region Local Field Potential mean vector and the per-channel
   sensor EEG mean vector across unperturbed healthy background trajectories. These mean vectors are stored directly
   inside the existing compressed archive alongside the periodogram quantile envelope, the mean-square power envelope,
   and the Observable log-power Frame envelope.

2. **Unified reference container abstraction.**
   A single immutable reference container encapsulates the healthy reference archive. It loads the archive once and
   exposes read-only properties for:
   - the source-space Local Field Potential mean vector,
   - the sensor-space EEG mean vector,
   - the raw-waveform PSD envelope,
   - the Observable log-power Frame envelope, and
   - the time-domain mean-square envelope.
   For unit tests and synthetic benchmarking setups, the container can also be constructed directly from in-memory
   arrays without reading from disk.

3. **Model-space reference resolution in Jansen-Rit tracking Cost.**
   The Jansen-Rit tracking Cost is parameterized by a reference vector matching its output dimension. If the
   Predictor's output is source-space Local Field Potential (identity Leadfield), the Cost tracks the regional Local
   Field Potential mean vector. If the Predictor projects regional activity through a Leadfield to sensor-space EEG
   channels, the Cost tracks the sensor-space EEG mean vector (or the regional mean vector projected through the
   Leadfield). The stage Cost scores the horizon-scaled squared Euclidean residual between the predicted output and the
   reference vector.

4. **State-space reference translation in Waveform MPC.**
   Waveform optimal control problems operating on standardized autoregressive Predictor states compute the tracking
   target state vector by mapping the physical healthy reference through the Predictor's input standardizer center and
   scale vectors:

   $$x_f = \frac{y_{\text{ref}} - y_{\text{center}}}{y_{\text{scale}}}$$

   The diagonal quadratic tracking objective targets this healthy standardized operating point instead of centering
   on zero (which corresponded to $-y_{\text{center}} / y_{\text{scale}}$).

5. **Single unified reference parameter in problem builders.**
   Problem builders for biophysical Jansen-Rit models, learned waveform Predictors, and learned Observable Predictors
   replace fragmented reference path arguments with a single polymorphic parameter. The parameter accepts a path
   string, a Path object, or a pre-loaded reference container. Each active Cost (quadratic tracking, spectral hinge,
   Observable hinge) extracts its required reference slice from this single source.

6. **Strict reference requirement when tracking is active.**
   Whenever quadratic output tracking is assigned nonzero weight ($w_y > 0$) in an optimal control problem, the
   healthy reference is strictly required. Omitting the reference when tracking is enabled raises a configuration
   consistency error rather than silently falling back to tracking zero.

7. **Terminal Cost consistency.**
   The terminal tracking Cost in both biophysical and learned Predictor formulations targets the identical healthy
   reference vector as the stage Cost, applying the terminal tracking weight to penalize terminal deviations from the
   healthy operating point.

8. **Initial knot exclusion preserves invariance.**
   The wrapper that excludes the knot-0 absorbed measurement state from the reported stage Cost operates without
   modification: the knot-0 deviation from the healthy reference is constant with respect to future Control Currents
   and does not affect the optimal control sequence.

9. **Comparison suite and simulation configuration alignment.**
   Oracle MPC base configurations supply the healthy reference artifact in their controller problem definition. The
   Cost comparison arms inherit this reference; tracking arms sweep Control Horizon and effort weights against the
   healthy operating point, and redundant per-arm reference paths in spectral hinge arms are removed. Existing waveform
   simulation configurations with nonzero tracking weight specify the healthy reference artifact.

## Testing Decisions

- **What makes a good test**: Tests must verify externally observable behavior at functional boundaries—optimal
  control trajectories, computed control actions, solver convergence, and explicit error rejection—rather than asserting
  internal array slicing or private attributes.
- **Modules tested**:
  - Optimal control problem builders and cost evaluation modules.
  - Spectral reference extraction, serialization, and container modules.
  - Configuration consistency validation modules.
  - Closed-loop Receding-Horizon execution and simulation comparison runners.
- **Prior art**:
  - `tests/test_cost.py` for MPC cost evaluation, solver parity, and constraint enforcement.
  - `tests/test_jansen_rit_oracle.py` for oracle MPC problem assembly and state handover.
  - `tests/test_spectral.py` for healthy envelope extraction, window pooling, and serialization.
  - `tests/test_validation.py` for cross-component configuration validation.
- **Specific test scenarios**:
  1. *Zero control at healthy operating point*: Initialize an optimal control problem at the healthy operating point
     ($y = y_{\text{ref}}$) without external disturbances. Verify that the solver computes an optimal Control Current of
     zero (within numerical solver tolerance) and yields zero tracking stage cost.
  2. *Active suppression under seizure deviation*: Displace the initial state from $y_{\text{ref}}$ with a simulated
     seizure excursion. Verify that the solver mobilizes nonzero Control Current oriented to steer the trajectory back
     toward $y_{\text{ref}}$.
  3. *Error rejection without reference*: Attempt to assemble an optimal control problem with $w_y > 0$ while omitting
     the reference. Verify that a `ValueError` is raised with a descriptive error message.
  4. *Reference serialization round-trip*: Run the healthy reference pipeline over synthetic multi-seed trajectories,
     verifying that computed regional and channel means match exact analytical sample averages, and confirm that the
     unified container round-trips all spectral, Observable, and mean-square envelopes.
  5. *Configuration cross-validation*: Validate configurations where $w_y > 0$, ensuring that valid reference paths pass
     and missing references or mismatched channel counts are caught prior to simulation startup.

## Out of Scope

- Time-varying, periodic, or trajectory-dependent tracking references (the reference is a stationary operating point
  mean vector).
- Online adaptive estimation or filtering of the reference from running closed-loop measurements.
- Modifications to spectral hinge or Observable hinge Costs, which already evaluate excess log-power over healthy
  envelopes and exclude the DC bin.
- Retention of legacy zero-tracking comparison arms in evaluation suites.

## Further Notes

- The mathematical linearity of the forward Leadfield operator ensures that projecting a source-space regional mean
  vector through the Leadfield yields the exact sensor-space channel mean vector:
  $E[\text{EEG}] = E[G \cdot \text{LFP}] = G \cdot E[\text{LFP}]$.
- Pre-calculating statistics into the compressed reference artifact achieves a ~300× speedup at initialization (37 ms
  versus 10.6 s) and avoids the impracticality of tracking 15 GB of raw trajectory files in version control or test
  environments.
