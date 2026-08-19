# Language

## Plant & Electrophysiology

**Plant**:
The ground-truth whole-brain biophysical simulator running delayed Jansen-Rit neural mass differential equations over a structural connectome.
_Avoid_: Model

**Local Field Potential (LFP)**:
The source-space regional signal representing pyramidal somatic-dendritic potential differences.
_Avoid_: EEG, Cortical Voltage

**(Raw) EEG**:
Sensor-space voltage recordings obtained by projecting regional LFPs through the leadfield matrix.
_Avoid_: Surface Signal, Channel Observation, LFP, scalp EEG

**Leadfield**:
The forward head volume conduction matrix mapping source-space regional LFPs to sensor-space EEG channels.
_Avoid_: Field Projection, Gain Matrix, Sensor Forward Matrix

**Epileptogenic Zone (EZ)**:
Brain regions with autonomously hyper-excitable dynamics that initiate seizure activity.
_Avoid_: Seizure Core, Primary Focus, Trigger Node

**Propagation Zone (PZ)**:
Brain regions with intermediate excitability recruited secondary to seizure spread from the epileptogenic zone.
_Avoid_: Spread Area, Secondary Focus, Margin Zone

## Transcranial Electrical Stimulation (tES)

**Control Current/current**:
External electrical currents delivered through physical scalp electrodes, constrained by Kirchhoff's Current Law.
_Avoid_: Stimulation Voltage, Input Voltage, Drive

**Stimulation Drive**:
Regional somatic polarization perturbations induced in neural populations via the spatial field projection.
_Avoid_: Control Input, Applied Current, Electrode Drive

**Field Projection**:
The spatial transfer operator mapping electrode currents to regional somatic stimulation drives.
_Avoid_: Leadfield, Spatial Matrix, Gain Matrix

## Predictors

**Predictor/prediction model**:
A learned model used by the controller to forecast future EEG measurements.
_Avoid_: Plant Model, System Model, Estimator

**State Absorption**:
The causal single-step operation where a surrogate predictor ingests the latest measurement and applied control into its internal state.
_Avoid_: State Estimation, Filtering, Observer Step, Teacher Step

**Priming**:
Feeding a historical sequence of measurements and controls into an uninitialized surrogate predictor to bring its internal state to a valid prediction baseline.
_Avoid_: Seeding, History Injection, Washout, State Warm-up

**Priming Steps**:
The minimum number of history samples a surrogate predictor must absorb before its state is valid for prediction or control.
_Avoid_: Washout Steps, History Lag, Delay Buffer Size, Burn-in Steps

**Warm-up Period**:
The initial closed-loop simulation phase during which the controller outputs zero control while the surrogate predictor primes its state.
_Avoid_: Burn-in Period, Transient Phase, Settling Time

## Seizure Dynamics & Ground Truth

**(Regional) Seizure Threshhold**:
The condition on source-space local field potential peak-to-peak amplitude over a trailing time window that classifies a single brain region as seizing.
_Avoid_: Seizure Level, Seizure Threshold, Regional Trigger

**Seizure State**:
The instantaneous network-wide fraction of brain regions meeting the regional seizure criterion at a given time step.
_Avoid_: Seizure Level, Seizure Extent, Seizure Fraction, Seizure Severity

**Seizure Burden**:
The continuous time-average of the seizure state over the duration of a simulation run.
_Avoid_: Cumulative Seizure, Total Seizure Time, Seizure Cost, Seizure Area

## Closed-Loop Control & Execution

**Control Horizon**:
The number of discrete future time steps over which the optimal control problem is solved.
_Avoid_: Prediction Window, Planning Lookahead

**Seizure Suppression**:
Active real-time attenuation of seizure amplitude and prevention of recruitment into the propagation zone under current and rate constraints.
_Avoid_: Seizure Cancellation, Seizure Quenching, Seizure Abatement

**Control Budget**:
The hard constraints on maximum per-electrode current, total current, and rate of change.
_Avoid_: Power Constraints, Energy Limits, Current Bounds

## Objectives, Losses & Scores

**Observable**:
A causal sensor-space signal or windowed reduction derived from scalp EEG channels available in real time to the controller.
_Avoid_: Control Metric, Feature, Sensory Input, Observed State

**Loss**:
A differentiable scalar objective minimized during offline gradient-based training of surrogate predictors.
_Avoid_: Cost, Training Cost

**Cost**:
The mathematical objective minimized over the control horizon by the receding-horizon controller at each decision step.
_Avoid_: Loss, Predictor Objective, Optimization Metric

**Segment**:
The fixed-length slice of a trajectory fed to a single Fourier transform, used identically by the spectral training loss and the receding-horizon controller's spectral cost.
_Avoid_: Window, Analysis Window, nperseg, Block

**Frame**:
The spectrum a single segment produces, indexed by its position on the hop grid.
_Avoid_: Segment, Slice, Bin, Column

**Frame Kernel**:
The non-negative smoother applied to power along the frame axis before the log. Buys estimator degrees of freedom, not frequency resolution, and is therefore distinct from lengthening a segment.
_Avoid_: Smoothing Window, Averaging Window, Filter

**Closed-Loop Evaluation Score**:
A scalar performance criterion computed post hoc over simulation runs to rank controller configurations.
_Avoid_: Controller Cost, Run Loss, Evaluation Loss, Benchmark Metric
