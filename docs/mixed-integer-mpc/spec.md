# Mixed-integer MPC

## Problem Statement

The current receding-horizon controllers can reduce stimulation effort, but their continuous Control Current decisions do not directly favor steps with exactly zero stimulation. The new controller should choose zero-current steps more often while preserving Seizure Suppression. In paired 12-second Plant runs on seeds 7000, 7001, 7002, 7004, and 7005, the target is fewer than five final seizing regions on at least four seeds. Total delivered charge and energy are secondary measures. This target is a Closed-Loop Evaluation Score, not a constraint in the optimal control problem.

## Solution

Add a CasADi controller alongside the trajopt controller. It solves a continuous problem with IPOPT and two mixed-integer variants with Bonmin. Each Control Horizon step has one binary activation decision shared by all electrodes. If the decision is zero, all Control Currents for that step are zero. If it is one, the currents may vary within their per-electrode bounds. The controller applies the first zero-order-held Control Current and solves again after the next State Absorption.

One mixed-integer variant limits the number of enabled steps within each Control Horizon. The other adds a weighted enabled-step count to the existing Cost. They are separate modes. Both retain the chosen continuous Cost, including any current-effort terms. The horizon-local limit imposes no run-wide duty-cycle guarantee.

## Implementation Decisions

- The new controller uses the existing configuration pattern and controller update contract. It exposes solve success, solver status, solve time, applied and planned Control Currents, predicted outputs, and decomposed Cost contributions. Mixed-integer logs also expose the planned activation decisions.
- Continuous CasADi and mixed-integer CasADi share the same Predictor transcription, Cost, per-electrode amplitude bounds, and explicit sum-to-zero Kirchhoff constraint. The new implementation uses single shooting. It does not use reduced current coordinates.
- Waveform and Observable Predictors each advance at their native controller step. Support both MLP and CNN checkpoints in each representation. Preserve the current normalization, residual connection, activation, history update, and physical-output mapping of each Predictor. For softplus activations in CasADi graphs, evaluate the shifted log-sum-exp formulation ($\max(z, 0) + \log(\exp(-\max(z, 0)) + \exp(z - \max(z, 0)))$) to prevent floating-point overflow and NaN gradients when $z > 709$.
- Support every Cost option exposed by the current waveform and Observable problem builders: waveform tracking with its optional terminal weight, the corresponding spectral or Observable hinge, quadratic current effort, and smooth L1 current effort. Preserve the current Frame geometry, normalization, and terminal scoring. A standalone Cost not exposed by either builder is outside this scope. For the waveform spectral hinge Cost, transcribe the STFT via explicit constant Fourier projection matrices and Hann tapering, applying a numerical floor ($10^{-12}$) prior to the logarithm to avoid infinite gradients from $\log(0)$.
- At each step, multiply each electrode's existing positive and negative amplitude bounds by the shared binary activation decision. Require the sum of electrode currents to equal zero at that step. The per-electrode bounds are the tight coupling constants. An enabled step is allowed to carry zero current.
- Configure either a horizon-local maximum enabled-step count or an enabled-step Cost weight. Reject configurations specifying both. Continuous mode uses neither. The existing Cost remains the suppression proxy; the final regional seizure count is evaluated on the Plant after a run.
- Shift successful continuous currents and binary decisions to initialize the next solve. Before the first successful solve, use a feasible initial guess. Do not apply a control from a failed solve: expose its status and stop that seed's run. The first implementation has no per-solve deadline or fallback policy.
- Treat a feasible integer solution with successful Bonmin status as a usable result. Record status and objective without claiming global optimality. Measure solver construction and per-update solve time so later work can decide whether a time limit is needed.
- Preserve the existing physical constraints: per-electrode amplitude bounds and Kirchhoff balance. No minimum on/off duration, switching limit, current change bound, or run-wide stimulation budget is introduced.

## Testing Decisions

- Test behavior at the configured-controller seam: construct each mode through configuration, feed measurements through State Absorption and `update`, and check the emitted physical Control Current, plan, log, and failure behavior. Existing controller tests provide the prior pattern.
- For every Predictor family, compare CasADi and the current JAX/trajopt Predictor on identical fixed histories and Control Current sequences. Check one-step and Control Horizon outputs, including the shape and timing of Observable Frames. These fixed-sequence checks isolate transcription errors from optimizer differences.
- Compare each CasADi Cost contribution and the total Cost against the current implementation on fixed predicted sequences, including terminal and whole-horizon Frame terms. Check amplitudes, Kirchhoff balance, activation coupling, and the horizon-local count directly on solved plans.
- On small, well-behaved continuous cases, compare the first control and Cost returned by CasADi IPOPT and trajopt's IPOPT within numerical tolerances. On full checkpoints, require valid constraints and comparable Cost, and record differences between local solutions rather than requiring identical sequences.
- Run paired closed-loop comparisons on the five specified seeds with the same Plant setup and duration. Report final seizing regions, whether the 4/5 Seizure Suppression target is met, enabled-step count, delivered-current duty cycle in seconds, charge, energy, solver failures, and solve times. Report both enabled steps and applied nonzero-current steps with a stated numerical zero threshold.
- Stop a seed's run on solver failure and preserve the failure status in the result. Tests should observe the controller and run outcomes, not its internal symbolic graph or decision-vector layout.

## Out of Scope

- A hard Seizure Suppression constraint, a whole-run duty-cycle cap, and real-time guarantees.
- A sweep design for the activation limit or weight. The first work delivers both formulations and their measurements.
- Continuous pulse-onset timing, sub-step events, pulse-shape optimization, and additional physical switching constraints.
- Multiple shooting, reduced Kirchhoff coordinates, and a relaxation-and-rounding alternative.
- Restoring the old CasADi controller wholesale or replacing the trajopt controller.

## Further Notes

Use "active interval" for one controller step with stimulation enabled. It is not a physiological pulse. An enabled interval may carry zero current, so evaluate realized stimulation from applied physical currents as well as from the binary plan. Predictor artifacts used for the final Plant comparison can be selected when that comparison is run; synthetic or existing test checkpoints can establish transcription parity beforehand.
