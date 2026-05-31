---
title: Neurostimulation Approaches Synthesis
type: synthesis
tags: [neurostimulation, control-theory, comparison, mpc, optimal-control]
sources: [chang-et-al-2020.md, chouzouris-et-al-2021.md, froehlich-jezernik-2005.md, salfenmose-obermayer-2022.md, steffen-cannon-2025.md]
created: 2026-05-18
updated: 2026-05-18
---

# Neurostimulation Approaches Synthesis

This page synthesizes the five primary neurostimulation approaches documented in the wiki, following a standardized structure for comparison across control objectives, controller design, plant models, and state estimation.

## Approach 1: Cascaded Feedback Linearization + MPC

**Source:** [[summaries/froehlich-jezernik-2005.md|Fröhlich & Jezernik 2005]]

### Control objective

* **High level:** Control biophysical state variables to generate action potentials, suppress unwanted oscillations, or block (annihilate) action potential propagation in axons.
* **Low level:** Tracking desired trajectories for sodium channel activation ($m$) and inactivation ($h$).

### Controller

* **MPC / OC:**
  * **Model:** Mechanistic HH equations linearized around specific biophysical operating points (e.g., $V=20$ mV).
    * **System identification/parameter estimation:** Prediction model is the **same** as the plant "ground truth" model (mechanistic HH parameters are assumed known).
  * **Constraints:** Incorporates hard biophysical constraints such as membrane voltage and conductance limits.
  * **Cost function:** Minimizes tracking error while weighting the control signal to ensure "biological plausibility."
* **Cascaded control structure?** Yes. An inner loop uses [[concepts/feedback-linearization.md|feedback linearization]] to cancel HH nonlinearities, while an outer MPC layer determines the optimal membrane voltage.

### Plant Model

* **Scope:** Single-cell to multicompartment (axon).
* **Inputs:** Stimulation current or membrane voltage.
* **Outputs:** Direct biophysical state variables ($V, m, h, n$).

### State Estimation

* **Method:** Not explicitly detailed; assumes full state feedback for the linearization and predictive layers.

---

## Approach 2: Optimal Nonlinear Control on Whole-Brain Networks

**Source:** [[summaries/chouzouris-et-al-2021.md|Chouzouris et al. 2021]]

### Control objective

* **High level:** Targeted attractor switching between multistable network states (e.g., transitioning from a pathological to a healthy state) and increasing network synchronization.
* **Low level:** Steering the system between low- and high-activity fixed points in a large-scale network.

### Controller

* **MPC / OC:** [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] using offline gradient-based optimization (4th order Runge-Kutta).
  * **Model:** Whole-brain network of [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo oscillators]] coupled via a structural connectome.
    * **System identification/parameter estimation:** Prediction model is the **same** as the plant "ground truth" model (FHN node dynamics and structural connectome are assumed known).
  * **Constraints:** Soft constraints enforced via the cost functional, including penalties for spatial nonsparsity.
  * **Cost function:** Minimizes deviations from the target dynamics, total control energy, and the number of control sites (nonsparsity).
* **Cascaded control structure?** No.

### Plant Model

* **Scope:** Whole brain (large-scale network).
* **Inputs:** Node-specific stimulation inputs $\mathbf{u}_i$ targeting specific cortical areas.
* **Outputs:** Average neural activity (voltage-like state variables of FHN oscillators) for each node.

### State Estimation

* **Method:** Assumes full network state knowledge (offline, open-loop analysis).

---

## Approach 3: Optimal Nonlinear Control on Neural Mass Models

**Source:** [[summaries/salfenmose-obermayer-2022.md|Salfenmose & Obermayer 2022]]

### Control objective

* **High level:** Efficiently switching between bistable states (down-state to up-state and vice-versa) in neural populations.
* **Low level:** Determining the minimal intervention pulse required to steer the system across the basin of attraction boundary.

### Controller

* **MPC / OC:** [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] using gradient descent via the adjoint method.
  * **Model:** EI EIF [[concepts/neural-mass-model.md|neural mass model]] (mean-field firing rate equations).
    * **System identification/parameter estimation:** Prediction model is the **same** as the plant "ground truth" model (mechanistic biophysical parameters are assumed known).
  * **Constraints:** Soft constraints via cost function regularization.
  * **Cost function:** Balances target accuracy against control strength, using L1-norm for sparsity (single-population control) and L2-norm for energy (distributed control).
* **Cascaded control structure?** No.

### Plant Model

* **Scope:** Population level (neural mass).
* **Inputs:** Stimulation current to excitatory (E) or inhibitory (I) populations.
* **Outputs:** Population firing rates ($r_E, r_I$).

### State Estimation

* **Method:** Offline optimization using the adjoint method; assumes model parameters are known.

---

## Approach 4: MPC with Black-box Volterra Model

**Source:** [[summaries/chang-et-al-2020.md|Chang et al. 2020]]

### Control objective

* **High level:** [[concepts/seizure-suppression.md|Seizure suppression]] through the elimination of epileptiform EEG discharges.
* **Low level:** Regulating EEG signal outputs to follow non-epileptiform reference patterns.

### Controller

* **MPC / OC:** [[concepts/model-predictive-control.md|Model Predictive Control]] using a receding horizon strategy.
  * **Model:** Nonlinear Auto-Regressive Moving-Average (NARMA) [[concepts/volterra-model.md|Volterra model]].
    * **System identification/parameter estimation:** Prediction model is **not** the same as the plant "ground truth" model (Plant is NMM; Prediction is a data-driven NARMA Volterra proxy).
  * **Constraints:** Hard constraints on stimulation amplitude and rate of change.
  * **Cost function:** Optimizes stimulation waveforms over a future horizon to minimize tracking error and energy consumption.
* **Cascaded control structure?** No.

### Plant Model

* **Scope:** Population level (simulated via NMM).
* **Inputs:** Stimulation input $u$ (representing TES or DBS).
* **Outputs:** EEG signal (summed post-synaptic membrane potentials).

### State Estimation

* **Method:** Real-time feedback provided directly by measurable EEG signals.

---

## Approach 5: DCNN Tube MPC

**Source:** [[summaries/steffen-cannon-2025.md|Steffen & Cannon 2025]]

### Control objective

* **High level:** [[concepts/beta-band-oscillations.md|Beta-band]] suppression in [[concepts/parkinsons-disease.md|Parkinson's Disease]] to alleviate motor symptoms.
* **Low level:** Tracking a desired beta-band envelope level ($y_0$), specifically minimizing pathological "over-shoots" $[y(t) - y_0]_{\geq 0}$.

### Controller

* **MPC / OC:** Robust Tube [[concepts/model-predictive-control.md|MPC]] using Sequential Convex Programming (SCP).
  * **Model:** Difference of Convex Neural Networks (DCNN) built with [[concepts/difference-of-convex-functions.md|Input-Convex Neural Networks (ICNN)]].
    * **System identification/parameter estimation:** Prediction model is **not** the same as the plant "ground truth" model (Plant is actual patient LFP; Prediction is a data-driven DCNN proxy).
  * **Constraints:** Robust "tubes" handle linearization errors and modeling uncertainty to ensure stability and constraint satisfaction.
  * **Cost function:** Minimizes pathological beta-band activity and stimulation energy $Ru(t)^2$.
* **Cascaded control structure?** No.

### Plant Model

* **Scope:** Population level (LFP beta-band envelope).
* **Inputs:** Deep brain stimulation (DBS) amplitude $u(t)$.
* **Outputs:** LFP beta-band envelope $y(t)$.

### State Estimation

* **Method:** Real-time feedback from LFP beta-band power; robust tubes provide a buffer against estimation and modeling errors.
