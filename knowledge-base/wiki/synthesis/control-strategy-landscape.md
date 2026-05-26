---
title: Control Strategy Landscape
type: synthesis
tags: [control-theory, synthesis, comparison]
sources: [chang-et-al-2020.md, chouzouris-et-al-2021.md, froehlich-jezernik-2005.md, salfenmose-obermayer-2022.md, steffen-cannon-2025.md]
created: 2026-05-16
updated: 2026-05-16
---

# Control Strategy Landscape

Across the five sources in this wiki, four distinct control paradigms appear for closed-loop neurostimulation. Each makes different assumptions about model availability, biological scale, and real-time requirements.

## The Four Paradigms

### 1. Cascaded Feedback Linearization + MPC

**Source:** [[sources/froehlich-jezernik-2005.md|Fröhlich & Jezernik 2005]]
**Model:** [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley]] (mechanistic, single-neuron)
**Target:** Direct biophysical state control — ion channel activation ($m$) and inactivation ($h$)
**Approach:** Inner loop cancels HH nonlinearities via [[concepts/feedback-linearization.md|feedback linearization]] (reducing error dynamics to $\dot{e} + Ke = 0$); outer [[concepts/model-predictive-control.md|MPC]] layer optimally tracks ion channel references with hard constraints.
**Key insight:** Even highly nonlinear biophysical models become tractable by combining local linearization with a model-based optimizer.
**Limitation:** Requires exact model parameters; valid only within the operating range of the linearization.

### 2. Optimal Nonlinear Control on Whole-Brain Networks

**Source:** [[sources/chouzouris-et-al-2021.md|Chouzouris et al. 2021]]
**Model:** Whole-brain [[concepts/fitzhugh-nagumo-model.md|FHN]] network with DTI-derived [[concepts/human-connectome.md|connectome]] (mechanistic, large-scale)
**Target:** Attractor switching between multistable network states; network synchronization
**Approach:** Minimize a cost functional over a full control trajectory using gradient-based optimization. Offline, open-loop computation.
**Key insight:** Node importance for control is task-dependent and state-dependent; linear controllability metrics do not predict nonlinear control effort.
**Limitation:** Offline computation limits real-time applicability; requires full network state knowledge.

### 3. Optimal Nonlinear Control on Neural Mass Models

**Source:** [[sources/salfenmose-obermayer-2022.md|Salfenmose & Obermayer 2022]]
**Model:** EI EIF [[concepts/neural-mass-model.md|neural mass model]] (biophysically grounded, population-level)
**Target:** Bistable state switching (down→up, up→down)
**Approach:** Gradient descent via adjoint method; L1 vs L2 regularization of control signal.
**Key insight:** Optimal strategy in noiseless bistable systems is a minimal pulse just crossing the basin boundary — the system then relaxes naturally. L1 promotes sparse single-channel control; L2 promotes distributed multi-channel control.
**Limitation:** Gradient-based offline optimization; not directly adaptive to real-time state changes.

### 4. MPC with Black-box Volterra Model

**Source:** [[sources/chang-et-al-2020.md|Chang et al. 2020]]
**Model:** NARMA [[concepts/volterra-model.md|Volterra model]] (data-driven, black-box)
**Target:** Suppression of epileptiform EEG discharges ([[concepts/seizure-suppression.md|seizure suppression]])
**Approach:** System identification from experimental data; receding-horizon [[concepts/model-predictive-control.md|MPC]] optimizes stimulation in real-time.
**Key insight:** Bypassing mechanistic model knowledge makes the approach clinically practical; robustness to disturbances is a core benefit.
**Limitation:** Volterra series can have high parameter counts; may not generalize across patients without re-identification.

### 5. DCNN Tube MPC

**Source:** [[sources/steffen-cannon-2025.md|Steffen & Cannon 2025]]
**Model:** [[concepts/difference-of-convex-functions.md|Difference of Convex Neural Networks]] (data-driven, structured nonlinear)
**Target:** [[concepts/beta-band-oscillations.md|Beta-band]] suppression in [[concepts/parkinsons-disease.md|Parkinson's Disease]] via closed-loop DBS
**Approach:** Multi-step DCNN predictor trained on patient LFP data; sequential convex programming for online optimization; robust Tube MPC handles linearization errors.
**Key insight:** Combining neural network expressiveness with convex optimization tractability outperforms linear MPC and PI controllers — even when the DCNN was pre-trained on *different* patients.
**Limitation:** Requires offline training; limited model interpretability.

## Synthesis: Key Tradeoffs

| Dimension | FL + MPC | ONC (Network) | ONC (NMM) | Volterra MPC | DCNN Tube MPC |
|---|---|---|---|---|---|
| Model type | Mechanistic | Mechanistic | Mechanistic | Data-driven | Data-driven |
| Scale | Single neuron | Whole brain | Population | Population | Population |
| Real-time | Yes | No | No | Yes | Yes |
| Constraint handling | Hard | Soft | Soft | Hard | Hard |
| Patient-specific | Yes | Yes | Yes | Yes | No (generalizable) |
| Core technique | Feedback linearization | Adjoint method | Adjoint method | System ID | DCNN + SCP |

## Emerging Thesis

The field is evolving along two axes simultaneously:

1. **Scale**: single-neuron (HH) → population (NMM/FHN) → whole-brain networks
2. **Model knowledge**: mechanistic first-principles → data-driven black-box

The most recent work (Steffen & Cannon 2025) sits at the frontier: data-driven, population-scale, real-time, and patient-generalizable. The central open question is whether the generalization advantage holds in prospective clinical settings with hardware-in-the-loop constraints.

## Related Concepts

- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Overarching framework for all strategies.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Shared algorithmic framework for strategies 1, 4, and 5.
- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Core framework for strategies 2 and 3.
- [[concepts/feedback-linearization.md|Feedback Linearization]] — Inner-loop component of strategy 1.
- [[concepts/volterra-model.md|Volterra Model]] — Predictive model for strategy 4.
- [[concepts/difference-of-convex-functions.md|Difference of Convex Functions]] — Model architecture for strategy 5.
