---
title: Optimal Nonlinear Control
type: concept
tags: [control-theory, optimization, nonlinear-systems]
sources: [chouzouris-et-al-2021.md, salfenmose-obermayer-2022.md]
created: 2026-05-14
updated: 2026-05-14
---

# Optimal Nonlinear Control

Optimal nonlinear control is an optimization-based framework for deriving control policies that steer nonlinear dynamical systems toward a desired state or trajectory while minimizing a cost functional.

## Key Points

- **Cost Functional:** Typically penalizes the deviation from the desired network dynamics (precision), the control energy used, and sometimes the spatial sparsity of the control inputs. A general form of the cost functional $\mathcal{F}$ is:
    $$ \mathcal{F} = F_P + W_1 F_1 + W_2 F_2 $$
    where $F_P$ is the precision term (e.g., tracking error), and $F_1, F_2$ are weighted penalties on control effort (e.g., $L_1$ or $L_2$ norms). [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (I. Introduction)]]
- **Nonlinear Dynamics:** Unlike linear control theory, this approach accounts for the complex behaviors of neural systems, such as bifurcations, multistability, and limit cycles. [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (I. Introduction)]]
- **State-Dependency:** The effectiveness and optimal strategy of control are highly dependent on the system's current location in state space. [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (Abstract)]]
- **Sparsity and Energy Constraints:**
  - **L1-norm (Sparsity):** Penalizing the 1-norm of the control signal encourages "sparse" solutions, often targeting only a single population or node [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (2.2.1.)]].
  - **L2-norm (Energy):** Penalizing the 2-norm (squared amplitude) distributes control effort to minimize absolute values, often resulting in multi-channel control [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (2.2.1.)]].
- **Minimal Intervention Strategy:** In noiseless bistable systems, the optimal control signal is often a finite pulse that pushes the system just across the basin of attraction boundary. The system then relaxes naturally to the target state [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (3.2.)]].

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — A specific implementation of optimal control that uses a receding horizon.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Application of optimal control to modulate brain activity.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — A biophysically realistic model type often used with nonlinear control.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Node dynamics model in whole-brain networks steered via optimal nonlinear control.
- [[concepts/human-connectome.md|Human Connectome]] — Structural backbone of the network; topology influences optimal control strategies.

## Sources

- [[sources/chouzouris-et-al-2021.md|Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators]] — Application of the framework to state-switching and synchronization in a whole-brain connectome model.
- [[sources/salfenmose-obermayer-2022.md|Nonlinear optimal control of a mean-field model of neural population dynamics]] — Detailed analysis of optimal control in a two-population neural mass model.
