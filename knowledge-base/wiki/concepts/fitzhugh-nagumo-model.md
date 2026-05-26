---
title: FitzHugh-Nagumo Model
type: concept
tags: [neuron-model, nonlinear-dynamics, oscillators]
sources: [chouzouris-et-al-2021.md]
created: 2026-05-14
updated: 2026-05-14
---

# FitzHugh-Nagumo Model

The FitzHugh-Nagumo (FHN) model is a simplified version of the Hodgkin-Huxley model that describes the prototype of an excitable system (e.g., a neuron). It is widely used in neuroscience to model the average neural activity of a brain region.

## Key Points

- **Dynamics:** The model captures key features of neural activity, including excitability, fixed points (low and high activity), and limit cycles (oscillations). [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#Abstract | Chouzouris et al. 2021]]
- **Bifurcations:** FHN oscillators can reproduce the transitions from one dynamical regime to another, which is essential for describing brain state transitions. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#I. Introduction | Chouzouris et al. 2021]]
- **Large-Scale Networks:** When coupled in a network (e.g., using a human connectome), FHN oscillators can simulate whole-brain dynamics and synchronization phenomena. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#I. Introduction | Chouzouris et al. 2021]]
- **Mathematical Form:** The model consists of two coupled ordinary differential equations:
  $$ \begin{aligned} \dot{x}_1 &= R(x_1) - x_2 + \mu \\ \dot{x}_2 &= \frac{1}{\tau} (x_1 - \delta x_2) \end{aligned} $$
  where $x_1$ is the activity variable (fast), $x_2$ is the recovery variable (slow), $\mu$ is the background input, and $R(x) = -\alpha x^3 + \beta x^2 - \gamma x$ defines the nonlinear excitability. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#III. RESULTS | Chouzouris et al. 2021]]

## Related Concepts

- [[concepts/neural-mass-model.md|Neural Mass Model]] — FHN is often used as the local dynamics in neural mass models of large-scale brain networks.
- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Used to steer FHN network dynamics.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Application domain for FHN-based whole-brain network control.

## Sources

- [[sources/chouzouris-et-al-2021.md|Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators]] — Uses FHN oscillators as nodes in a whole-brain network model.
