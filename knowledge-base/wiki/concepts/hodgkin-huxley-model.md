---
title: Hodgkin-Huxley Model
type: concept
tags: [neuroscience, mathematical-modeling, action-potential, electrophysiology]
sources: [froehlich-jezernik-2005.md]
created: 2026-05-14
updated: 2026-05-14
---

# Hodgkin-Huxley Model

The Hodgkin-Huxley (HH) model is a phenomenological model of action potential generation in nerve cells. It describes the temporal evolution of the membrane voltage and the dynamics of voltage-gated ion channels.

## Key Points

- **Mathematical Structure:** The model consists of four coupled, nonlinear differential equations. The membrane voltage $V$ evolves according to:
  $$ C_{m} \dot{V} = G_{K} n^{4} (E_{K} - V) + G_{Na} m^{3} h (E_{Na} - V) + G_{L} (V_{L} - V) + I_{inj} $$
  The gating variables $y \in \{m, h, n\}$ follow first-order kinetics:
  $$ \dot{y} = \alpha_y(V)(1-y) - \beta_y(V)y = \frac{y_\infty(V) - y}{\tau_y(V)} $$
  where $m$ and $h$ describe sodium activation/inactivation, and $n$ describes potassium activation [[Literature/Froehlich_Jezernik_2005.pdf | Fröhlich & Jezernik 2005 (2.1. Single compartment HH model)]].
- **Ionic Currents:** It models ionic flow through sodium, potassium, and leakage channels. Each current depends on the membrane voltage and the specific state of the channel gates [[Literature/Froehlich_Jezernik_2005.pdf | Fröhlich & Jezernik 2005 (2.1. Single compartment HH model)]].
- **Action Potential Generation:** If a certain voltage threshold is exceeded, a positive feedback loop of sodium influx triggers an action potential, which is subsequently repolarized by sodium inactivation and delayed potassium outflux [[Literature/Froehlich_Jezernik_2005.pdf | Fröhlich & Jezernik 2005 (2.1. Single compartment HH model)]].
- **Spatial Propagation:** The model can be extended to multicompartment versions to simulate the propagation of action potentials along axons (e.g., node-to-node conduction in myelinated axons) [[Literature/Froehlich_Jezernik_2005.pdf | Fröhlich & Jezernik 2005 (2.2. Multicompartment model)]].

## Related Concepts

- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — HH models are often used as the "plant" in closed-loop control simulations.
- [[concepts/feedback-linearization.md|Feedback Linearization]] — A technique used to control the nonlinear HH dynamics.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Used as the outer control layer for optimal ion channel tracking.

## Sources

- [[sources/froehlich-jezernik-2005.md|Feedback control of Hodgkin–Huxley nerve cell dynamics]] — Uses the HH model to develop cascaded feedback control schemes for AP generation and annihilation.
