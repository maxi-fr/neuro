---
title: Feedback Linearization
type: concept
tags: [control-theory, nonlinear-control, feedback-control]
sources: [froehlich-jezernik-2005.md]
created: 2026-05-14
updated: 2026-05-14
---

# Feedback Linearization

Feedback linearization is a common method for controlling nonlinear systems by transforming them into equivalent linear differential equations through a change of variables and a suitable control law.

## Key Points

- **Application to Neural Dynamics:** In the context of [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley]] dynamics, feedback linearization can be used to cancel out the nonlinear ionic current terms, allowing the design of a linear state feedback controller for the membrane voltage [[raw/Froehlich_Jezernik_2005/Froehlich_Jezernik_2005.md#2.3. State feedback controller for control of membrane voltage | Fröhlich & Jezernik 2005]].
- **Controller Design:** By applying this method, the tracking error dynamics can be reduced to a simple linear form (e.g., $\dot{e} + Ke = 0$, where $e = V_{ref} - V$), where the performance is determined by a feedback gain $K$. This allows the use of linear control tools to achieve desired convergence rates. [[raw/Froehlich_Jezernik_2005/Froehlich_Jezernik_2005.md#2.3. State feedback controller for control of membrane voltage | Fröhlich & Jezernik 2005]]

## Related Concepts

- [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley Model]] — A primary target for feedback linearization in neurostimulation.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Often used in cascade with a linearization layer.

## Sources

- [[sources/froehlich-jezernik-2005.md|Feedback control of Hodgkin–Huxley nerve cell dynamics]] — Employs feedback linearization as the inner loop of a cascaded controller.
