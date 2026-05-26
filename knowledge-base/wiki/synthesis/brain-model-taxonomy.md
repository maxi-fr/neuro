---
title: Brain Model Taxonomy
type: synthesis
tags: [modeling, neuroscience, synthesis, comparison]
sources: [chang-et-al-2020.md, chouzouris-et-al-2021.md, froehlich-jezernik-2005.md, salfenmose-obermayer-2022.md, steffen-cannon-2025.md]
created: 2026-05-16
updated: 2026-05-16
---

# Brain Model Taxonomy

Each source in this wiki uses a different mathematical representation of neural dynamics. These models differ in biological fidelity, spatial scale, and computational tractability. Understanding the taxonomy clarifies when each model type is appropriate.

## The Four Model Types

### 1. Hodgkin-Huxley Model (Single Neuron, Mechanistic)

**Used in:** [[sources/froehlich-jezernik-2005.md|Fröhlich & Jezernik 2005]]
**Scale:** Single neuron or axon segment
**Fidelity:** High — models individual ion channels (Na⁺, K⁺, leakage), gating dynamics, and action potential generation
**State variables:** Membrane voltage $V$, sodium activation $m$, inactivation $h$, potassium activation $n$
**Computational cost:** Low (4 ODEs per compartment), scales to multicompartment axons
**When to use:** Precise biophysical control of single-cell dynamics; mechanistic understanding of ion-channel-level interventions; in vitro validation.

### 2. FitzHugh-Nagumo Oscillator (Simplified Excitable System)

**Used in:** [[sources/chouzouris-et-al-2021.md|Chouzouris et al. 2021]]
**Scale:** Individual brain region (mesoscale), coupled into large networks via [[concepts/human-connectome.md|connectome]] weights
**Fidelity:** Medium — captures excitability, fixed points, and limit cycles without ion-channel specifics
**State variables:** Fast activity variable $x_1$, slow recovery variable $x_2$
**Computational cost:** Very low (2 ODEs per node), scales well to hundreds of coupled nodes
**When to use:** Whole-brain network simulations; studying synchronization, state transitions, and network-level controllability; when topology matters more than single-cell precision.

### 3. Neural Mass Model (Population-Level, Biophysical)

**Used in:** [[sources/chang-et-al-2020.md|Chang et al. 2020]], [[sources/salfenmose-obermayer-2022.md|Salfenmose & Obermayer 2022]]
**Scale:** Neural population (thousands of neurons)
**Fidelity:** Medium-high — derived from mean-field approximations; captures excitatory/inhibitory interactions, bistability, and EEG-like dynamics
**State variables:** Population firing rates $r_E$, $r_I$; governed by $\tau \dot{r} = -r + \Phi(I)$
**Computational cost:** Low-moderate (2–8 ODEs per population); analytically tractable
**When to use:** Epilepsy research (seizure simulation), Parkinson's disease population dynamics; when population-level biophysical realism is needed without single-neuron resolution.

### 4. Data-Driven Models (Black-box, System Identification)

**Used in:** [[sources/chang-et-al-2020.md|Chang et al. 2020]] ([[concepts/volterra-model.md|Volterra series]]), [[sources/steffen-cannon-2025.md|Steffen & Cannon 2025]] ([[concepts/difference-of-convex-functions.md|DCNN]])
**Scale:** Population/whole brain (input-output relationship only)
**Fidelity:** Low mechanistic, high empirical — captures actual patient dynamics from recorded data
**State variables:** None explicit; mapping from stimulation $u$ to biomarker $y$
**Computational cost:** Depends on model complexity; inference is fast (especially DCNN via convex programming)
**When to use:** Clinical applications; when patient-specific response data is available; when mechanistic parameters are unknown or variable; when cross-patient generalizability is a goal.

## Model Selection Tradeoffs

| Property | HH | FHN | NMM | Data-driven |
|---|---|---|---|---|
| Mechanistic insight | +++ | ++ | ++ | — |
| Scalability to networks | — | +++ | ++ | ++ |
| Parameter identifiability | — | ++ | ++ | +++ |
| Clinical translatability | — | + | ++ | +++ |
| Control tractability | + | ++ | +++ | +++ |
| Handles nonlinearity | +++ | ++ | ++ | +++ |

## Synthesis: Convergence Toward Data-Driven Models

There is a clear trend in the field from mechanistic to data-driven models as the focus shifts from understanding mechanisms (single-neuron, in vitro) toward clinical translation (population, in vivo). Data-driven models sacrifice explanatory power for adaptability and deployability.

A key open question is whether **hybrid models** — data-driven predictors constrained to be consistent with known biophysical structure — can capture the best of both: the interpretability and generalization of mechanistic models with the patient-specificity of learned ones.

## Related Concepts

- [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley Model]] — Single-neuron mechanistic model.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Simplified excitable unit for large-scale network models.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — Population-level biophysical model.
- [[concepts/volterra-model.md|Volterra Model]] — Data-driven black-box approach via functional series.
- [[concepts/difference-of-convex-functions.md|Difference of Convex Functions]] — Structured data-driven model with optimization properties.
- [[concepts/human-connectome.md|Human Connectome]] — Structural connectivity used to couple FHN oscillators in whole-brain models.
