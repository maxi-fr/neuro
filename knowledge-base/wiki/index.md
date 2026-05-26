# LLM Wiki Index

## Sources

- [[sources/chang-et-al-2020.md|Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model]] — Closed-loop brain stimulation using MPC and Volterra models.
- [[sources/chouzouris-et-al-2021.md|Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators]] — Optimal nonlinear control for state-switching and synchronization in whole-brain networks.
- [[sources/froehlich-jezernik-2005.md|Feedback control of Hodgkin–Huxley nerve cell dynamics]] — Cascaded feedback control for direct state variable tracking in Hodgkin-Huxley neurons.
- [[sources/salfenmose-obermayer-2022.md|Nonlinear optimal control of a mean-field model of neural population dynamics]] — Analyzes efficient switching strategies in biophysically grounded models.
- [[sources/steffen-cannon-2025.md|Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease]] — Nonlinear data-driven MPC for beta-band suppression in PD.
- [[sources/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Master's thesis proposal on MPC, state estimation, and LLM integration.

## Concepts

- [[concepts/beta-band-oscillations.md|Beta-band Oscillations]] — Neural rhythms (13-30 Hz) used as a primary biomarker for Parkinson's Disease.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Automated adjustment of stimulation parameters based on real-time feedback.
- [[concepts/difference-of-convex-functions.md|Difference of Convex Functions and ICNN]] — DC functions and ICNN architecture for structured nonlinear modeling and efficient MPC optimization.
- [[concepts/feedback-linearization.md|Feedback Linearization]] — Technique used to handle nonlinearities by transforming them into linear forms.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Simplified neuron model for excitable systems and large-scale brain dynamics.
- [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley Model]] — Detailed biophysical model of action potential generation and ion channel dynamics.
- [[concepts/human-connectome.md|Human Connectome]] — Comprehensive map of structural neural connections in the human brain.
- [[concepts/llms-in-neurostimulation.md|LLMs in Neurostimulation]] — Integration of Large Language Models for treatment protocol design and multimodal feedback.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Advanced control method using optimization and predictive models.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — Macroscopic neurophysiological model for simulating EEG activity.
- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Optimization-based framework for steering nonlinear dynamical systems.
- [[concepts/parkinsons-disease.md|Parkinson's Disease]] — Progressive movement disorder targeted by deep brain stimulation.
- [[concepts/seizure-suppression.md|Seizure Suppression]] — Methods and strategies to reduce or eliminate epileptic activity.
- [[concepts/state-estimation.md|State Estimation]] — Inference of unmeasurable internal states for feedback control.
- [[concepts/volterra-model.md|Volterra Model]] — Functional series representation for nonlinear dynamic system identification.

## Synthesis

- [[synthesis/neurostimulation-approaches.md|Neurostimulation Approaches Synthesis]] — Standardized comparison of the five primary neurostimulation approaches across control objectives, controllers, and models.
- [[synthesis/control-strategy-landscape.md|Control Strategy Landscape]] — Compares the four control paradigms across all sources: feedback linearization, optimal nonlinear control, Volterra MPC, and DCNN Tube MPC.
- [[synthesis/brain-model-taxonomy.md|Brain Model Taxonomy]] — Compares the four brain model types (HH, FHN, NMM, data-driven) by scale, fidelity, and control tractability.
