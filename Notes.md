* Idea for presentations: marimo slides, manim slides with export to pptx.

## Questions
* What are bifurcation lines?
    * and bifurcation diagrams

---

## MPC for impulsive systems

### L1-norm relaxation
* Cost function L1-norm at least on the input
* Bang bang (on-off) output controller

## For presentation
I would say we focus on controlling networks of oscillators (wilson cowan, jansen ritt or FitzHugh-Magumo).
For general control tasks: state-switching, network synchronization/desynchronization
Then later we can maybe do specific examples: DBS - parkinsons or noninvasive neurostimulation for epilepsy

Questions:
* What oscillator model
* how are inputs modeled? continuous variable (voltage, injected current), pulses...
* How sparse are the inputs - i assume not every node has its own input (electrode)
* How to implement parameters changing - regarding system identification and adaptation for the MPC


So we use neurolib / i implement the stuff myself

## First meeting

### How are inputs modeled
* Model: continuous variables (membrane voltage)
* Real-world actuation: pulsed inputs

### Should the MPCs prediction model have the same structure as the Plant?
* prediction model should be simplified - for real time use
* First steps: how to distill plant model into lower complexity prediction model

### Brain Libraries to investigate
* Neurolib
* NEST
* TVB
