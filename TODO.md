# TODO

## Knowledge Base

* Rename sources folder to summaries

## Implementations

<details>
<summary>Setup brain model</summary>

* Finding good parameters
* Running uncontrolled simulations

</details>
<details>
<summary>Defining control objectives</summary>

* For parkinsons: desynchronization [[knowledge-base/raw/control-objectives#1. Parkinson’s Disease (PD)|]]
* For epilepsy: state transition [[knowledge-base/raw/control-objectives#2. Epilepsy|]]

</details>

## For first meeting (26.05)

Research:

* possible MPC formulations for impulsive systems
    1) Formulation as a continuous optimal control problem where pulse times and amplitudes are free and optimized over along the horizon.

    2) Continuous relaxations where sparsity/pulsing is achieved via regularization terms in the cost function (similar to what some of the attached papers do), or other relaxation strategies.

    3) Formulation as a mixed-integer problem with a fixed timing, where at each time point a discrete variable (0 - no pulse, 1 - pulse) and the continuous amplitude are optimized.
* modeling for brain dynamics
  * neural-mass models: Cowan-Wilson, Jansen-Rit
* implementation frameworks for brain dynamics
  * which python simulators for brain dynamics, neurostimulation, and EEG signals?
    * library by TU Berlin people - neurolib
    *

so that we have a first direction for the project.
