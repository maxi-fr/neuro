# TODO

## General

## Knowledge Base

* Rename sources folder to summaries

## Implementations

<details>
<summary>Setup brain model</summary>

* Finding good parameters
* Running uncontrolled simulations

* maybe refactor tvb leadfield. Doesnst have to be this complicated surface projection stuff
* Figure out how inputs are modeled

</details>
<details>
<summary>Defining control objectives</summary>

* For parkinsons: desynchronization [[knowledge-base/Notes/control-objectives#1. Parkinson’s Disease (PD)|]]
* For epilepsy: state transition [[knowledge-base/Notes/control-objectives#2. Epilepsy|]]

Task is twofold: reference trajectory/cost function for MPC and brain model parameter set

* Need to find
* Make the different plants show the pathological states. As in create plots in which they are visible

</details>

<details>
<summary>Add marimo notebooks</summary>

* for state exploration
* simulation plots and so on

</details>

<details>
<summary>Add plotting utils</summary>

* for plotting eeg signals
* graphs
* bifurcation diagramms
* plots should save plotting data

</details>

<details>
<summary>Saving plots and data </summary>

* plots should be saved as images
* their data
* what function was used to generate them + parameters

</details>

<details>
<summary>What is AAL2 atlas</summary>

* what is HCP-80

</details>

## For first meeting (26.05)

So that we have a first direction for the project.

<details>
<summary>Research possible MPC formulations for impulsive systems</summary>

* Formulation as a continuous optimal control problem where pulse times and amplitudes are free and optimized over along the horizon.

* Continuous relaxations where sparsity/pulsing is achieved via regularization terms in the cost function (similar to what some of the attached papers do), or other relaxation strategies.

* Formulation as a mixed-integer problem with a fixed timing, where at each time point a discrete variable (0 - no pulse, 1 - pulse) and the continuous amplitude are optimized.

</details>

<details>
<summary>Research modeling for brain dynamics</summary>

* neural-mass models: Cowan-Wilson, Jansen-Rit

</details>

<details>
<summary>Research implementation frameworks for brain dynamics</summary>

* which python simulators for brain dynamics, neurostimulation, and EEG signals?
  * library by TU Berlin people - neurolib

</details>
