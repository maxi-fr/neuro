# TODO

## General

## Knowledge Base

## Implementations

<details>
<summary>fix configs</summary>

* base level config folder with all the different configs for the different scripts in subfolders

</details>

<details>
<summary>Idea: NN for JR identification</summary>

* "u", "x_k" -> MLP/RNN - "x_dot" -> heun -> "x_k+1" -> x_1-x_2 -> Linear -> "y_k+1"

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
