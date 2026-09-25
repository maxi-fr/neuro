# TODO list

* clean up old notebooks and scripts

* is the LOG_FLOOR really necessary?

* add predictability of a number of seizing nodes to the predictability experiment

* try training waveform only on stft loss

* figure out this roast field projection matrix thing

## Refactors

* potential estimator refactor: move State Absorption/Priming (the "lift") out of the model into the estimator, so the model is pure `x_{k+1} = f(x_k, u_k)`. The lift would become a shared function used by both the runtime estimator and the evaluation free_run. Deferred for now: the model owns the window, the estimator emits one native measurement (y_k / o_k).

## Not urgent

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* investigate running controller slower than predictor model. Less decision variables but still prediction model with high resolution
  * Idea: dynamics faster than input can affect

* reservoir computer w. hopf nodes

## Other

## Simulate package

* allow for cross component config validation (like i've done here)
* package script for running simulations

## MPC package (trajopt)

* make fully yaml configurable - also cost functions, constraints, etc.

## python-project-template

* Needs updating with the stuff from here
