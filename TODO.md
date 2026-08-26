# TODO list

## Review refactor?

* remove all mentions of the old MPC/casadi implementation

* **Monday 24.08** Start predictor sweeps: Papa in München

## Refactors

* is mpc hinge spectral cost function tested?

* potential estimator refactor: move State Absorption/Priming (the "lift") out of the model into the estimator, so the model is pure `x_{k+1} = f(x_k, u_k)`. The lift would become a shared function used by both the runtime estimator and the evaluation free_run. Deferred for now: the model owns the window, the estimator emits one native measurement (y_k / o_k).

## Not urgent

* MPC metric: solver iterations, pred error along horizon

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* investigate running controller slower than predictor model. Less decision variables but still prediction model with high resolution
  * Idea: dynamics faster than input can affect

* reservoir computer w. hopf nodes

### Efficiency

* possible to remove torch.cat from AutoregressiveMLP.forward? for efficiency?
  > in other branch: perf/predictor-rollout-optimization

## Other

* write chapter on Jansen-Rit model
* figure out GPU training

## Simulate package

* allow for cross component config validation (like i've done here)

## MPC package

* investigate if it could be used well here
* investigate what parts could maybe be simplified, etc.
* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here

## python-project-template

* Needs updating with the stuff from here
