# TODO list

* **Monday 24.08** Start predictor sweeps: Papa in München

## Refactors

* MPC metric: solver iterations, pred error along horizon

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* investigate running controller slower than predictor model. Less decision variables but still prediction model with high resolution
  * Idea: dynamics faster than input can affect

* what about the effect of filter on spectrum losses/cost

* what about predicting the metric directly?

* reservoir computer w. hopf nodes

* is mpc hinge spectral cost function tested?

* check again if curriculum is actually helpful in closed loop performance

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
