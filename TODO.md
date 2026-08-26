# TODO list

## Review refactor?

* The costs dont need the model i think because MPC class can do that now
* why is output() even extra?
* "build_observable_problem" etc. should be fully configurable via yaml not only through these helper.
* remove _apply_activation
* remove all mentions of the old MPC/casadi implementation
* how is the observable model different from the normal NN model? not at all. Arent they both just Autoregressive MLP at least on the MPC side yes. But the ingestion is still using u and y
* what is lift? What is transition?
* predictor protocol doesnt make sense: We assume completely different engines. One for training and one for inference/control
  * then prime(), etc. dont have to emit FloatArray, numpy completely removed
* remove _predictor_reference.py

* **Monday 24.08** Start predictor sweeps: Papa in München
  * is sweeping implemented for ObservableSpace?
  * what are all the hyperparameters?
  * how to split into different sweeps?
    * check again if curriculum is actually helpful in closed loop performance?
    * lr, weight decay, etc.
    * ny, nu, etc.
    * different losses
  * In what oder to run sweeps?
  * How to update sweeps based on the results of the prior?
  * what optuna score do the different sweeps use?

## Refactors

* is MLP residual?

* is mpc hinge spectral cost function tested?

* Refactor predictor + mpc setup to use trajopt package

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
