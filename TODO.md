# TODO list

* Test cost functions with JR model as the MPC predictor!!!

## Review refactor?

* remove all mentions of the old MPC/casadi implementation

* **Monday 24.08** Start predictor sweeps: Papa in München

## Refactors

* is mpc hinge spectral cost function tested?

* potential estimator refactor: move State Absorption/Priming (the "lift") out of the model into the estimator, so the model is pure `x_{k+1} = f(x_k, u_k)`. The lift would become a shared function used by both the runtime estimator and the evaluation free_run. Deferred for now: the model owns the window, the estimator emits one native measurement (y_k / o_k).

## Not urgent

* MPC metric: solver iterations, pred error along horizon

* quadratic tracking Costs drive the Observable to zero, but zero is not the healthy operating
  point. Jansen-Rit LFP `x2 - x3` sits at +1.5 mV with a healthy fluctuation std of 0.16 mV, so
  99% of `sum(y^2)` in healthy background is the operating point and only 1% is the dynamics the
  Cost is meant to shape. The controller spends its authority on a DC offset it cannot null under
  the amplitude bound, and it penalises healthy activity as hard as seizure activity. Costs should
  track a reference, `y - y_ref`, with `y_ref` the healthy per-region mean; this is general, not
  Jansen-Rit specific -- any Observable with a nonzero operating point has it. Note the hinge
  Costs do not: they score log excess over a healthy envelope and skip the DC bin.

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

## MPC package (trajopt)

* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here
* Add output function y = g(x). Would simplify cost fucntions etc.

## python-project-template

* Needs updating with the stuff from here
