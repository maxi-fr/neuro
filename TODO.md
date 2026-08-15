# TODO list

## Refactors

* MPC metric: solver iterations, pred error along horizon
* training data

* move _spread_profile_from_trajectory to seizure

* Potentially: full rewrite
  * at least of the predictor part
  * move from jax to pytorch
  * list every feature and remove what is not necessary

## Other

* write chapter on Jansen-Rit model
* check out ROAST leadfield matrix: Ex8 is all zeros - that seems weird
* look into unified vocabulary - matt pocock skills

## Simulate package

* potential bug in logging: `MPCControllerLog gained n_iter and capped (defaulted, so the warm-up path and the linear MPC are untouched). capped is Maximum_Iterations_Exceeded specifically, so success / capped / other-failure is now a clean trichotomy. I avoided logging the status string: simulate's logger sizes each buffer from the first value's dtype, so a <U15 first status would silently truncate longer ones.`

## MPC package

* investigate if it could be used well here
* investigate what parts could maybe be simplified, etc.
* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here
